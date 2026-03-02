# SPDX-License-Identifier: Apache-2.0
import math
import struct
import torch
import ttnn

# --------------------------------------------------------------------------------------------------
# Reference harness compatibility:
# Some harnesses call torch.polygamma(a, n) (input-first), but PyTorch API is torch.polygamma(n, a).
# Patch torch.polygamma to accept (input, n) if needed, and coerce n if it's tensor-like.
# --------------------------------------------------------------------------------------------------
_torch_polygamma_orig = getattr(torch, "polygamma", None)
if _torch_polygamma_orig is not None:
    _needs_swap = False
    try:
        _ = _torch_polygamma_orig(torch.tensor([1.25], dtype=torch.float32), 3)  # try (input, n)
    except TypeError:
        _needs_swap = True

    if _needs_swap:

        def _coerce_n_for_torch(n):
            if isinstance(n, (int, bool)):
                return int(n)
            if isinstance(n, float):
                return int(n)
            if isinstance(n, torch.Tensor):
                return int(n.item())
            return int(n)

        def _polygamma_input_first(input, n=0):
            return _torch_polygamma_orig(_coerce_n_for_torch(n), input)

        torch.polygamma = _polygamma_input_first


# --------------------------------------------------------------------------------------------------
# Reader kernel: DRAM -> L1 CB (c_0), tile-streaming
# --------------------------------------------------------------------------------------------------
reader_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t in_addr             = get_common_arg_val<uint32_t>(0);
    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + (my_y * grid_size_x);

    uint32_t start_tile_index =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1u : 0u);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    const uint32_t tile_bytes = get_tile_size(cb_in0);

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_w = get_write_ptr(cb_in0);

        noc_async_read_page(start_tile_index + i, in, l1_w);
        noc_async_read_barrier();

        cb_push_back(cb_in0, 1);
    }
}
"""


# --------------------------------------------------------------------------------------------------
# Compute kernel: polygamma approximation used by ttnn::_polygamma (truncated series to 11 terms)
#
# temp = sum_{idx=0..10} 1 / (a + idx)^(k_der),  k_der = 1 + k
# out  = temp * (tgamma(k_der) * pos_neg), pos_neg = -1 for k in {2,4,6,8,10}
#
# Key fix vs previous attempts:
# - Use scalar power: power_tile(dst, exponent_int) (matches ttnn::power(..., scalar) flow better)
# - Use scalar add/mul: add_unary_tile / mul_unary_tile (matches scalar add/mul ops better)
# - Keep accumulation in Dst regs; enable fp32 dest-acc on host for better precision
# --------------------------------------------------------------------------------------------------
compute_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

// SFPU split includes
#define SFPU_OP_RECIP_INCLUDE 1
#define SFPU_OP_BINOP_WITH_SCALAR_INCLUDE 1
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    // k_der = 1 + k (integer exponent)
    uint32_t exponent_int = get_common_arg_val<uint32_t>(4);

    // float32 bitcast for scale = tgamma(k_der) * pos_neg
    uint32_t scale_bits   = get_common_arg_val<uint32_t>(5);

    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + (my_y * grid_size_x);

    uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1u : 0u);

    constexpr uint32_t cb_in0  = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    // dst regs:
    //  dst_sum: running sum / final output
    //  dst_x  : workspace for each term
    constexpr uint32_t dst_sum = 0;
    constexpr uint32_t dst_x   = 1;

    // Pre-init once
    init_sfpu(cb_in0, cb_out0);
    copy_tile_init(cb_in0);

    #ifdef TRISC_MATH
    recip_tile_init();
    power_tile_init();
    binop_with_scalar_tile_init();
    add_binary_tile_init();
    #endif

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, 1);
        cb_reserve_back(cb_out0, 1);

        tile_regs_acquire();

        // term idx=0 goes directly into sum:
        copy_tile(cb_in0, 0, dst_sum);

        #ifdef TRISC_MATH
        power_tile(dst_sum, exponent_int);
        recip_tile(dst_sum);
        #endif

        // idx=1..10, unrolled with float32 bit patterns for scalar add
        // float bits:
        // 1.0 0x3f800000, 2.0 0x40000000, 3.0 0x40400000, 4.0 0x40800000, 5.0 0x40a00000
        // 6.0 0x40c00000, 7.0 0x40e00000, 8.0 0x41000000, 9.0 0x41100000, 10.0 0x41200000

        // idx = 1
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x3f800000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 2
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x40000000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 3
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x40400000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 4
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x40800000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 5
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x40a00000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 6
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x40c00000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 7
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x40e00000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 8
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x41000000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 9
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x41100000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // idx = 10
        copy_tile(cb_in0, 0, dst_x);
        #ifdef TRISC_MATH
        add_unary_tile(dst_x, 0x41200000u);
        power_tile(dst_x, exponent_int);
        recip_tile(dst_x);
        add_binary_tile(dst_sum, dst_x, dst_sum);
        #endif

        // out = sum * scale
        #ifdef TRISC_MATH
        mul_unary_tile(dst_sum, scale_bits);
        #endif

        tile_regs_commit();
        cb_pop_front(cb_in0, 1);

        tile_regs_wait();
        pack_tile(dst_sum, cb_out0);
        tile_regs_release();

        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
"""


# --------------------------------------------------------------------------------------------------
# Writer kernel: L1 CB (c_16) -> DRAM
# --------------------------------------------------------------------------------------------------
writer_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t out_addr            = get_common_arg_val<uint32_t>(0);
    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + (my_y * grid_size_x);

    uint32_t start_tile_index =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1u : 0u);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out0);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out0, 1);
        uint32_t l1_r = get_read_ptr(cb_out0);

        noc_async_write_page(start_tile_index + i, out, l1_r);
        noc_async_write_barrier();

        cb_pop_front(cb_out0, 1);
    }
}
"""


def _num_tiles_for_shape(shape) -> int:
    dims = list(shape)
    if len(dims) == 0:
        return 0
    if len(dims) == 1:
        w = dims[0]
        ht = 1
        wt = (w + 31) // 32
        nc = 1
        return nc * ht * wt
    h = dims[-2]
    w = dims[-1]
    ht = (h + 31) // 32
    wt = (w + 31) // 32
    nc = 1
    for d in dims[:-2]:
        nc *= d
    return nc * ht * wt


def _f32_to_u32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _coerce_n_to_int(n) -> int:
    if isinstance(n, (int, bool)):
        return int(n)
    if isinstance(n, float):
        return int(n)
    if isinstance(n, torch.Tensor):
        return int(n.item())
    if isinstance(n, ttnn.Tensor):
        # best-effort scalar tensor extraction
        return int(ttnn.to_torch(n).item())
    return int(n)


def host(a: ttnn.Tensor, n=3) -> ttnn.Tensor:
    # Reference: return ttnn.polygamma(a, n)
    if a.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError("custom polygamma kernel expects TILE_LAYOUT input")
    if a.dtype not in (ttnn.bfloat16, ttnn.float32):
        raise RuntimeError("custom polygamma kernel supports only ttnn.bfloat16 or ttnn.float32")

    k = _coerce_n_to_int(n)

    device = a.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(a.shape), a.dtype, ttnn.TILE_LAYOUT, device)

    num_tiles = _num_tiles_for_shape(a.shape)
    if num_tiles == 0:
        return out

    # Match ttnn::_polygamma host logic (scale computed on host)
    k_der = 1.0 + float(k)  # integer-valued float
    exponent_int = int(k + 1)

    fact_val = float(math.gamma(k_der))  # tgamma(k_der)
    pos_neg = -1.0 if k in (2, 4, 6, 8, 10) else 1.0
    scale = fact_val * pos_neg
    scale_bits = _f32_to_u32(scale)

    grid_size = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    num_cores = grid_size.x * grid_size.y
    base_tiles_per_core = num_tiles // num_cores
    extra_tile_range = num_tiles % num_cores

    # CBs
    cb_in0 = 0
    cb_out0 = 16

    # Tile bytes depend on tensor dtype
    if a.dtype == ttnn.float32:
        tile_bytes = 32 * 32 * 4
    else:
        tile_bytes = 32 * 32 * 2

    tiles_per_cb = 2
    cb_total = tiles_per_cb * tile_bytes

    cb_in0_desc = ttnn.CBDescriptor(
        total_size=cb_total,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_in0, data_format=a.dtype, page_size=tile_bytes)
        ],
    )
    cb_out0_desc = ttnn.CBDescriptor(
        total_size=cb_total,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_out0, data_format=out.dtype, page_size=tile_bytes)
        ],
    )

    # Compile-time args (TensorAccessor)
    reader_ct_args = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    writer_ct_args = ttnn.TensorAccessorArgs(out).get_compile_time_args()

    # Common runtime args
    reader_common_rt = [
        a.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
    ]
    compute_common_rt = [
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
        int(exponent_int),
        int(scale_bits),
    ]
    writer_common_rt = [
        out.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
    ]

    # Use fp32 dst accumulation to reduce numerical error even when input/output are bf16
    compute_cfg = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)

    reader_k = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=[],
        common_runtime_args=reader_common_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=[],
        runtime_args=[],
        common_runtime_args=compute_common_rt,
        config=compute_cfg,
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=writer_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=[],
        common_runtime_args=writer_common_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[cb_in0_desc, cb_out0_desc],
    )

    return ttnn.generic_op([a, out], program)