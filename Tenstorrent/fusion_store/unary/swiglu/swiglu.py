# SPDX-License-Identifier: Apache-2.0
import ttnn


# --------------------------------------------------------------------------------------
# reference(a) = ttnn.swiglu(a)
#
# TTNN swiglu (device) does:
#   ab = split_tensor_for_glu(a, dim=3)   # split last dim (W) into 2 equal halves
#   swish_b = silu(ab[1])
#   out = multiply(ab[0], swish_b)
#
# This fully-manual kernel implements:
#   out[..., j] = a[..., j] * silu(a[..., j + W_out])
#
# IMPORTANT RESTRICTION (same as previous attempt):
#   This implementation assumes the split is TILE-ALIGNED along W:
#     W_in must be divisible by 64  (so W_out divisible by 32, and Wt_in == 2*Wt_out).
#   This avoids intra-tile column shifting/gather.
#
# Fix vs previous attempt:
#   - init_sfpu() compile error fixed by including eltwise_unary.h (init_sfpu declaration).
#   - Use noc_async_{read,write}_page (preferred) instead of deprecated *_tile.
# --------------------------------------------------------------------------------------


reader_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Common args
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    const uint32_t Ht      = get_common_arg_val<uint32_t>(5);
    const uint32_t Wt_in   = get_common_arg_val<uint32_t>(6);
    const uint32_t Wt_out  = get_common_arg_val<uint32_t>(7);

    // Core/work derivation
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    (void)grid_size_y;

    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    // CBs
    constexpr uint32_t cb_a = tt::CBIndex::c_0;  // A tiles (first half)
    constexpr uint32_t cb_b = tt::CBIndex::c_1;  // B tiles (second half)

    const uint32_t tile_bytes = get_tile_size(cb_a);

    // Input accessor
    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in_acc = TensorAccessor(in_args, in_addr, tile_bytes);

    const uint32_t HtWt_out = Ht * Wt_out;
    const uint32_t HtWt_in  = Ht * Wt_in;

    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t g_out = start_tile_id + i;

        // g_out is in [0, NC*Ht*Wt_out)
        const uint32_t nc  = g_out / HtWt_out;
        const uint32_t rem = g_out - nc * HtWt_out;
        const uint32_t tr  = rem / Wt_out;
        const uint32_t tc  = rem - tr * Wt_out;

        // Map output tile -> input tiles in each half (tile-aligned split)
        const uint32_t g_in0 = nc * HtWt_in + tr * Wt_in + tc;        // A tile
        const uint32_t g_in1 = g_in0 + Wt_out;                        // B tile

        cb_reserve_back(cb_a, onetile);
        cb_reserve_back(cb_b, onetile);

        const uint32_t l1_a = get_write_ptr(cb_a);
        const uint32_t l1_b = get_write_ptr(cb_b);

        noc_async_read_page(g_in0, in_acc, l1_a);
        noc_async_read_page(g_in1, in_acc, l1_b);
        noc_async_read_barrier();

        cb_push_back(cb_a, onetile);
        cb_push_back(cb_b, onetile);
    }
}
"""


compute_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Common args
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    // Core/work derivation
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    (void)grid_size_y;

    const uint32_t core_idx = my_x + my_y * grid_size_x;
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Configure SFPU pipeline once
    init_sfpu(cb_b, cb_out);

    // Init SFPU ops once
    silu_tile_init();
    mul_binary_tile_init();

    constexpr uint32_t onetile = 1;

    // Dst regs:
    //   dst0 := B -> silu(B) -> output
    //   dst1 := A
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_a, onetile);
        cb_wait_front(cb_b, onetile);
        cb_reserve_back(cb_out, onetile);

        tile_regs_acquire();

        // dst0 = B; dst0 = silu(dst0)
        copy_tile(cb_b, 0, 0);
        silu_tile(0);

        // dst1 = A
        copy_tile(cb_a, 0, 1);

        // dst0 = dst1 * dst0
        mul_binary_tile(1, 0, 0);

        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_a, onetile);
        cb_pop_front(cb_b, onetile);
        cb_push_back(cb_out, onetile);
    }
}
}  // namespace NAMESPACE
"""


writer_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Common args
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    // Core/work derivation
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    (void)grid_size_y;

    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out, onetile);
        const uint32_t l1_read = get_read_ptr(cb_out);

        noc_async_write_page(start_tile_id + i, out_acc, l1_read);
        noc_async_write_barrier();

        cb_pop_front(cb_out, onetile);
    }
}
"""


def _prod(xs):
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _tile_hw_for_shape(shape):
    dims = list(shape)
    if len(dims) < 2:
        raise RuntimeError(f"swiglu kernel expects rank>=2 tiled tensor, got shape={shape}")
    H = int(dims[-2])
    W = int(dims[-1])
    Ht = (H + 31) // 32
    Wt = (W + 31) // 32
    NC = _prod(dims[:-2])
    return NC, H, W, Ht, Wt


def host(a: ttnn.Tensor) -> ttnn.Tensor:
    # ---- Validate ----
    if a.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError("swiglu: this manual kernel expects TILE_LAYOUT input")

    if a.dtype not in (ttnn.bfloat16, ttnn.float32):
        raise RuntimeError(f"swiglu: unsupported dtype {a.dtype}; expected bfloat16 or float32")

    device = a.device()

    # Input is split along last dim (W) into 2 halves
    NC, H, W_in, Ht, Wt_in = _tile_hw_for_shape(a.shape)

    if W_in % 2 != 0:
        raise RuntimeError(f"swiglu: last dim must be even, got W={W_in}")

    W_out = W_in // 2
    Wt_out = (W_out + 31) // 32

    # Enforce tile-aligned split: Wt_in must be exactly 2*Wt_out (no intra-tile slicing)
    if Wt_in != 2 * Wt_out:
        raise RuntimeError(
            f"swiglu: requires tile-aligned split along W. Got W={W_in} -> Wt_in={Wt_in}, "
            f"W_out={W_out} -> Wt_out={Wt_out} (need Wt_in == 2*Wt_out)."
        )

    out_shape = list(a.shape)
    out_shape[-1] = W_out

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        a.dtype,
        ttnn.TILE_LAYOUT,
        device,
    )

    num_tiles_out = NC * Ht * Wt_out
    if num_tiles_out == 0:
        return out

    # ---- Core grid/work split (kernel-side distribution) ----
    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y
    base_tiles_per_core = num_tiles_out // total_cores
    extra_tile_range = num_tiles_out % total_cores

    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    # ---- CB sizing ----
    if a.dtype == ttnn.bfloat16:
        tile_bytes = 32 * 32 * 2
        compute_cfg = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False)
    else:
        tile_bytes = 32 * 32 * 4
        compute_cfg = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)

    tiles_per_cb = 4
    cb_total_size = tiles_per_cb * tile_bytes

    cb_a, cb_b, cb_out = 0, 1, 16

    cb_a_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_a, data_format=a.dtype, page_size=tile_bytes)
        ],
    )
    cb_b_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_b, data_format=a.dtype, page_size=tile_bytes)
        ],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_out, data_format=a.dtype, page_size=tile_bytes)
        ],
    )

    # ---- Compile-time args (TensorAccessor) ----
    reader_ct_args = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    writer_ct_args = ttnn.TensorAccessorArgs(out).get_compile_time_args()

    # ---- Common runtime args ----
    reader_common_rt_args = [
        a.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
        Ht,
        Wt_in,
        Wt_out,
    ]

    compute_common_rt_args = [
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
    ]

    writer_common_rt_args = [
        out.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
    ]

    # ---- Kernel descriptors ----
    reader_k = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=[],
        common_runtime_args=reader_common_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=[],
        runtime_args=[],
        common_runtime_args=compute_common_rt_args,
        config=compute_cfg,
    )

    writer_k = ttnn.KernelDescriptor(
        kernel_source=writer_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=[],
        common_runtime_args=writer_common_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[cb_a_desc, cb_b_desc, cb_out_desc],
    )

    return ttnn.generic_op([a, out], program)