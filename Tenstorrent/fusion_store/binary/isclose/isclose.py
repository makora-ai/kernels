# SPDX-License-Identifier: Apache-2.0
import struct
import ttnn


# -----------------------------------------------------------------------------
# Reader: streams A tiles -> CB0, B tiles -> CB1
# -----------------------------------------------------------------------------
reader_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Common args
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t in1_addr = get_common_arg_val<uint32_t>(1);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(2);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(4);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(5);

    // Core/work mapping
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    const uint32_t tile_bytes0 = get_tile_size(cb_in0);
    const uint32_t tile_bytes1 = get_tile_size(cb_in1);

    constexpr auto in0_args = TensorAccessorArgs<0>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes0);
    const auto in1 = TensorAccessor(in1_args, in1_addr, tile_bytes1);

    constexpr uint32_t onetile = 1;

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t g = start_tile_id + i;

        cb_reserve_back(cb_in0, onetile);
        cb_reserve_back(cb_in1, onetile);

        const uint32_t l1_w0 = get_write_ptr(cb_in0);
        const uint32_t l1_w1 = get_write_ptr(cb_in1);

        noc_async_read_page(g, in0, l1_w0);
        noc_async_read_page(g, in1, l1_w1);

        noc_async_read_barrier();

        cb_push_back(cb_in0, onetile);
        cb_push_back(cb_in1, onetile);
    }
}
"""


# -----------------------------------------------------------------------------
# Compute: fused isclose (default semantics from ttnn::_isclose)
#   if (!equal_nan):
#     a = where(isnan(a), 1.0, a)
#     b = where(isnan(b), 0.0, b)
#   lhs = abs(a - b)
#   rhs = abs(b) * rtol + atol
#   out = (lhs <= rhs) as 1.0/0.0
#
# Implementation detail:
#   out = unary_le(lhs - rhs, 0.0)
# -----------------------------------------------------------------------------
compute_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

// SFPU split include macros (must be defined BEFORE including sfpu_split_includes.h)
#define SFPU_OP_ISINF_ISNAN_INCLUDE 1
#define SFPU_OP_WHERE_INCLUDE 1
#define SFPU_OP_UNARY_COMP_INCLUDE 1
#define SFPU_OP_BINOP_WITH_SCALAR_INCLUDE 1
#define SFPU_OP_ABS_INCLUDE 1
#define SFPU_OP_FILL_INCLUDE 1

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Common args
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    const uint32_t rtol_bits   = get_common_arg_val<uint32_t>(4);  // fp32 bitcast
    const uint32_t atol_bits   = get_common_arg_val<uint32_t>(5);  // fp32 bitcast
    const uint32_t equal_nan_u = get_common_arg_val<uint32_t>(6);  // 0/1

    // Core/work mapping
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr auto cb_a   = tt::CBIndex::c_0;
    constexpr auto cb_b   = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // SFPU init / op inits (keep OUTSIDE loop)
    init_sfpu(cb_a, cb_out);

    // SFPU/binary-sfpu op init
    sub_binary_tile_init();
    abs_tile_init();
    isnan_tile_init();
    where_tile_init();
    fill_tile_init();
    binop_with_scalar_tile_init();
    unary_le_tile_init();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t fzero_bits = 0x00000000u;  // fp32 bitcast(0.0f)

    // DST register allocation (bf16: up to 8 regs with default config)
    // dst0: maskA (if !equal_nan)
    // dst1: valA (sanitized)
    // dst2: const 1.0 (if !equal_nan)
    // dst3: valB (sanitized, later abs+rtol+atol rhs)
    // dst4: maskB (if !equal_nan)
    // dst5: const 0.0 (if !equal_nan)
    // dst6: diff/absdiff/compare_work/result

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_a, onetile);
        cb_wait_front(cb_b, onetile);
        cb_reserve_back(cb_out, onetile);

        tile_regs_acquire();

        if (equal_nan_u == 0) {
            // ---- A: mask + value ----
            copy_tile_init(static_cast<uint32_t>(cb_a));
            copy_tile(cb_a, 0, 1);  // A -> dst1 (value)
            copy_tile(cb_a, 0, 0);  // A -> dst0 (for isnan -> mask)
            isnan_tile(0);          // dst0 = isnan(A) mask (1.0/0.0)
            fill_tile(2, 1.0f);     // dst2 = 1.0
            where_tile(0, 2, 1, 1); // dst1 = where(maskA, 1.0, A)

            // ---- B: mask + value ----
            copy_tile_init(static_cast<uint32_t>(cb_b));
            copy_tile(cb_b, 0, 3);  // B -> dst3 (value)
            copy_tile(cb_b, 0, 4);  // B -> dst4 (for isnan -> mask)
            isnan_tile(4);          // dst4 = isnan(B) mask
            fill_tile(5, 0.0f);     // dst5 = 0.0
            where_tile(4, 5, 3, 3); // dst3 = where(maskB, 0.0, B)
        } else {
            // equal_nan path in ttnn::_isclose just skips nan-rewriting
            copy_tile_init(static_cast<uint32_t>(cb_a));
            copy_tile(cb_a, 0, 1);  // A -> dst1
            copy_tile_init(static_cast<uint32_t>(cb_b));
            copy_tile(cb_b, 0, 3);  // B -> dst3
        }

        // lhs = abs(A - B) -> dst6
        sub_binary_tile(1, 3, 6);
        abs_tile(6);

        // rhs = abs(B) * rtol + atol -> dst3
        abs_tile(3);
        mul_unary_tile(3, rtol_bits);
        add_unary_tile(3, atol_bits);

        // out = (lhs <= rhs) == ((lhs - rhs) <= 0)
        sub_binary_tile(6, 3, 6);
        unary_le_tile(6, fzero_bits);  // dst6 is now 1.0/0.0

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(6, cb_out);

        tile_regs_release();

        cb_pop_front(cb_a, onetile);
        cb_pop_front(cb_b, onetile);
        cb_push_back(cb_out, onetile);
    }
}
}  // namespace NAMESPACE
"""


# -----------------------------------------------------------------------------
# Writer: drains CB16 -> output
# -----------------------------------------------------------------------------
writer_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    constexpr uint32_t onetile = 1;

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t g = start_tile_id + i;

        cb_wait_front(cb_out, onetile);
        const uint32_t l1_r = get_read_ptr(cb_out);

        noc_async_write_page(g, out, l1_r);
        noc_async_write_barrier();

        cb_pop_front(cb_out, onetile);
    }
}
"""


def _bitcast_fp32_to_u32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def host(a: ttnn.Tensor, b: ttnn.Tensor, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> ttnn.Tensor:
    # NOTE: This custom kernel assumes:
    # - a and b have the same shape
    # - a and b are TILE_LAYOUT
    # - dtype is bf16 (recommended). float32 is supported by setting fp32_dest_acc_en.
    device = a.device()

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(a.shape),
        a.dtype,
        ttnn.TILE_LAYOUT,
        device,
    )

    # Compute number of tiles for a tile-layout tensor
    dims = list(a.shape)
    if len(dims) == 0:
        return out

    if len(dims) == 1:
        H, W = 1, int(dims[0])
        Ht, Wt = 1, (W + 31) // 32
        NC = 1
    else:
        H, W = int(dims[-2]), int(dims[-1])
        Ht, Wt = (H + 31) // 32, (W + 31) // 32
        NC = 1
        for d in dims[:-2]:
            NC *= int(d)

    num_tiles = NC * Ht * Wt
    if num_tiles == 0:
        return out

    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y
    base_tiles_per_core = num_tiles // total_cores
    extra_tile_range = num_tiles % total_cores

    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    # Tile/page sizing
    if a.dtype == ttnn.bfloat16:
        tile_size_bytes = 32 * 32 * 2
        fp32_dest_acc_en = False
        dst_full_sync_en = False
    elif a.dtype == ttnn.float32:
        tile_size_bytes = 32 * 32 * 4
        # Need fp32 dst regs; also need enough dst regs for our 7-register schedule
        fp32_dest_acc_en = True
        dst_full_sync_en = True
    else:
        raise RuntimeError(f"Unsupported dtype for isclose kernel: {a.dtype}")

    tiles_per_cb = 2
    cb_total_bytes = tiles_per_cb * tile_size_bytes

    cb_a_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=a.dtype, page_size=tile_size_bytes)],
    )
    cb_b_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=a.dtype, page_size=tile_size_bytes)],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=a.dtype, page_size=tile_size_bytes)],
    )

    # Compile-time args (TensorAccessor)
    reader_ct_args = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    reader_ct_args.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
    writer_ct_args = ttnn.TensorAccessorArgs(out).get_compile_time_args()
    compute_ct_args = []

    # Common runtime args
    reader_common_rt_args = [
        a.buffer_address(),
        b.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
    ]

    compute_common_rt_args = [
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
        _bitcast_fp32_to_u32(rtol),
        _bitcast_fp32_to_u32(atol),
        1 if bool(equal_nan) else 0,
    ]

    writer_common_rt_args = [
        out.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
    ]

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
        compile_time_args=compute_ct_args,
        runtime_args=[],
        common_runtime_args=compute_common_rt_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en),
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

    return ttnn.generic_op([a, b, out], program)