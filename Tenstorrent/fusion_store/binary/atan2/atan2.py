# SPDX-License-Identifier: Apache-2.0
import ttnn


# --------------------------------------------------------------------------------------
# Reader: stream y=a and x=b tiles from DRAM -> CBs
# CB0: y (a)
# CB1: x (b)
# --------------------------------------------------------------------------------------
reader_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);  // y
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);  // x

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(2);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(4);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(5);

    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_y = tt::CBIndex::c_0;
    constexpr uint32_t cb_x = tt::CBIndex::c_1;

    const uint32_t tile_bytes_y = get_tile_size(cb_y);
    const uint32_t tile_bytes_x = get_tile_size(cb_x);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, tile_bytes_y);

    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, tile_bytes_x);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t tile_id = start_tile_id + i;

        cb_reserve_back(cb_y, 1);
        cb_reserve_back(cb_x, 1);

        const uint32_t y_l1 = get_write_ptr(cb_y);
        const uint32_t x_l1 = get_write_ptr(cb_x);

        noc_async_read_page(tile_id, a, y_l1);
        noc_async_read_page(tile_id, b, x_l1);
        noc_async_read_barrier();

        cb_push_back(cb_y, 1);
        cb_push_back(cb_x, 1);
    }
}
"""


# --------------------------------------------------------------------------------------
# Compute: fused atan2(y, x)
#
# Implements:
#   atan = atan(y/x)
#   if x < 0 and y >= 0: atan += pi
#   if x < 0 and y <  0: atan -= pi
#   if x == 0: result = (y<0 ? -pi/2 : (y>0 ? pi/2 : 0))
#
# Notes:
# - Uses SFPU for recip/atan/comp/where and SFPU-binary for mul/add/sub on dst regs.
# - Uses mask-as-0/1 tiles; logical_and is implemented as mul on masks.
# --------------------------------------------------------------------------------------
compute_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/where.h"

// SFPU split includes (required for some unary ops)
#define SFPU_OP_RECIP_INCLUDE 1
#define SFPU_OP_FILL_INCLUDE 1
#define SFPU_OP_TRIG_FAMILY_INCLUDE 1
#define SFPU_OP_UNARY_COMP_INCLUDE 1
#define SFPU_OP_WHERE_INCLUDE 1
#define SFPU_OP_BINOP_WITH_SCALAR_INCLUDE 1
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_y   = tt::CBIndex::c_0;   // y (a)
    constexpr uint32_t cb_x   = tt::CBIndex::c_1;   // x (b)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Configure unpack/pack for SFPU-style kernels
    init_sfpu(cb_y, cb_out);

    // Unary SFPU inits
    recip_tile_init();
    atan_tile_init();

    ltz_tile_init();
    gtz_tile_init();
    eqz_tile_init();
    gez_tile_init();

    fill_tile_init();
    where_tile_init();

    // SFPU-binary inits
    mul_binary_tile_init();
    add_binary_tile_init();
    sub_binary_tile_init();

    // Scalar binop init (for mul_unary_tile)
    binop_with_scalar_tile_init();

    // float32 bit-pattern for pi (used by mul_unary_tile)
    constexpr uint32_t PI_U32 = 0x40490FDBu;  // 3.1415927410125732f (closest float32 pi)

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_y, 1);
        cb_wait_front(cb_x, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // ---- Compute atan(y/x) into dst0 ----
        // dst0 = y
        copy_tile_init(tt::CBIndex::c_0);
        copy_tile(cb_y, 0, 0);

        // dst3 = x ; recip(dst3)
        copy_tile_init(tt::CBIndex::c_1);
        copy_tile(cb_x, 0, 3);
        recip_tile(3);

        // dst0 = y * recip(x)
        mul_binary_tile(0, 3, 0);

        // dst0 = atan(dst0)
        atan_tile(0);

        // ---- Masks ----
        // dst4 = (x < 0)
        copy_tile_init(tt::CBIndex::c_1);
        copy_tile(cb_x, 0, 4);
        ltz_tile(4);

        // dst7 = (x == 0)
        copy_tile_init(tt::CBIndex::c_1);
        copy_tile(cb_x, 0, 7);
        eqz_tile(7);

        // dst5 = (y < 0)
        copy_tile_init(tt::CBIndex::c_0);
        copy_tile(cb_y, 0, 5);
        ltz_tile(5);

        // dst6 = (y > 0)
        copy_tile_init(tt::CBIndex::c_0);
        copy_tile(cb_y, 0, 6);
        gtz_tile(6);

        // dst1 = (y >= 0)
        copy_tile_init(tt::CBIndex::c_0);
        copy_tile(cb_y, 0, 1);
        gez_tile(1);

        // ---- Quadrant correction for x < 0 ----
        // mask_bltz (x<0 && y<0) -> dst2 = dst4 * dst5
        // mask_bgte (x<0 && y>=0) -> dst4 = dst4 * dst1  (overwrite dst4)
        mul_binary_tile(4, 5, 2);
        mul_binary_tile(4, 1, 4);

        // dst4 *= pi ; dst2 *= pi
        mul_unary_tile(4, PI_U32);
        mul_unary_tile(2, PI_U32);

        // dst0 += dst4 ; dst0 -= dst2
        add_binary_tile(0, 4, 0);
        sub_binary_tile(0, 2, 0);

        // ---- Special-case x == 0 ----
        // special = (y<0 ? -pi/2 : (y>0 ? pi/2 : 0))
        // build into dst3
        fill_tile(3, 0.0f);

        // if y>0 -> pi/2
        fill_tile(1, 1.5707963267948966f);
        where_tile(6, 1, 3, 3);

        // if y<0 -> -pi/2
        fill_tile(1, -1.5707963267948966f);
        where_tile(5, 1, 3, 3);

        // if x==0 -> choose special else corrected atan
        where_tile(7, 3, 0, 0);

        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_y, 1);
        cb_pop_front(cb_x, 1);
        cb_push_back(cb_out, 1);
    }
}
}  // namespace NAMESPACE
"""


# --------------------------------------------------------------------------------------
# Writer: stream CB16 -> DRAM
# --------------------------------------------------------------------------------------
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

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t tile_id = start_tile_id + i;

        cb_wait_front(cb_out, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_out);

        noc_async_write_page(tile_id, out, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
"""


def _num_tiles_from_shape(shape) -> int:
    dims = list(shape)
    if len(dims) == 0:
        return 0
    if len(dims) == 1:
        h, w = 1, int(dims[0])
        ht, wt = 1, (w + 31) // 32
        nc = 1
    else:
        h, w = int(dims[-2]), int(dims[-1])
        ht, wt = (h + 31) // 32, (w + 31) // 32
        nc = 1
        for d in dims[:-2]:
            nc *= int(d)
    return nc * ht * wt


def host(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    # NOTE: This kernel is written for TILE_LAYOUT tensors of bf16 (typical for eltwise).
    assert a.shape == b.shape, "atan2 kernel currently requires a and b to have identical shapes"
    assert a.layout == ttnn.TILE_LAYOUT and b.layout == ttnn.TILE_LAYOUT, "atan2 kernel requires TILE_LAYOUT inputs"
    assert a.dtype == ttnn.bfloat16 and b.dtype == ttnn.bfloat16, "atan2 kernel currently supports bfloat16 inputs only"

    device = a.device()

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(a.shape),
        a.dtype,
        ttnn.TILE_LAYOUT,
        device,
    )

    num_tiles = _num_tiles_from_shape(a.shape)
    if num_tiles == 0:
        return out

    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y
    base_tiles_per_core = num_tiles // total_cores
    extra_tile_range = num_tiles % total_cores

    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    # CB sizing (bf16 tile = 32*32*2 = 2048 bytes)
    tile_size_bytes = 32 * 32 * 2
    tiles_per_cb = 2
    cb_total_bytes = tiles_per_cb * tile_size_bytes

    cb_y_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=tile_size_bytes)
        ],
    )
    cb_x_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=tile_size_bytes)
        ],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=tile_size_bytes)
        ],
    )

    reader_ct_args = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    reader_ct_args.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())

    writer_ct_args = ttnn.TensorAccessorArgs(out).get_compile_time_args()
    compute_ct_args = []

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
        config=ttnn.ComputeConfigDescriptor(),  # bf16 dst-reg mode
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

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[cb_y_desc, cb_x_desc, cb_out_desc],
    )

    return ttnn.generic_op([a, b, out], program_descriptor)