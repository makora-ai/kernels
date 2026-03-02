# SPDX-License-Identifier: Apache-2.0
import ttnn

# ======================================================================================
# nextafter(a, b) reference (per ttnn composite):
#   eps = hal::get_eps()
#   eps_gt = where(gt(a,b), add(a, eps), a)
#   out    = where(lt(a,b), subtract(a, eps), eps_gt)
#
# NOTE: This follows the provided reference exactly (even though it differs from IEEE
# nextafter semantics). This implementation assumes:
#   - a and b are same-shape
#   - TILE_LAYOUT
#   - bfloat16
# ======================================================================================

reader_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr             = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr             = get_common_arg_val<uint32_t>(1);
    const uint32_t base_tiles_per_core= get_common_arg_val<uint32_t>(2);
    const uint32_t extra_tile_range   = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_x        = get_common_arg_val<uint32_t>(4);
    const uint32_t grid_size_y        = get_common_arg_val<uint32_t>(5);

    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_index =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles =
        base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    const uint32_t tile_bytes = get_tile_size(cb_a);

    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    const auto a_acc = TensorAccessor(a_args, a_addr, tile_bytes);
    const auto b_acc = TensorAccessor(b_args, b_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t t = start_tile_index + i;

        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        const uint32_t a_l1 = get_write_ptr(cb_a);
        const uint32_t b_l1 = get_write_ptr(cb_b);

        noc_async_read_tile(t, a_acc, a_l1);
        noc_async_read_tile(t, b_acc, b_l1);
        noc_async_read_barrier();

        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
"""

compute_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#define SFPU_OP_FILL_INCLUDE 1
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/where.h"
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

    const uint32_t n_tiles =
        base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Init binary engine for sub_tiles (FPU)
    binary_op_init_common(cb_a, cb_b, cb_out);
    sub_tiles_init(cb_a, cb_b);

    // Init SFPU once
    init_sfpu(cb_a, cb_out);
    where_tile_init();
    fill_tile_init();
    add_binary_tile_init();
    sub_binary_tile_init();
    gtz_tile_init();
    ltz_tile_init();

    // NOTE: eps hardcoded for bf16 (matches typical hal::get_eps() usage for bf16 paths)
    // If your environment uses a different eps, pass it as a runtime arg and use fill_tile(...).
    constexpr float eps = 0.0078125f;

    // DST usage (bf16): 4 regs
    //  dst0: predicate (gt then lt)
    //  dst1: eps tile
    //  dst2: a+eps then eps_gt then final out
    //  dst3: a then a-eps
    constexpr uint32_t dst_pred = 0;
    constexpr uint32_t dst_eps  = 1;
    constexpr uint32_t dst_t    = 2;
    constexpr uint32_t dst_f    = 3;

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        // backpressure-safe: reserve output space before holding tile regs
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // ---- pred_gt = (a - b) > 0  ----
        sub_tiles(cb_a, cb_b, 0, 0, dst_pred);
        gtz_tile(dst_pred);

        // ---- eps tile ----
        fill_tile(dst_eps, eps);

        // ---- a_plus = a + eps ----
        copy_tile_init(cb_a);
        copy_tile(cb_a, 0, dst_t);
        add_binary_tile(dst_t, dst_eps, dst_t);  // dst_t = a + eps

        // ---- a (for false branch) ----
        copy_tile_init(cb_a);
        copy_tile(cb_a, 0, dst_f);

        // ---- eps_gt = where(pred_gt, a_plus, a) ----
        // output into dst_t (overwrite true input)
        where_tile(dst_pred, dst_t, dst_f, dst_t);

        // ---- pred_lt = (a - b) < 0 ----
        sub_tiles(cb_a, cb_b, 0, 0, dst_pred);
        ltz_tile(dst_pred);

        // ---- a_minus = a - eps (reuse dst_f holding a) ----
        sub_binary_tile(dst_f, dst_eps, dst_f);

        // ---- out = where(pred_lt, a_minus, eps_gt) ----
        // output into dst_t
        where_tile(dst_pred, dst_f, dst_t, dst_t);

        tile_regs_commit();

        // inputs no longer needed after commit (unpack/math complete)
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);

        tile_regs_wait();
        pack_tile(dst_t, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
    }
}
}  // namespace NAMESPACE
"""

writer_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr            = get_common_arg_val<uint32_t>(0);
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_index =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles =
        base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t t = start_tile_index + i;

        cb_wait_front(cb_out, 1);
        const uint32_t l1_r = get_read_ptr(cb_out);

        noc_async_write_tile(t, out_acc, l1_r);
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
        w = int(dims[0])
        ht, wt = 1, (w + 31) // 32
        nc = 1
        return nc * ht * wt

    h = int(dims[-2])
    w = int(dims[-1])
    ht, wt = (h + 31) // 32, (w + 31) // 32
    nc = 1
    for d in dims[:-2]:
        nc *= int(d)
    return nc * ht * wt


def host(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    # Manual nextafter kernel (bf16 + TILE_LAYOUT only)
    assert list(a.shape) == list(b.shape), "nextafter custom kernel expects same-shape inputs (no broadcast)."
    assert a.device() == b.device(), "Inputs must be on the same device."
    assert a.layout == ttnn.TILE_LAYOUT and b.layout == ttnn.TILE_LAYOUT, "Inputs must be TILE_LAYOUT."
    assert a.dtype == ttnn.bfloat16 and b.dtype == ttnn.bfloat16, "This kernel currently supports bfloat16 only."

    device = a.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(a.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    num_tiles = _num_tiles_from_shape(a.shape)
    if num_tiles == 0:
        return out

    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )

    num_cores = grid.x * grid.y
    base_tiles_per_core = num_tiles // num_cores
    extra_tile_range = num_tiles % num_cores

    # ---- Circular buffers ----
    tile_bytes = 32 * 32 * 2  # bf16 tile
    tiles_per_cb = 4
    cb_total = tiles_per_cb * tile_bytes

    cb_a = 0
    cb_b = 1
    cb_out = 16

    cb_descs = [
        ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(cb_a, ttnn.bfloat16, tile_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(cb_b, ttnn.bfloat16, tile_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(cb_out, ttnn.bfloat16, tile_bytes)],
        ),
    ]

    # ---- Compile-time args (TensorAccessorArgs) ----
    reader_ct = []
    reader_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())

    writer_ct = []
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    # ---- Runtime args ----
    empty_rt = [[[] for _ in range(grid.y)] for _ in range(grid.x)]

    reader_common_rt = [
        a.buffer_address(),
        b.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        grid.y,
    ]
    compute_common_rt = [
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        grid.y,
    ]
    writer_common_rt = [
        out.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        grid.y,
    ]

    reader_k = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=empty_rt,
        common_runtime_args=reader_common_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=[],
        runtime_args=empty_rt,
        common_runtime_args=compute_common_rt,
        config=ttnn.ComputeConfigDescriptor(),
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=writer_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=empty_rt,
        common_runtime_args=writer_common_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=cb_descs,
    )

    return ttnn.generic_op([a, b, out], program)