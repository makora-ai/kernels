# SPDX-License-Identifier: Apache-2.0
import ttnn


# -----------------------------------------------------------------------------
# Reader: DRAM -> CB0
# -----------------------------------------------------------------------------
reader_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Common runtime args
    uint32_t src_addr            = get_common_arg_val<uint32_t>(0);
    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    uint32_t num_cores_used      = get_common_arg_val<uint32_t>(4);

    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + (my_y * grid_size_x);

    if (core_idx >= num_cores_used) {
        return;
    }

    uint32_t start_tile_index =
        core_idx * base_tiles_per_core + ((core_idx < extra_tile_range) ? core_idx : extra_tile_range);
    uint32_t n_tiles = base_tiles_per_core + ((core_idx < extra_tile_range) ? 1 : 0);

    constexpr uint32_t cb_in0 = 0;
    const uint32_t tile_bytes = get_tile_size(cb_in0);

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        noc_async_read_tile(start_tile_index + i, src, l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_in0, 1);
    }
}
"""


# -----------------------------------------------------------------------------
# Compute: digamma approximation used by ttnn composite
#
# Reference (from ttnn/cpp/.../unary_composite_op.cpp):
#   t_log_out = log(z)
#   out = 0.5*(1/z)
#   tmp = (1/z)^2
#   out -= (1/12) * tmp
#   tmp *= tmp0; out += (1/120) * tmp
#   tmp *= tmp0; out -= (1/252) * tmp
#   tmp *= tmp0; out += (1/240) * tmp
#   tmp *= tmp0; out -= (1/132) * tmp
#   tmp *= tmp0; out += (691/32760) * tmp
#   tmp *= tmp0; out -= (1/12) * tmp
#   return t_log_out - out
# -----------------------------------------------------------------------------
compute_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

// SFPU unary ops (macro-gated split includes)
#define SFPU_OP_LOG_INCLUDE   1
#define SFPU_OP_RECIP_INCLUDE 1
#define SFPU_OP_FILL_INCLUDE  1
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Common runtime args
    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    uint32_t num_cores_used      = get_common_arg_val<uint32_t>(3);

    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + (my_y * grid_size_x);

    if (core_idx >= num_cores_used) {
        return;
    }

    uint32_t n_tiles = base_tiles_per_core + ((core_idx < extra_tile_range) ? 1 : 0);

    constexpr uint32_t cb_in0  = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    // One-time init (MUST be outside the per-tile loop)
    init_sfpu(cb_in0, cb_out0);
    copy_tile_init(tt::CBIndex::c_0);

    // Unary SFPU inits
    log_tile_init();
    recip_tile_init();
    fill_tile_init();

    // SFPU binary inits
    mul_binary_tile_init();
    add_binary_tile_init();
    sub_binary_tile_init();

    // Dst regs (bf16 path): use exactly 8 regs [0..7]
    // dst0: recip(z)
    // dst1: log(z)
    // dst2: constant tile (reused)
    // dst3: series accumulator
    // dst4: r2 = (1/z)^2
    // dst5: tmp power accumulator (r2, r4, r6, ...)
    // dst6: term
    // dst7: final result

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_in0, 1);
        cb_reserve_back(cb_out0, 1);

        tile_regs_acquire();

        // Load z twice: once for recip, once for log
        copy_tile(cb_in0, 0, 0);  // dst0 = z
        copy_tile(cb_in0, 0, 1);  // dst1 = z

        // dst1 = log(z)
        log_tile(1);

        // dst0 = 1/z
        recip_tile(0);

        // series = 0.5 * (1/z)
        fill_tile(2, 0.5f);
        mul_binary_tile(0, 2, 3);  // dst3 = dst0 * 0.5

        // r2 = (1/z)^2
        mul_binary_tile(0, 0, 4);  // dst4 = r2
        // tmp starts at r2
        mul_binary_tile(0, 0, 5);  // dst5 = r2

        // out -= (1/12) * r2
        fill_tile(2, 0.083333333f);
        mul_binary_tile(5, 2, 6);
        sub_binary_tile(3, 6, 3);

        // tmp = r4 ; out += (1/120) * r4
        mul_binary_tile(5, 4, 5);
        fill_tile(2, 0.008333333333333333f);
        mul_binary_tile(5, 2, 6);
        add_binary_tile(3, 6, 3);

        // tmp = r6 ; out -= (1/252) * r6
        mul_binary_tile(5, 4, 5);
        fill_tile(2, 0.003968253968253968f);
        mul_binary_tile(5, 2, 6);
        sub_binary_tile(3, 6, 3);

        // tmp = r8 ; out += (1/240) * r8
        mul_binary_tile(5, 4, 5);
        fill_tile(2, 0.004166666666666667f);
        mul_binary_tile(5, 2, 6);
        add_binary_tile(3, 6, 3);

        // tmp = r10 ; out -= (1/132) * r10
        mul_binary_tile(5, 4, 5);
        fill_tile(2, 0.007575757575757576f);
        mul_binary_tile(5, 2, 6);
        sub_binary_tile(3, 6, 3);

        // tmp = r12 ; out += (691/32760) * r12
        mul_binary_tile(5, 4, 5);
        fill_tile(2, 0.021092796092796094f);
        mul_binary_tile(5, 2, 6);
        add_binary_tile(3, 6, 3);

        // tmp = r14 ; out -= (1/12) * r14
        mul_binary_tile(5, 4, 5);
        fill_tile(2, 0.08333333333333333f);
        mul_binary_tile(5, 2, 6);
        sub_binary_tile(3, 6, 3);

        // result = log(z) - out
        sub_binary_tile(1, 3, 7);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(7, cb_out0);

        tile_regs_release();

        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
    }
}
}  // namespace NAMESPACE
"""


# -----------------------------------------------------------------------------
# Writer: CB16 -> DRAM
# -----------------------------------------------------------------------------
writer_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Common runtime args
    uint32_t dst_addr            = get_common_arg_val<uint32_t>(0);
    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    uint32_t num_cores_used      = get_common_arg_val<uint32_t>(4);

    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + (my_y * grid_size_x);

    if (core_idx >= num_cores_used) {
        return;
    }

    uint32_t start_tile_index =
        core_idx * base_tiles_per_core + ((core_idx < extra_tile_range) ? core_idx : extra_tile_range);
    uint32_t n_tiles = base_tiles_per_core + ((core_idx < extra_tile_range) ? 1 : 0);

    constexpr uint32_t cb_out0 = 16;
    const uint32_t tile_bytes = get_tile_size(cb_out0);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out0);

        noc_async_write_tile(start_tile_index + i, dst, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_out0, 1);
    }
}
"""


def _num_tiles_from_shape(shape) -> int:
    dims = list(shape)
    if len(dims) == 0:
        return 0
    if len(dims) == 1:
        w = int(dims[0])
        return (w + 31) // 32
    h = int(dims[-2])
    w = int(dims[-1])
    ht = (h + 31) // 32
    wt = (w + 31) // 32
    nc = 1
    for d in dims[:-2]:
        nc *= int(d)
    return nc * ht * wt


def host(a: ttnn.Tensor) -> ttnn.Tensor:
    # This custom kernel targets TILE_LAYOUT + BF16 digamma approximation.
    if a.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError("custom digamma kernel expects input in TILE_LAYOUT")
    if a.dtype != ttnn.bfloat16:
        raise RuntimeError("custom digamma kernel expects input dtype bfloat16")

    device = a.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(a.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    num_tiles = _num_tiles_from_shape(a.shape)
    if num_tiles == 0:
        return out

    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y

    # Use only a prefix of the grid; extra cores (in the last used row) self-disable in kernels.
    num_cores_used = min(num_tiles, max_cores)
    used_grid_y = (num_cores_used + grid.x - 1) // grid.x
    if used_grid_y < 1:
        used_grid_y = 1

    core_ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, used_grid_y - 1))]
    )

    base_tiles_per_core = num_tiles // num_cores_used
    extra_tile_range = num_tiles % num_cores_used

    # CBs
    cb_in0 = 0
    cb_out0 = 16

    tile_bytes = 32 * 32 * 2  # bf16 tile
    tiles_per_cb = 2
    cb_total = tiles_per_cb * tile_bytes

    cb_in_desc = ttnn.CBDescriptor(
        total_size=cb_total,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_in0, data_format=ttnn.bfloat16, page_size=tile_bytes)
        ],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_out0, data_format=ttnn.bfloat16, page_size=tile_bytes)
        ],
    )

    # Compile-time args
    reader_ct_args = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    writer_ct_args = ttnn.TensorAccessorArgs(out).get_compile_time_args()

    # Common runtime args
    reader_common_rt = [
        a.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        num_cores_used,
    ]
    compute_common_rt = [
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        num_cores_used,
    ]
    writer_common_rt = [
        out.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        num_cores_used,
    ]

    reader_k = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=[],
        common_runtime_args=reader_common_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=[],
        runtime_args=[],
        common_runtime_args=compute_common_rt,
        config=ttnn.ComputeConfigDescriptor(),
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=writer_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=[],
        common_runtime_args=writer_common_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[cb_in_desc, cb_out_desc],
    )

    return ttnn.generic_op([a, out], program)