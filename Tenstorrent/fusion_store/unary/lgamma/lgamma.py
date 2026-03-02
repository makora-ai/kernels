# SPDX-License-Identifier: Apache-2.0
import ttnn


# -------------------------------------------------------------------------------------------------
# Reader: DRAM/SRAM -> CB0 (input a)
# -------------------------------------------------------------------------------------------------
reader_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Common RT args
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    // Core/work mapping
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1u : 0u);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;

    const uint32_t tile_bytes = get_tile_size(cb_in0);
    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t tile_id = start_tile_id + i;

        cb_reserve_back(cb_in0, onetile);
        const uint32_t l1_write_addr = get_write_ptr(cb_in0);

        // Page == tile when page_size == tile_bytes (host sets CB page_size accordingly)
        noc_async_read_page(tile_id, in, l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_in0, onetile);
    }
}
"""


# -------------------------------------------------------------------------------------------------
# Compute: lgamma(a) using the exact composite flow from ttnn (Lanczos-style approximation)
# Notes:
# - We run compute in FP32 dst regs for improved accuracy even when input/output are bf16.
# - Use where_fp32_tile because dst regs are FP32 (fp32_dest_acc_en=True on host).
# - Use log_tile<1u>() to match ttnn::log(x, true) calls in the composite.
# -------------------------------------------------------------------------------------------------
compute_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

// Pull in SFPU op headers via split includes (required for recip/log/fill/comp/where)
#define SFPU_OP_RECIP_INCLUDE 1
#define SFPU_OP_LOG_INCLUDE 1
#define SFPU_OP_FILL_INCLUDE 1
#define SFPU_OP_UNARY_COMP_INCLUDE 1
#define SFPU_OP_WHERE_INCLUDE 1
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Common RT args
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    // Core/work mapping
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1u : 0u);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // One-time init (MUST be outside the tile loop)
    init_sfpu(cb_in0, cb_out);
    copy_tile_init(cb_in0);

    fill_tile_init();
    recip_tile_init();
    // ttnn composite uses log(x, true) for both logs
    log_tile_init<1u>();

    unary_eq_tile_init();
    where_tile_init();

    add_binary_tile_init();
    sub_binary_tile_init();
    mul_binary_tile_init();

    // Dst regs (FP32 because host sets fp32_dest_acc_en=True, dst_full_sync_en=True => 8 regs)
    constexpr uint32_t R_X      = 0; // x
    constexpr uint32_t R_INP    = 1; // input = x - 1
    constexpr uint32_t R_TEMP   = 2; // temp accumulator (then temp_log in-place)
    constexpr uint32_t R_TLOG   = 3; // t_log
    constexpr uint32_t R_T      = 4; // t
    constexpr uint32_t R_RES    = 5; // result
    constexpr uint32_t R_PRED   = 6; // predicate scratch
    constexpr uint32_t R_SCR    = 7; // scalar scratch (also holds 0.0 for where)

    // Bitcasts for unary_eq_tile param0 (float32 bits)
    constexpr uint32_t FP32_1_0 = 0x3f800000u;
    constexpr uint32_t FP32_2_0 = 0x40000000u;

    constexpr uint32_t onetile = 1;

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in0, onetile);
        cb_reserve_back(cb_out, onetile);

        tile_regs_acquire();

        // x -> R_X
        copy_tile(cb_in0, 0, R_X);
        cb_pop_front(cb_in0, onetile);

        // input = x - 1
        fill_tile(R_SCR, 1.0f);
        sub_binary_tile(R_X, R_SCR, R_INP);

        // temp = 1.0
        fill_tile(R_TEMP, 1.0f);

        // temp += coeff * recip(input + shift)
#define ACCUM_TERM(SHIFT_F, COEFF_F)                  \
        fill_tile(R_SCR, (SHIFT_F));                  \
        add_binary_tile(R_INP, R_SCR, R_PRED);        \
        recip_tile(R_PRED);                           \
        fill_tile(R_SCR, (COEFF_F));                  \
        mul_binary_tile(R_PRED, R_SCR, R_PRED);       \
        add_binary_tile(R_TEMP, R_PRED, R_TEMP);

        ACCUM_TERM(1.0f,  76.18009172947146f);
        ACCUM_TERM(2.0f, -86.50532032941677f);
        ACCUM_TERM(3.0f,  24.01409824083091f);
        ACCUM_TERM(4.0f,  -1.231739572450155f);
        ACCUM_TERM(5.0f,   0.001208650973866179f);
        ACCUM_TERM(6.0f,  -0.000005395239384953f);

#undef ACCUM_TERM

        // t = input + 5.5
        fill_tile(R_SCR, 5.5f);
        add_binary_tile(R_INP, R_SCR, R_T);

        // t_log = log(t) (approx mode to match ttnn::log(x, true))
        fill_tile(R_SCR, 0.0f);
        add_binary_tile(R_T, R_SCR, R_TLOG);   // copy t -> t_log
        log_tile<1u>(R_TLOG);

        // temp_log = log(temp) (approx mode)
        log_tile<1u>(R_TEMP);

        // result = (input + 0.5) * t_log + 0.918... + temp_log - t
        fill_tile(R_SCR, 0.5f);
        add_binary_tile(R_INP, R_SCR, R_RES);      // input + 0.5
        mul_binary_tile(R_RES, R_TLOG, R_RES);     // * t_log

        fill_tile(R_SCR, 0.918938531357171f);
        add_binary_tile(R_RES, R_SCR, R_RES);      // + log(sqrt(2*pi))

        add_binary_tile(R_RES, R_TEMP, R_RES);     // + temp_log
        sub_binary_tile(R_RES, R_T, R_RES);        // - t

        // result = where(eq(x, 1.0), 0.0, result)
        fill_tile(R_SCR, 0.0f);                    // zero tile
        add_binary_tile(R_X, R_SCR, R_PRED);        // copy x -> pred
        unary_eq_tile(R_PRED, FP32_1_0);
        where_fp32_tile(R_PRED, R_SCR, R_RES, R_RES);

        // result = where(eq(x, 2.0), 0.0, result)
        add_binary_tile(R_X, R_SCR, R_PRED);        // copy x -> pred
        unary_eq_tile(R_PRED, FP32_2_0);
        where_fp32_tile(R_PRED, R_SCR, R_RES, R_RES);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(R_RES, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, onetile);
    }
}
} // namespace NAMESPACE
"""


# -------------------------------------------------------------------------------------------------
# Writer: CB16 -> DRAM/SRAM (output)
# -------------------------------------------------------------------------------------------------
writer_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Common RT args
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    // Core/work mapping
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1u : 0u);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    const uint32_t tile_bytes = get_tile_size(cb_out0);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t tile_id = start_tile_id + i;

        cb_wait_front(cb_out0, onetile);
        const uint32_t l1_read_addr = get_read_ptr(cb_out0);

        noc_async_write_page(tile_id, out, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_out0, onetile);
    }
}
"""


def _num_tiles_for_tile_layout(shape: ttnn.Shape) -> int:
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


def host(a: ttnn.Tensor) -> ttnn.Tensor:
    # reference(a) = ttnn.lgamma(a)
    device = a.device()

    # The kernels operate on TILE_LAYOUT. Convert if needed and convert back after.
    orig_layout = a.layout
    a_tiled = a if a.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(a, ttnn.TILE_LAYOUT)

    # This implementation supports bf16 and fp32 device tensors (matches tile byte sizing logic).
    # If other dtype is provided, cast to bf16 (best-effort).
    orig_dtype = a_tiled.dtype
    if a_tiled.dtype not in (ttnn.bfloat16, ttnn.float32):
        a_tiled = ttnn.to_dtype(a_tiled, ttnn.bfloat16)

    out_tiled = ttnn.allocate_tensor_on_device(
        ttnn.Shape(a_tiled.shape),
        a_tiled.dtype,
        ttnn.TILE_LAYOUT,
        device,
    )

    num_tiles = _num_tiles_for_tile_layout(a_tiled.shape)
    if num_tiles == 0:
        out = out_tiled
        # Restore original layout (and dtype if we had to cast)
        if orig_layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, orig_layout)
        if a.dtype != out.dtype:
            out = ttnn.to_dtype(out, a.dtype)
        return out

    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y
    base_tiles_per_core = num_tiles // total_cores
    extra_tile_range = num_tiles % total_cores

    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    # CB sizing by dtype
    if a_tiled.dtype == ttnn.float32:
        tile_size_bytes = 32 * 32 * 4
    else:
        tile_size_bytes = 32 * 32 * 2

    tiles_per_cb = 2
    cb_total_bytes = tiles_per_cb * tile_size_bytes
    cb_page_size = tile_size_bytes

    cb_in_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=a_tiled.dtype, page_size=cb_page_size)],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=out_tiled.dtype, page_size=cb_page_size)],
    )

    reader_ct_args = ttnn.TensorAccessorArgs(a_tiled).get_compile_time_args()
    writer_ct_args = ttnn.TensorAccessorArgs(out_tiled).get_compile_time_args()

    reader_common_rt_args = [
        a_tiled.buffer_address(),
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
        out_tiled.buffer_address(),
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

    # FP32 dst regs for accuracy, but keep IO dtype as-is (packer converts to CB format)
    compute_cfg = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, dst_full_sync_en=True)
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
        cbs=[cb_in_desc, cb_out_desc],
    )

    out_tiled = ttnn.generic_op([a_tiled, out_tiled], program)

    # Restore original layout/dtype if we had to coerce them for this kernel
    out = out_tiled
    if orig_layout != ttnn.TILE_LAYOUT:
        out = ttnn.to_layout(out, orig_layout)
    if a.dtype != out.dtype:
        out = ttnn.to_dtype(out, a.dtype)
    return out