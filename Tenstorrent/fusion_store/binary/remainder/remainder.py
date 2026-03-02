# SPDX-License-Identifier: Apache-2.0
import ttnn


# --------------------------------------------------------------------------------------
# Reader Kernel: stream A,B tiles -> CB0,CB1 with broadcast support
# Uses noc_async_read_page (non-deprecated).
# Broadcast is implemented by mapping each output tile_id -> (n,c,tr,tc) in a canonical
# 4D (N,C,H,W) view, then computing the corresponding A/B tile_id with per-dim ==1 rules.
# --------------------------------------------------------------------------------------
reader_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Base addresses
    uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    uint32_t b_addr = get_common_arg_val<uint32_t>(1);

    // Work distribution
    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(2);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(3);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(4);
    uint32_t grid_size_y         = get_common_arg_val<uint32_t>(5);

    // Canonical 4D (N,C,Ht,Wt) tiling params
    // Output
    uint32_t out_N  = get_common_arg_val<uint32_t>(6);
    uint32_t out_C  = get_common_arg_val<uint32_t>(7);
    uint32_t out_Ht = get_common_arg_val<uint32_t>(8);
    uint32_t out_Wt = get_common_arg_val<uint32_t>(9);

    // A
    uint32_t a_N  = get_common_arg_val<uint32_t>(10);
    uint32_t a_C  = get_common_arg_val<uint32_t>(11);
    uint32_t a_Ht = get_common_arg_val<uint32_t>(12);
    uint32_t a_Wt = get_common_arg_val<uint32_t>(13);

    // B
    uint32_t b_N  = get_common_arg_val<uint32_t>(14);
    uint32_t b_C  = get_common_arg_val<uint32_t>(15);
    uint32_t b_Ht = get_common_arg_val<uint32_t>(16);
    uint32_t b_Wt = get_common_arg_val<uint32_t>(17);

    // Core id
    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + my_y * grid_size_x;

    uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    const uint32_t a_tile_bytes = get_tile_size(cb_a);
    const uint32_t b_tile_bytes = get_tile_size(cb_b);

    // Tensor accessors (compile-time args packed by host)
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, a_tile_bytes);

    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, b_tile_bytes);

    constexpr uint32_t one_tile = 1;

    const uint32_t out_tiles_per_nc = out_Ht * out_Wt;
    const uint32_t a_tiles_per_nc   = a_Ht * a_Wt;
    const uint32_t b_tiles_per_nc   = b_Ht * b_Wt;

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t out_tile_id = start_tile_id + i;

        // Decompose out_tile_id -> (n, c, tr, tc) in canonical (N,C,Ht,Wt)
        const uint32_t out_nc = out_tile_id / out_tiles_per_nc;
        const uint32_t rem0   = out_tile_id - out_nc * out_tiles_per_nc;
        const uint32_t tr     = rem0 / out_Wt;
        const uint32_t tc     = rem0 - tr * out_Wt;

        const uint32_t n = out_nc / out_C;
        const uint32_t c = out_nc - n * out_C;

        // Broadcast map for A
        const uint32_t a_n  = (a_N  == 1) ? 0 : n;
        const uint32_t a_c  = (a_C  == 1) ? 0 : c;
        const uint32_t a_tr = (a_Ht == 1) ? 0 : tr;
        const uint32_t a_tc = (a_Wt == 1) ? 0 : tc;
        const uint32_t a_tile_id = (a_n * a_C + a_c) * a_tiles_per_nc + a_tr * a_Wt + a_tc;

        // Broadcast map for B
        const uint32_t b_n  = (b_N  == 1) ? 0 : n;
        const uint32_t b_c  = (b_C  == 1) ? 0 : c;
        const uint32_t b_tr = (b_Ht == 1) ? 0 : tr;
        const uint32_t b_tc = (b_Wt == 1) ? 0 : tc;
        const uint32_t b_tile_id = (b_n * b_C + b_c) * b_tiles_per_nc + b_tr * b_Wt + b_tc;

        cb_reserve_back(cb_a, one_tile);
        cb_reserve_back(cb_b, one_tile);

        uint32_t l1_a = get_write_ptr(cb_a);
        uint32_t l1_b = get_write_ptr(cb_b);

        noc_async_read_page(a_tile_id, a, l1_a);
        noc_async_read_page(b_tile_id, b, l1_b);

        noc_async_read_barrier();

        cb_push_back(cb_a, one_tile);
        cb_push_back(cb_b, one_tile);
    }
}
"""


# --------------------------------------------------------------------------------------
# Compute Kernel: remainder(a,b) FP32 composite (matches ttnn::run_remainder)
#   q = floor(a / b)
#   r = a - b*q
#   r = where(r >= b, r - b, r)
#   r = where(b < 0,  r + b, r)
#
# NOTE: Previous failures were due to calling non-existent floor_tile_float32().
#       The correct API is floor_tile(idst) from rounding.h.
# --------------------------------------------------------------------------------------
compute_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/rounding.h"

#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + my_y * grid_size_x;

    uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Configure SFPU pipe (format/packer/unpacker) once.
    init_sfpu(cb_a, cb_out);

    // SFPU binary ops (DST->DST)
    div_binary_tile_init();
    mul_binary_tile_init();
    sub_binary_tile_init();
    add_binary_tile_init();

    // SFPU unary ops
    rounding_op_tile_init();
    gez_tile_init();
    ltz_tile_init();
    where_tile_init();

    constexpr uint32_t one_tile = 1;

    // FP32 dest acc enabled => 4 dst regs available (DST0..DST3)
    //  DST0: a then pred1
    //  DST1: b then pred2
    //  DST2: temporaries (q, b*q, r-b, r+b)
    //  DST3: r (running)
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_a, one_tile);
        cb_wait_front(cb_b, one_tile);

        // Avoid holding regs while waiting for output space
        cb_reserve_back(cb_out, one_tile);

        tile_regs_acquire();

        // Load a -> DST0
        copy_tile_init(cb_a);
        copy_tile(cb_a, 0, 0);

        // Load b -> DST1
        copy_tile_init(cb_b);
        copy_tile(cb_b, 0, 1);

        // Inputs are no longer needed in CBs after copy
        cb_pop_front(cb_a, one_tile);
        cb_pop_front(cb_b, one_tile);

        // q = floor(a/b) -> DST2
        div_binary_tile(0, 1, 2);
        floor_tile(2);

        // b*q -> DST2
        mul_binary_tile(1, 2, 2);

        // r0 = a - b*q -> DST3
        sub_binary_tile(0, 2, 3);

        // tmp = r0 - b -> DST2  (true value for first where)
        sub_binary_tile(3, 1, 2);

        // pred1 = (r0 >= b)  <=> (r0 - b) >= 0  -> DST0
        sub_binary_tile(3, 1, 0);
        gez_tile(0);

        // r1 = where(pred1, tmp, r0) -> DST3
        where_fp32_tile(0, 2, 3, 3);

        // tmp2 = r1 + b -> DST2
        add_binary_tile(3, 1, 2);

        // pred2 = (b < 0) -> DST1
        ltz_tile(1);

        // r2 = where(pred2, tmp2, r1) -> DST3
        where_fp32_tile(1, 2, 3, 3);

        tile_regs_commit();

        tile_regs_wait();
        pack_tile(3, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, one_tile);
    }
}
}  // namespace NAMESPACE
"""


# --------------------------------------------------------------------------------------
# Writer Kernel: CB16 -> output tiles
# Uses noc_async_write_page (non-deprecated).
# --------------------------------------------------------------------------------------
writer_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    uint32_t my_x = get_absolute_logical_x();
    uint32_t my_y = get_absolute_logical_y();
    uint32_t core_idx = my_x + my_y * grid_size_x;

    uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, out_tile_bytes);

    constexpr uint32_t one_tile = 1;

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t tile_id = start_tile_id + i;

        cb_wait_front(cb_out, one_tile);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        noc_async_write_page(tile_id, out, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_out, one_tile);
    }
}
"""


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _broadcast_shapes(shape_a, shape_b):
    a = list(shape_a)
    b = list(shape_b)
    out_rank = max(len(a), len(b))
    a = [1] * (out_rank - len(a)) + a
    b = [1] * (out_rank - len(b)) + b
    out = []
    for da, db in zip(a, b):
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            raise ValueError(f"Shapes not broadcastable: {shape_a} vs {shape_b}")
    return out


def _canonical_4d_shape(shape) -> tuple[int, int, int, int]:
    # Canonicalize to (N,C,H,W) for kernel-side broadcast mapping.
    dims = list(shape)
    if len(dims) == 0:
        return (1, 1, 1, 1)
    if len(dims) == 1:
        return (1, 1, 1, dims[0])
    if len(dims) == 2:
        return (1, 1, dims[0], dims[1])
    if len(dims) == 3:
        return (dims[0], 1, dims[1], dims[2])
    # len >= 4: fold leading dims into N, keep last-3 as C,H,W
    n = 1
    for d in dims[:-3]:
        n *= d
    c, h, w = dims[-3], dims[-2], dims[-1]
    return (n, c, h, w)


def _tile_params_from_shape_4d(n: int, c: int, h: int, w: int):
    ht = _ceil_div(h, 32)
    wt = _ceil_div(w, 32)
    return n, c, ht, wt


def host(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    # Implements:
    #   do_typecast = (a.dtype != fp32) or (b.dtype != fp32)
    #   a32 = typecast(a, fp32) if do_typecast else a
    #   b32 = typecast(b, fp32) if do_typecast else b
    #   out32 = run_remainder_fp32(a32, b32)   (this custom fused kernel)
    #   return typecast(out32, a.dtype) if do_typecast and a.dtype != fp32 else out32

    device = a.device()
    input_dtype = a.dtype

    # Expect TILE_LAYOUT for this manual tiled kernel path
    assert a.layout == ttnn.TILE_LAYOUT and b.layout == ttnn.TILE_LAYOUT
    assert a.device() == b.device()

    out_shape = _broadcast_shapes(a.shape, b.shape)

    do_typecast = (a.dtype != ttnn.float32) or (b.dtype != ttnn.float32)
    a_fp32 = ttnn.typecast(a, ttnn.float32) if do_typecast else a
    b_fp32 = ttnn.typecast(b, ttnn.float32) if do_typecast else b

    out_fp32 = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
    )

    # Canonicalize shapes to (N,C,H,W) for kernel-side broadcast mapping
    out_N, out_C, out_H, out_W = _canonical_4d_shape(out_shape)
    a_N, a_C, a_H, a_W = _canonical_4d_shape(a_fp32.shape)
    b_N, b_C, b_H, b_W = _canonical_4d_shape(b_fp32.shape)

    out_N, out_C, out_Ht, out_Wt = _tile_params_from_shape_4d(out_N, out_C, out_H, out_W)
    a_N, a_C, a_Ht, a_Wt = _tile_params_from_shape_4d(a_N, a_C, a_H, a_W)
    b_N, b_C, b_Ht, b_Wt = _tile_params_from_shape_4d(b_N, b_C, b_H, b_W)

    num_tiles = (out_N * out_C) * (out_Ht * out_Wt)
    if num_tiles == 0:
        return ttnn.typecast(out_fp32, input_dtype) if (do_typecast and input_dtype != ttnn.float32) else out_fp32

    # Grid/work split (kernel-side distribution)
    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    base_tiles_per_core = num_tiles // total_cores
    extra_tile_range = num_tiles % total_cores

    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )

    # FP32 tile is 32*32*4 = 4096 bytes
    tile_size_bytes = 32 * 32 * 4
    tiles_per_cb = 2
    cb_total_bytes = tiles_per_cb * tile_size_bytes
    cb_page_size = tile_size_bytes

    cb_a_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=cb_page_size)
        ],
    )
    cb_b_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.float32, page_size=cb_page_size)
        ],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.float32, page_size=cb_page_size)
        ],
    )

    # Compile-time args for TensorAccessor
    reader_ct_args = ttnn.TensorAccessorArgs(a_fp32).get_compile_time_args()
    reader_ct_args.extend(ttnn.TensorAccessorArgs(b_fp32).get_compile_time_args())
    writer_ct_args = ttnn.TensorAccessorArgs(out_fp32).get_compile_time_args()

    # Common runtime args
    reader_common_rt_args = [
        a_fp32.buffer_address(),
        b_fp32.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        grid.y,
        out_N,
        out_C,
        out_Ht,
        out_Wt,
        a_N,
        a_C,
        a_Ht,
        a_Wt,
        b_N,
        b_C,
        b_Ht,
        b_Wt,
    ]
    compute_common_rt_args = [
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        grid.y,
    ]
    writer_common_rt_args = [
        out_fp32.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        grid.y,
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
        compile_time_args=[],
        runtime_args=[],
        common_runtime_args=compute_common_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=True,  # fp32 dst tile ops; we use DST0..DST3 only
        ),
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

    out_fp32 = ttnn.generic_op([a_fp32, b_fp32, out_fp32], program)

    # Match reference: cast result back to dtype(a) only if dtype(a) != fp32
    if do_typecast and input_dtype != ttnn.float32:
        return ttnn.typecast(out_fp32, input_dtype)
    return out_fp32