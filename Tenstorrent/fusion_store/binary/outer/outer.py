# SPDX-License-Identifier: Apache-2.0
import ttnn


# -------------------------------------------------------------------------------------------------
# Fully-manual Reader–Compute–Writer implementation of:
#
#   reference(a, b) = ttnn.outer(a, b)
#
# Reference behavior (from op docs):
# - flatten a to volume_a and reshape to [1, 1, volume_a, 1]
# - flatten b to volume_b and reshape to [1, 1, 1, volume_b]
# - to_layout(..., TILE_LAYOUT)
# - matmul(a_slim, b_slim)
#
# This implementation fuses the "reshape+to_layout" logically into the reader by *assembling*
# per-output-tile Acol and Brow tiles directly in L1, then uses a single-tile matmul:
#   Acol: only first column holds a[rows], rest = 0
#   Brow: only first row holds b[cols], rest = 0
#   out_tile = Acol @ Brow  => out[i,j] = a[i] * b[j]
#
# Supported:
# - a.layout == b.layout == TILE_LAYOUT
# - dtype: bfloat16 or float32 (a.dtype == b.dtype)
# - shapes: 4D with >=3 dimensions equal to 1 (same constraint as reference)
# -------------------------------------------------------------------------------------------------


reader_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"

static inline uint32_t tile_element_offset_bytes(uint32_t r, uint32_t c, uint32_t elem_size_bytes) {
    // Assumes TT "face" tile layout: 4 faces of 16x16, each face row-major.
    // face index: (r>=16)*2 + (c>=16) in {0,1,2,3}
    const uint32_t face = ((r >> 4) << 1) | (c >> 4);
    const uint32_t in_face_r = r & 15;
    const uint32_t in_face_c = c & 15;
    const uint32_t elem_index = face * 256 + in_face_r * 16 + in_face_c;  // 256 elems per face
    return elem_index * elem_size_bytes;
}

static inline void clear_tile(uint32_t l1_addr, uint32_t tile_bytes) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(l1_addr);
    const uint32_t words = tile_bytes >> 2;
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = 0;
    }
}

static inline void copy_elem(uint32_t dst, uint32_t src, uint32_t elem_size_bytes) {
    if (elem_size_bytes == 2) {
        *reinterpret_cast<volatile uint16_t*>(dst) = *reinterpret_cast<volatile uint16_t*>(src);
    } else {
        *reinterpret_cast<volatile uint32_t*>(dst) = *reinterpret_cast<volatile uint32_t*>(src);
    }
}

void kernel_main() {
    // common args
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(2);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(4);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(5);

    // outer-specific
    const uint32_t Nt      = get_common_arg_val<uint32_t>(6);   // output tiles in W dimension
    const uint32_t len_a   = get_common_arg_val<uint32_t>(7);   // volume(a)
    const uint32_t len_b   = get_common_arg_val<uint32_t>(8);   // volume(b)
    const uint32_t axis_a  = get_common_arg_val<uint32_t>(9);   // which dim is non-1 for a  (0..3)
    const uint32_t axis_b  = get_common_arg_val<uint32_t>(10);  // which dim is non-1 for b (0..3)

    // core index
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles =
        base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    // CBs
    constexpr uint32_t cb_a = tt::CBIndex::c_0;   // assembled Acol
    constexpr uint32_t cb_b = tt::CBIndex::c_1;   // assembled Brow
    constexpr uint32_t cb_s = tt::CBIndex::c_2;   // scratch tile

    const uint32_t tile_bytes = get_tile_size(cb_a);
    const uint32_t elem_size_bytes = tile_bytes >> 10;  // tile_bytes / (32*32)

    // Tensor accessors (compile-time args set by host)
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, a_addr, tile_bytes);
    const auto b = TensorAccessor(b_args, b_addr, tile_bytes);

    for (uint32_t it = 0; it < n_tiles; ++it) {
        const uint32_t g = start_tile_id + it;  // global output tile id in [0, Mt*Nt)

        // Map output tile id -> (tile_row, tile_col)
        const uint32_t tile_row = g / Nt;
        const uint32_t tile_col = g - tile_row * Nt;

        // -------------------------
        // Assemble Acol into cb_a
        // -------------------------
        cb_reserve_back(cb_a, 1);
        uint32_t l1_a = get_write_ptr(cb_a);

        if (axis_a == 2) {
            // a is already a column-vector shape in TILE_LAYOUT ([1,1,len_a,1] after reference reshape)
            // tile_row selects which 32-row block.
            noc_async_read_page(tile_row, a, l1_a);
            noc_async_read_barrier();
        } else if (axis_a == 3) {
            // a is a row-vector tile stream ([1,1,1,len_a] style): read full tile then transpose-gather row0 -> col0
            cb_reserve_back(cb_s, 1);
            uint32_t l1_s = get_write_ptr(cb_s);

            noc_async_read_page(tile_row, a, l1_s);
            noc_async_read_barrier();

            clear_tile(l1_a, tile_bytes);

            // Copy up to 32 elements from source row0 into destination col0
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t idx = tile_row * 32 + r;
                if (idx >= len_a) break;

                const uint32_t src_off = tile_element_offset_bytes(0, r, elem_size_bytes);
                const uint32_t dst_off = tile_element_offset_bytes(r, 0, elem_size_bytes);
                copy_elem(l1_a + dst_off, l1_s + src_off, elem_size_bytes);
            }

            cb_push_back(cb_s, 1);
            cb_pop_front(cb_s, 1);
        } else {
            // axis_a in {0,1}: a elements are stored as 1 value per tile at (0,0).
            // Gather 32 scalar tiles into one Acol tile.
            clear_tile(l1_a, tile_bytes);

            cb_reserve_back(cb_s, 1);
            uint32_t l1_s = get_write_ptr(cb_s);

            // Read 32-byte chunk from each scalar tile into scratch at stride 32 bytes (aligned-ish)
            // Then copy the first element into the destination column.
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t idx = tile_row * 32 + r;
                if (idx >= len_a) break;

                const uint64_t src_noc = get_noc_addr(idx, a, 0 /*offset*/);
                const uint32_t scratch_dst = l1_s + (r * 32);
                noc_async_read(src_noc, scratch_dst, 32);
            }
            noc_async_read_barrier();

            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t idx = tile_row * 32 + r;
                if (idx >= len_a) break;

                const uint32_t dst_off = tile_element_offset_bytes(r, 0, elem_size_bytes);
                copy_elem(l1_a + dst_off, l1_s + (r * 32), elem_size_bytes);
            }

            cb_push_back(cb_s, 1);
            cb_pop_front(cb_s, 1);
        }

        cb_push_back(cb_a, 1);

        // -------------------------
        // Assemble Brow into cb_b
        // -------------------------
        cb_reserve_back(cb_b, 1);
        uint32_t l1_b = get_write_ptr(cb_b);

        if (axis_b == 3) {
            // b already row-vector tile stream
            noc_async_read_page(tile_col, b, l1_b);
            noc_async_read_barrier();
        } else if (axis_b == 2) {
            // b is column-vector tile stream: read full tile then gather col0 -> row0
            cb_reserve_back(cb_s, 1);
            uint32_t l1_s = get_write_ptr(cb_s);

            noc_async_read_page(tile_col, b, l1_s);
            noc_async_read_barrier();

            clear_tile(l1_b, tile_bytes);

            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t idx = tile_col * 32 + c;
                if (idx >= len_b) break;

                const uint32_t src_off = tile_element_offset_bytes(c, 0, elem_size_bytes);
                const uint32_t dst_off = tile_element_offset_bytes(0, c, elem_size_bytes);
                copy_elem(l1_b + dst_off, l1_s + src_off, elem_size_bytes);
            }

            cb_push_back(cb_s, 1);
            cb_pop_front(cb_s, 1);
        } else {
            // axis_b in {0,1}: one scalar per tile at (0,0)
            clear_tile(l1_b, tile_bytes);

            cb_reserve_back(cb_s, 1);
            uint32_t l1_s = get_write_ptr(cb_s);

            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t idx = tile_col * 32 + c;
                if (idx >= len_b) break;

                const uint64_t src_noc = get_noc_addr(idx, b, 0 /*offset*/);
                const uint32_t scratch_dst = l1_s + (c * 32);
                noc_async_read(src_noc, scratch_dst, 32);
            }
            noc_async_read_barrier();

            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t idx = tile_col * 32 + c;
                if (idx >= len_b) break;

                const uint32_t dst_off = tile_element_offset_bytes(0, c, elem_size_bytes);
                copy_elem(l1_b + dst_off, l1_s + (c * 32), elem_size_bytes);
            }

            cb_push_back(cb_s, 1);
            cb_pop_front(cb_s, 1);
        }

        cb_push_back(cb_b, 1);
    }
}
"""


compute_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // common args
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    // core index
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t num_tiles =
        base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);
    mm_init(cb_a, cb_b, cb_out, /*transpose=*/0);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        tile_regs_acquire();

        // matmul_tiles(in0_cb, in1_cb, in0_tile_idx, in1_tile_idx, dst)
        matmul_tiles(cb_a, cb_b, 0, 0, 0);

        tile_regs_commit();

        // Overlap CB management with packing
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
    }
}
}  // namespace NAMESPACE
"""


writer_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // common args
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    // core index
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles =
        base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    // Tensor accessor (compile-time args set by host)
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t l1 = get_read_ptr(cb_out);

        noc_async_write_page(start_tile_id + i, out, l1);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
"""


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _volume_and_axis(shape) -> tuple[int, int]:
    dims = [int(x) for x in list(shape)]
    # Expect 4D like reference, but tolerate other by padding to 4D
    if len(dims) < 4:
        dims = [1] * (4 - len(dims)) + dims
    if len(dims) != 4:
        raise RuntimeError(f"outer: expected <=4D shape, got {dims}")

    vol = 1
    axis = None
    for i, d in enumerate(dims):
        vol *= d
        if d != 1:
            if axis is None:
                axis = i
            else:
                axis = -1  # multiple non-1 dims
    if axis is None:
        axis = 2  # scalar; arbitrary
    return vol, axis


def host(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    # ---- Validate device/dtype/layout ----
    if a.device() != b.device():
        raise RuntimeError("outer: a and b must be on the same device")
    if a.dtype != b.dtype:
        raise RuntimeError("outer: a and b must have the same dtype")
    if a.dtype not in (ttnn.bfloat16, ttnn.float32):
        raise RuntimeError("outer: only bfloat16/float32 supported in this manual kernel")
    if a.layout != ttnn.TILE_LAYOUT or b.layout != ttnn.TILE_LAYOUT:
        # Reference does to_layout(TILE_LAYOUT). Here we require already tiled (no calling ttnn.to_layout allowed).
        raise RuntimeError("outer: this manual kernel requires TILE_LAYOUT inputs")

    len_a, axis_a = _volume_and_axis(a.shape)
    len_b, axis_b = _volume_and_axis(b.shape)

    if axis_a == -1 or axis_b == -1:
        raise RuntimeError(
            f"outer: reference requires >=3 dims == 1 for each input; got a.shape={list(a.shape)} b.shape={list(b.shape)}"
        )

    device = a.device()
    out_shape = ttnn.Shape([1, 1, int(len_a), int(len_b)])
    out = ttnn.allocate_tensor_on_device(out_shape, a.dtype, ttnn.TILE_LAYOUT, device)

    Mt = _ceil_div(len_a, 32)
    Nt = _ceil_div(len_b, 32)
    num_out_tiles = Mt * Nt
    if num_out_tiles == 0:
        return out

    grid = device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    base_tiles_per_core = num_out_tiles // total_cores
    extra_tile_range = num_out_tiles % total_cores

    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
    )

    element_size = 2 if a.dtype == ttnn.bfloat16 else 4
    tile_bytes = 32 * 32 * element_size

    # CBs: 0=Acol, 1=Brow, 2=scratch, 16=out
    tiles_per_cb = 2
    cb_ab_out_total = tiles_per_cb * tile_bytes

    cbs = [
        ttnn.CBDescriptor(
            total_size=cb_ab_out_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=a.dtype, page_size=tile_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=cb_ab_out_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=b.dtype, page_size=tile_bytes)],
        ),
        # scratch: 1 page is enough
        ttnn.CBDescriptor(
            total_size=tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2, data_format=a.dtype, page_size=tile_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=cb_ab_out_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=out.dtype, page_size=tile_bytes)],
        ),
    ]

    # Compile-time args
    reader_ct = []
    reader_ct += ttnn.TensorAccessorArgs(a).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(b).get_compile_time_args()
    writer_ct = []
    writer_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()

    # Common runtime args
    reader_common_rt = [
        a.buffer_address(),
        b.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid.x,
        grid.y,
        Nt,
        int(len_a),
        int(len_b),
        int(axis_a),
        int(axis_b),
    ]
    compute_common_rt = [base_tiles_per_core, extra_tile_range, grid.x, grid.y]
    writer_common_rt = [out.buffer_address(), base_tiles_per_core, extra_tile_range, grid.x, grid.y]

    reader_k = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=[],
        common_runtime_args=reader_common_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_cfg = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=(a.dtype == ttnn.float32))
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
        compile_time_args=writer_ct,
        runtime_args=[],
        common_runtime_args=writer_common_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    prog = ttnn.ProgramDescriptor(kernels=[reader_k, compute_k, writer_k], semaphores=[], cbs=cbs)
    return ttnn.generic_op([a, b, out], prog)