# SPDX-License-Identifier: Apache-2.0
import ttnn


# -------------------------------------------------------------------------------------------------
# Fused GLU(a) for TILE_LAYOUT BF16, dim = last dimension (W).
#
# Reference:
#   a0, a1 = split(a, 2, dim=-1)
#   out = a0 * sigmoid(a1)
#
# IMPORTANT LIMITATION (tile-aligned split):
# - We require the split point to be tile-aligned in W, i.e. input W must be a multiple of 64
#   so that Wt_in is even and Wt_out == Wt_in/2.
# - This matches common GLU/FFN feature sizes (e.g., 256/512/1024/...).
# -------------------------------------------------------------------------------------------------


reader_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Common RT args
    const uint32_t in_addr             = get_common_arg_val<uint32_t>(0);
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    // GLU tiling args
    const uint32_t Ht     = get_common_arg_val<uint32_t>(5);
    const uint32_t Wt_in  = get_common_arg_val<uint32_t>(6);
    const uint32_t Wt_out = get_common_arg_val<uint32_t>(7);

    // Core-id -> linear core index
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    // Work split over OUTPUT tiles
    const uint32_t start_out_tile =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;  // first half tiles
    constexpr uint32_t cb_b = tt::CBIndex::c_1;  // second half tiles

    const uint32_t tile_bytes = get_tile_size(cb_a);

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in_args, in_addr, tile_bytes);

    const uint32_t HtWt_out = Ht * Wt_out;
    const uint32_t HtWt_in  = Ht * Wt_in;

    constexpr uint32_t onetile = 1;

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t out_linear = start_out_tile + i;

        // Map output tile linear index -> (nc, tr, tc_out)
        const uint32_t nc = out_linear / HtWt_out;
        const uint32_t rem = out_linear - nc * HtWt_out;
        const uint32_t tr = rem / Wt_out;
        const uint32_t tc_out = rem - tr * Wt_out;

        // Input tile ids for A and B halves (tile-aligned split in W)
        const uint32_t a_tile = nc * HtWt_in + tr * Wt_in + tc_out;
        const uint32_t b_tile = a_tile + Wt_out;

        cb_reserve_back(cb_a, onetile);
        cb_reserve_back(cb_b, onetile);

        const uint32_t l1_write_a = get_write_ptr(cb_a);
        const uint32_t l1_write_b = get_write_ptr(cb_b);

        noc_async_read_tile(a_tile, in0, l1_write_a);
        noc_async_read_tile(b_tile, in0, l1_write_b);

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

// Standalone SFPU include path
#define SFPU_OP_COMPUTE_KERNEL_API_INCLUDE 1
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Common RT args
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    // Core-id -> linear core index
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    // Work split
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr auto cb_a   = tt::CBIndex::c_0;   // A
    constexpr auto cb_b   = tt::CBIndex::c_1;   // B
    constexpr auto cb_out = tt::CBIndex::c_16;  // out

    // SFPU setup (BF16 tile in/out)
    init_sfpu(cb_a, cb_out);
    sigmoid_tile_init();
    mul_binary_tile_init();

    for (uint32_t t = 0; t < n_tiles; ++t) {
        // Ensure we can make forward progress on output
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        // dst0 = sigmoid(B)
        copy_tile(cb_b, 0, 0);
        sigmoid_tile(0);

        // dst1 = A
        copy_tile(cb_a, 0, 1);

        // dst0 = A * sigmoid(B)
        mul_binary_tile(1, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);

        tile_regs_release();
    }
}
}  // namespace NAMESPACE
"""


writer_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // Common RT args
    const uint32_t out_addr            = get_common_arg_val<uint32_t>(0);
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    // Core-id -> linear core index
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    // Work split over OUTPUT tiles
    const uint32_t start_out_tile =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out_args, out_addr, tile_bytes);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out, onetile);
        const uint32_t l1_read_addr = get_read_ptr(cb_out);

        noc_async_write_tile(start_out_tile + i, out0, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_out, onetile);
    }
}
"""


def _num_tiles_tiled(shape) -> tuple[int, int, int, int]:
    """
    Returns (num_tiles, NC, Ht, Wt) for a TILE_LAYOUT tensor interpreted as (..., H, W).
    """
    dims = list(shape)
    if len(dims) == 0:
        return 0, 1, 0, 0
    if len(dims) == 1:
        H = 1
        W = dims[0]
        Ht = 1
        Wt = (W + 31) // 32
        NC = 1
        return NC * Ht * Wt, NC, Ht, Wt

    H = dims[-2]
    W = dims[-1]
    Ht = (H + 31) // 32
    Wt = (W + 31) // 32
    NC = 1
    for d in dims[:-2]:
        NC *= d
    return NC * Ht * Wt, NC, Ht, Wt


def host(a: ttnn.Tensor) -> ttnn.Tensor:
    # Minimal constraints for this fused kernel
    if a.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError("glu: this kernel expects TILE_LAYOUT input")
    if a.dtype != ttnn.bfloat16:
        raise RuntimeError("glu: this kernel expects BF16 input")

    dims = list(a.shape)
    if len(dims) == 0:
        return ttnn.allocate_tensor_on_device(ttnn.Shape(a.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, a.device())

    W = dims[-1] if len(dims) >= 1 else dims[0]
    if W % 2 != 0:
        raise RuntimeError(f"glu: last dim must be even, got W={W}")

    out_dims = list(dims)
    out_dims[-1] = W // 2

    # Tile-aligned split requirement: W must be multiple of 64 (so Wt_in is even and split is tile boundary).
    _, _, _, Wt_in = _num_tiles_tiled(a.shape)
    _, _, _, Wt_out_shape = _num_tiles_tiled(out_dims)
    if (Wt_in % 2) != 0 or (Wt_in // 2) != Wt_out_shape:
        raise RuntimeError(
            "glu: this fused kernel requires tile-aligned split in W (typically W % 64 == 0). "
            f"Got W={W}, Wt_in={Wt_in}, Wt_out={Wt_out_shape}"
        )

    device = a.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(out_dims), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    out_num_tiles, _, Ht, Wt_out = _num_tiles_tiled(out_dims)
    if out_num_tiles == 0:
        return out

    grid_size = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    num_cores = grid_size.x * grid_size.y
    base_tiles_per_core = out_num_tiles // num_cores
    extra_tile_range = out_num_tiles % num_cores

    # CBs (BF16 tiles)
    tile_bytes = 32 * 32 * 2
    tiles_per_cb = 2
    cb_bytes = tiles_per_cb * tile_bytes

    cb_a, cb_b, cb_out = 0, 1, 16
    cb_a_desc = ttnn.CBDescriptor(
        total_size=cb_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_a, data_format=ttnn.bfloat16, page_size=tile_bytes)],
    )
    cb_b_desc = ttnn.CBDescriptor(
        total_size=cb_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_b, data_format=ttnn.bfloat16, page_size=tile_bytes)],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=cb_out, data_format=ttnn.bfloat16, page_size=tile_bytes)
        ],
    )

    # Compile-time args (TensorAccessorArgs)
    reader_ct = []
    reader_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())

    writer_ct = []
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    # Common runtime args
    reader_common_rt = [
        a.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
        Ht,
        Wt_in,
        Wt_out,
    ]
    compute_common_rt = [
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
    ]
    writer_common_rt = [
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
        compile_time_args=reader_ct,
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
        config=ttnn.ComputeConfigDescriptor(),
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

    prog = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[cb_a_desc, cb_b_desc, cb_out_desc],
    )

    return ttnn.generic_op([a, out], prog)