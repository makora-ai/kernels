# SPDX-License-Identifier: Apache-2.0
import ttnn

# --------------------------------------------------------------------------------------
# Fused TT-Metalium Reader / Compute / Writer for:
#   ttnn.reglu(a, dim=-1)
#
# Reference semantics:
#   A, B = split(a, dim=last)      # equal halves along last dim
#   out  = A * relu(B)
#
# Constraints of this manual kernel:
# - TILE_LAYOUT only
# - BF16 only
# - dim must be -1 or 3 (and tensor rank must be 4)
# - last dim (W) must be divisible by 64 so the split is tile-aligned (Wt_in even)
# --------------------------------------------------------------------------------------

reader_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // ---- common runtime args ----
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    const uint32_t Ht     = get_common_arg_val<uint32_t>(5);
    const uint32_t Wt_in  = get_common_arg_val<uint32_t>(6);
    const uint32_t Wt_out = get_common_arg_val<uint32_t>(7);

    // ---- derive per-core work (kernel-side distribution) ----
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_out =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    // ---- CBs ----
    constexpr uint32_t cb_a = tt::CBIndex::c_0;  // A half tiles
    constexpr uint32_t cb_b = tt::CBIndex::c_1;  // B half tiles

    const uint32_t tile_bytes = get_tile_size(cb_a);

    // ---- Tensor accessor for input ----
    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);

    const uint32_t HtWt_out = Ht * Wt_out;

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t g_out = start_tile_out + i;

        // Map output tile id -> (nc, ht, wt_out) in output tensor
        const uint32_t nc  = g_out / HtWt_out;
        const uint32_t rem = g_out - nc * HtWt_out;
        const uint32_t ht  = rem / Wt_out;
        const uint32_t wt  = rem - ht * Wt_out;

        // Map to input tile ids (tile-aligned split along width tiles)
        const uint32_t in_base = nc * (Ht * Wt_in) + ht * Wt_in + wt;
        const uint32_t tile_a_id = in_base;
        const uint32_t tile_b_id = in_base + Wt_out;

        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        const uint32_t l1_a = get_write_ptr(cb_a);
        const uint32_t l1_b = get_write_ptr(cb_b);

        noc_async_read_tile(tile_a_id, in, l1_a);
        noc_async_read_tile(tile_b_id, in, l1_b);

        noc_async_read_barrier();

        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
"""

compute_kernel_src = r"""
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // ---- common runtime args ----
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    // ---- derive per-core work ----
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Configure SFPU datapath (formats inferred from CB metadata)
    init_sfpu(cb_a, cb_out);

    // Copy uses CB metadata; we assume cb_a and cb_b have identical data format (bf16)
    copy_tile_init(cb_a);

    // SFPU op inits
    relu_tile_init();
    mul_binary_tile_init();

    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // Load A and B into DST regs
        copy_tile(cb_a, 0, 0);  // DST0 = A
        copy_tile(cb_b, 0, 1);  // DST1 = B

        // DST1 = relu(DST1)
        relu_tile(1);

        // DST0 = DST0 * DST1
        mul_binary_tile(0, 1, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);

        tile_regs_release();

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
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
    // ---- common runtime args ----
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(1);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(4);

    // ---- derive per-core work ----
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_out =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t l1_read = get_read_ptr(cb_out);

        noc_async_write_tile(start_tile_out + i, out, l1_read);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
"""


def _num_tiles_for_tiled_tensor(shape) -> int:
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


def host(a: ttnn.Tensor, dim: int = -1) -> ttnn.Tensor:
    # Match ttnn::reglu constraints from op doc
    if dim not in (-1, 3):
        raise RuntimeError("reglu: only dim=-1 or dim=3 supported by this kernel")
    if dim == -1:
        dim = 3
    if len(list(a.shape)) != 4 or dim != 3:
        raise RuntimeError(f"reglu: this kernel expects a 4D tensor with dim=3, got shape={a.shape}, dim={dim}")

    if a.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError("reglu: this kernel expects TILE_LAYOUT input")
    if a.dtype != ttnn.bfloat16:
        raise RuntimeError("reglu: this kernel expects BF16 input")

    N, C, H, W = list(a.shape)
    if (W % 2) != 0:
        raise RuntimeError(f"reglu: last dimension must be even, got W={W}")

    # Tile-aligned split requirement for this manual implementation:
    # W must be divisible by 64 so that both halves are multiples of 32 elements.
    if (W % 64) != 0:
        raise RuntimeError(
            f"reglu: this kernel requires tile-aligned split (W % 64 == 0). Got W={W}."
        )

    W_out = W // 2
    out_shape = ttnn.Shape([N, C, H, W_out])

    device = a.device()
    out = ttnn.allocate_tensor_on_device(out_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # Tile geometry for mapping
    Ht = (H + 31) // 32
    Wt_in = (W + 31) // 32
    Wt_out = (W_out + 31) // 32
    if Wt_in != 2 * Wt_out:
        raise RuntimeError("reglu: internal tiling mismatch (expected Wt_in == 2*Wt_out)")

    num_tiles_out = _num_tiles_for_tiled_tensor(out_shape)
    if num_tiles_out == 0:
        return out

    grid_size = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    total_cores = grid_size.x * grid_size.y
    base_tiles_per_core = num_tiles_out // total_cores
    extra_tile_range = num_tiles_out % total_cores

    # CB sizing (bf16 tile = 2048 bytes)
    tile_bytes = 32 * 32 * 2
    tiles_per_cb = 4
    cb_total = tiles_per_cb * tile_bytes

    cb_a, cb_b, cb_out = 0, 1, 16

    cbs = [
        ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_a, data_format=ttnn.bfloat16, page_size=tile_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_b, data_format=ttnn.bfloat16, page_size=tile_bytes)],
        ),
        ttnn.CBDescriptor(
            total_size=cb_total,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_out, data_format=ttnn.bfloat16, page_size=tile_bytes)],
        ),
    ]

    # Compile-time args for TensorAccessors
    reader_ct = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    writer_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()

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

    program = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=cbs,
    )

    return ttnn.generic_op([a, out], program)