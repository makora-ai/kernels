# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn

# -------------------------------------------------------------------------------------------------
# Reader Kernel
#   - Streams input tiles into cb0
#   - Streams per-tile-selected mask tile (from 65-tile mask bank) into cb1
# -------------------------------------------------------------------------------------------------
reader_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    // ---- common runtime args ----
    const uint32_t in_addr        = get_common_arg_val<uint32_t>(0);
    const uint32_t mask_bank_addr = get_common_arg_val<uint32_t>(1);

    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(2);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(4);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(5);

    const int32_t diag = (int32_t)get_common_arg_val<uint32_t>(6);
    const uint32_t Ht  = get_common_arg_val<uint32_t>(7);
    const uint32_t Wt  = get_common_arg_val<uint32_t>(8);

    // ---- core index -> work range ----
    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t start_tile_id =
        core_idx * base_tiles_per_core + (core_idx < extra_tile_range ? core_idx : extra_tile_range);
    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    // ---- CBs ----
    constexpr uint32_t cb_in   = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask = tt::CBIndex::c_1;

    const uint32_t in_tile_bytes = get_tile_size(cb_in);
    const uint32_t m_tile_bytes  = get_tile_size(cb_mask);

    // ---- accessors ----
    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in_acc = TensorAccessor(in_args, in_addr, in_tile_bytes);

    constexpr auto m_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto m_acc = TensorAccessor(m_args, mask_bank_addr, m_tile_bytes);

    const uint32_t HtWt = Ht * Wt;

    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t g = start_tile_id + i;

        // tile row/col within the last-2D plane (independent per batch)
        const uint32_t rem = (HtWt == 0) ? 0 : (g % HtWt);
        const uint32_t tr = (Wt == 0) ? 0 : (rem / Wt);
        const uint32_t tc = (Wt == 0) ? 0 : (rem % Wt);

        // Condition: (global_c - global_r) >= diag
        // global_c - global_r = 32*(tc-tr) + (cc-rr)
        // => (cc-rr) >= diag - 32*(tc-tr)  := k
        const int32_t delta = (int32_t)tc - (int32_t)tr;
        int32_t k = diag - (int32_t)(32 * delta);

        // Clamp to [-32, 32] for the 65-tile mask bank
        if (k < -32) k = -32;
        if (k >  32) k =  32;
        const uint32_t mask_tile_id = (uint32_t)(k + 32);  // 0..64

        cb_reserve_back(cb_in, 1);
        cb_reserve_back(cb_mask, 1);

        const uint32_t l1_in_addr = get_write_ptr(cb_in);
        const uint32_t l1_m_addr  = get_write_ptr(cb_mask);

        noc_async_read_page(g, in_acc, l1_in_addr);
        noc_async_read_page(mask_tile_id, m_acc, l1_m_addr);

        noc_async_read_barrier();

        cb_push_back(cb_in, 1);
        cb_push_back(cb_mask, 1);
    }
}
"""

# -------------------------------------------------------------------------------------------------
# Compute Kernel
#   - out = in * mask  (FPU eltwise mul)
#   NOTE: Keep CB wait/reserve/pop/push ordering aligned with known-good eltwise patterns to avoid
#         cross-TRISC ordering hazards / deadlocks.
# -------------------------------------------------------------------------------------------------
compute_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
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

    constexpr uint32_t cb_in   = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask = tt::CBIndex::c_1;
    constexpr uint32_t cb_out  = tt::CBIndex::c_16;

    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_in, cb_mask, cb_out);
    mul_tiles_init(cb_in, cb_mask);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_in, 1);
        cb_wait_front(cb_mask, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        mul_tiles(cb_in, cb_mask, 0, 0, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(dst0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_in, 1);
        cb_pop_front(cb_mask, 1);
        cb_push_back(cb_out, 1);
    }
}
}  // namespace NAMESPACE
"""

# -------------------------------------------------------------------------------------------------
# Writer Kernel
#   - Drains cb16 and writes tiles back to output tensor pages
# -------------------------------------------------------------------------------------------------
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
    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_acc = TensorAccessor(out_args, out_addr, out_tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_out);

        noc_async_write_page(start_tile_id + i, out_acc, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
"""

# Simple per-(device,dtype) cache for the 65-tile mask bank tensor
_MASK_BANK_CACHE = {}


def _get_grid_size(device):
    # Prefer compute_grid_size() if present (more likely to exclude non-worker cores on some stacks)
    fn = getattr(device, "compute_grid_size", None)
    if callable(fn):
        return fn()
    return device.compute_with_storage_grid_size()


def _get_or_create_mask_bank(device, dtype: ttnn.DataType):
    key = (id(device), str(dtype))
    if key in _MASK_BANK_CACHE:
        return _MASK_BANK_CACHE[key]

    ar = torch.arange(32, dtype=torch.int32)
    cc = ar.view(1, 32)
    rr = ar.view(32, 1)
    base = cc - rr  # (c-r) in [-31..31]

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    bank = []
    for k in range(-32, 33):
        bank.append((base >= k).to(dtype=torch_dtype))
    mask_bank_host = torch.stack(bank, dim=0)  # [65, 32, 32]

    mask_bank = ttnn.from_torch(mask_bank_host, layout=ttnn.TILE_LAYOUT, device=device)
    _MASK_BANK_CACHE[key] = mask_bank
    return mask_bank


def host(a: ttnn.Tensor, diag: int = 0) -> ttnn.Tensor:
    device = a.device()

    # This kernel assumes tiled input/output and tile-granular DRAM pages.
    # (ttnn.triu internally operates in TILE layout as well.)
    assert a.layout == ttnn.TILE_LAYOUT, "This custom triu kernel currently supports TILE_LAYOUT input only."

    out = ttnn.allocate_tensor_on_device(ttnn.Shape(a.shape), a.dtype, a.layout, device)

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

    mask_bank = _get_or_create_mask_bank(device, a.dtype)

    grid_size = _get_grid_size(device)
    total_cores = grid_size.x * grid_size.y
    base_tiles_per_core = num_tiles // total_cores
    extra_tile_range = num_tiles % total_cores

    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    bytes_per_elem = 4 if a.dtype == ttnn.float32 else 2
    tile_size_bytes = 32 * 32 * bytes_per_elem
    tiles_per_cb = 4  # slightly deeper buffering to reduce likelihood of pipeline stalls
    cb_total_bytes = tiles_per_cb * tile_size_bytes

    cb_in_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=a.dtype, page_size=tile_size_bytes)],
    )
    cb_mask_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=a.dtype, page_size=tile_size_bytes)],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=a.dtype, page_size=tile_size_bytes)],
    )

    reader_ct_args = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    reader_ct_args.extend(ttnn.TensorAccessorArgs(mask_bank).get_compile_time_args())
    writer_ct_args = ttnn.TensorAccessorArgs(out).get_compile_time_args()

    diag_u32 = int(diag) & 0xFFFFFFFF
    reader_common_rt_args = [
        a.buffer_address(),
        mask_bank.buffer_address(),
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
        diag_u32,
        Ht,
        Wt,
    ]
    compute_common_rt_args = [base_tiles_per_core, extra_tile_range, grid_size.x, grid_size.y]
    writer_common_rt_args = [out.buffer_address(), base_tiles_per_core, extra_tile_range, grid_size.x, grid_size.y]

    reader_k = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=[],
        common_runtime_args=reader_common_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_config = ttnn.ComputeConfigDescriptor()
    if a.dtype == ttnn.float32:
        compute_config.fp32_dest_acc_en = True

    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=[],
        runtime_args=[],
        common_runtime_args=compute_common_rt_args,
        config=compute_config,
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
        cbs=[cb_in_desc, cb_mask_desc, cb_out_desc],
    )

    return ttnn.generic_op([a, mask_bank, out], program_descriptor)