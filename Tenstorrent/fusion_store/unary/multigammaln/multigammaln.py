# SPDX-License-Identifier: Apache-2.0
import struct
import ttnn


reader_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);

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

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    const uint32_t tile_bytes = get_tile_size(cb_in0);

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        noc_async_read_page(start_tile_id + i, in0, l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_in0, onetile);
    }
}
"""


compute_kernel_src = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

// --- SFPU split includes (only what we use) ---
#define SFPU_OP_RECIP_INCLUDE 1
#define SFPU_OP_LOG_INCLUDE 1
#define SFPU_OP_FILL_INCLUDE 1
#define SFPU_OP_BINOP_WITH_SCALAR_INCLUDE 1
#define SFPU_OP_UNARY_COMP_INCLUDE 1
#define SFPU_OP_WHERE_INCLUDE 1
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

// Binary SFPU ops on DST regs
#include "compute_kernel_api/eltwise_binary_sfpu.h"

// For log_tile_init<...>/log_tile<...> template variants (fast_and_approx)
#include "compute_kernel_api.h"

namespace NAMESPACE {

static constexpr uint32_t F32_0P5  = 0x3f000000u;
static constexpr uint32_t F32_1P0  = 0x3f800000u;
static constexpr uint32_t F32_1P5  = 0x3fc00000u;
static constexpr uint32_t F32_2P0  = 0x40000000u;
static constexpr uint32_t F32_3P0  = 0x40400000u;
static constexpr uint32_t F32_4P0  = 0x40800000u;
static constexpr uint32_t F32_4P5  = 0x40900000u;
static constexpr uint32_t F32_5P0  = 0x40a00000u;

// Produce x_shifted into requested DST reg from CB0
#define COPY_X_SHIFT0(dst_reg)  do { copy_tile(tt::CBIndex::c_0, 0, (dst_reg)); } while (0)
#define COPY_X_SHIFT05(dst_reg) do { copy_tile(tt::CBIndex::c_0, 0, (dst_reg)); sub_unary_tile((dst_reg), F32_0P5); } while (0)
#define COPY_X_SHIFT10(dst_reg) do { copy_tile(tt::CBIndex::c_0, 0, (dst_reg)); sub_unary_tile((dst_reg), F32_1P0); } while (0)
#define COPY_X_SHIFT15(dst_reg) do { copy_tile(tt::CBIndex::c_0, 0, (dst_reg)); sub_unary_tile((dst_reg), F32_1P5); } while (0)

// Compute lgamma(x_shift) into DST[1], then accumulate into DST[0] (sum).
// Uses: DST[1]=scratch/result, DST[2]=temp/temp_log/zero, DST[3]=scratch
#define LGAMMA_AND_ACCUMULATE(COPY_X_SHIFTED)                                                      \
    do {                                                                                            \
        /* temp = 1 */                                                                              \
        fill_tile(2, 1.0f);                                                                         \
                                                                                                    \
        /* term1: c1 / (x+0) */                                                                     \
        COPY_X_SHIFTED(1);                                                                          \
        recip_tile(1);                                                                              \
        mul_unary_tile(1, c1_bits);                                                                  \
        add_binary_tile_init();                                                                      \
        add_binary_tile(2, 1, 2);                                                                    \
                                                                                                    \
        /* term2: c2 / (x+1) */                                                                     \
        COPY_X_SHIFTED(1);                                                                          \
        add_unary_tile(1, F32_1P0);                                                                  \
        recip_tile(1);                                                                              \
        mul_unary_tile(1, c2_bits);                                                                  \
        add_binary_tile(2, 1, 2);                                                                    \
                                                                                                    \
        /* term3: c3 / (x+2) */                                                                     \
        COPY_X_SHIFTED(1);                                                                          \
        add_unary_tile(1, F32_2P0);                                                                  \
        recip_tile(1);                                                                              \
        mul_unary_tile(1, c3_bits);                                                                  \
        add_binary_tile(2, 1, 2);                                                                    \
                                                                                                    \
        /* term4: c4 / (x+3) */                                                                     \
        COPY_X_SHIFTED(1);                                                                          \
        add_unary_tile(1, F32_3P0);                                                                  \
        recip_tile(1);                                                                              \
        mul_unary_tile(1, c4_bits);                                                                  \
        add_binary_tile(2, 1, 2);                                                                    \
                                                                                                    \
        /* term5: c5 / (x+4) */                                                                     \
        COPY_X_SHIFTED(1);                                                                          \
        add_unary_tile(1, F32_4P0);                                                                  \
        recip_tile(1);                                                                              \
        mul_unary_tile(1, c5_bits);                                                                  \
        add_binary_tile(2, 1, 2);                                                                    \
                                                                                                    \
        /* term6: c6 / (x+5) */                                                                     \
        COPY_X_SHIFTED(1);                                                                          \
        add_unary_tile(1, F32_5P0);                                                                  \
        recip_tile(1);                                                                              \
        mul_unary_tile(1, c6_bits);                                                                  \
        add_binary_tile(2, 1, 2);                                                                    \
                                                                                                    \
        /* temp_log = log(temp) */                                                                   \
        log_tile_init<1u>();                                                                         \
        log_tile<1u>(2);                                                                             \
                                                                                                    \
        /* t_log = log(x + 4.5) */                                                                   \
        COPY_X_SHIFTED(1);                                                                          \
        add_unary_tile(1, F32_4P5);                                                                  \
        log_tile<1u>(1);                                                                             \
                                                                                                    \
        /* x_minus_half = x - 0.5 */                                                                 \
        COPY_X_SHIFTED(3);                                                                          \
        sub_unary_tile(3, F32_0P5);                                                                  \
                                                                                                    \
        /* (x - 0.5) * log(x + 4.5) */                                                               \
        mul_binary_tile_init();                                                                      \
        mul_binary_tile(3, 1, 1);                                                                    \
                                                                                                    \
        /* + 0.918938531357171 */                                                                    \
        add_unary_tile(1, c0_bits);                                                                  \
                                                                                                    \
        /* + log(temp) */                                                                            \
        add_binary_tile_init();                                                                      \
        add_binary_tile(1, 2, 1);                                                                    \
                                                                                                    \
        /* - (x + 4.5) */                                                                            \
        COPY_X_SHIFTED(3);                                                                          \
        add_unary_tile(3, F32_4P5);                                                                  \
        sub_binary_tile_init();                                                                      \
        sub_binary_tile(1, 3, 1);                                                                    \
                                                                                                    \
        /* where(x==1, 0, result); where(x==2, 0, result) */                                         \
        fill_tile(2, 0.0f);                                                                          \
        unary_eq_tile_init();                                                                        \
        where_tile_init();                                                                           \
                                                                                                    \
        COPY_X_SHIFTED(3);                                                                          \
        unary_eq_tile(3, F32_1P0);                                                                   \
        where_fp32_tile(3, 2, 1, 1);                                                                  \
                                                                                                    \
        COPY_X_SHIFTED(3);                                                                          \
        unary_eq_tile(3, F32_2P0);                                                                   \
        where_fp32_tile(3, 2, 1, 1);                                                                  \
                                                                                                    \
        /* sum += result */                                                                          \
        add_binary_tile_init();                                                                      \
        add_binary_tile(0, 1, 0);                                                                    \
    } while (0)

void MAIN {
    const uint32_t base_tiles_per_core = get_common_arg_val<uint32_t>(0);
    const uint32_t extra_tile_range    = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_size_x         = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_size_y         = get_common_arg_val<uint32_t>(3);

    const uint32_t c1_bits  = get_common_arg_val<uint32_t>(4);
    const uint32_t c2_bits  = get_common_arg_val<uint32_t>(5);
    const uint32_t c3_bits  = get_common_arg_val<uint32_t>(6);
    const uint32_t c4_bits  = get_common_arg_val<uint32_t>(7);
    const uint32_t c5_bits  = get_common_arg_val<uint32_t>(8);
    const uint32_t c6_bits  = get_common_arg_val<uint32_t>(9);
    const uint32_t c0_bits  = get_common_arg_val<uint32_t>(10);  // 0.918938531357171
    const uint32_t cpi_bits = get_common_arg_val<uint32_t>(11);  // 3.434189657547 (3*log(pi))

    const uint32_t my_x = get_absolute_logical_x();
    const uint32_t my_y = get_absolute_logical_y();
    const uint32_t core_idx = my_x + my_y * grid_size_x;

    const uint32_t n_tiles = base_tiles_per_core + (core_idx < extra_tile_range ? 1 : 0);

    constexpr uint32_t cb_in0  = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    // SFPU pipeline init (unpack/pack config)
    init_sfpu(cb_in0, cb_out0);

    // We repeatedly copy from CB0 -> DST regs
    copy_tile_init(cb_in0);

    // Scalar SFPU init (add/sub/mul by scalar bits)
    binop_with_scalar_tile_init();

    // Other SFPU init
    recip_tile_init();
    fill_tile_init();

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_in0, 1);

        tile_regs_acquire();

        // sum = 0
        fill_tile(0, 0.0f);

        // multigammaln(x) for p=4:
        // lgamma(x) + lgamma(x-0.5) + lgamma(x-1.0) + lgamma(x-1.5) + 3*log(pi)
        LGAMMA_AND_ACCUMULATE(COPY_X_SHIFT0);
        LGAMMA_AND_ACCUMULATE(COPY_X_SHIFT05);
        LGAMMA_AND_ACCUMULATE(COPY_X_SHIFT10);
        LGAMMA_AND_ACCUMULATE(COPY_X_SHIFT15);

        // + 3*log(pi)
        add_unary_tile(0, cpi_bits);

        tile_regs_commit();

        cb_pop_front(cb_in0, 1);
        cb_reserve_back(cb_out0, 1);

        tile_regs_wait();
        pack_tile(0, cb_out0);
        tile_regs_release();

        cb_push_back(cb_out0, 1);
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

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out0);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_out0, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_out0);

        noc_async_write_page(start_tile_id + i, out_acc, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_out0, onetile);
    }
}
"""


def _f32_to_u32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _num_tiles_from_shape(shape) -> int:
    dims = list(shape)
    if len(dims) == 0:
        return 0
    if len(dims) == 1:
        h, w = 1, dims[0]
        ht, wt = 1, (w + 31) // 32
        nc = 1
    else:
        h, w = dims[-2], dims[-1]
        ht, wt = (h + 31) // 32, (w + 31) // 32
        nc = 1
        for d in dims[:-2]:
            nc *= d
    return nc * ht * wt


def host(a: ttnn.Tensor) -> ttnn.Tensor:
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

    # CB sizing
    tiles_per_cb = 2
    bytes_per_elem = 4 if a.dtype == ttnn.float32 else 2
    tile_size_bytes = 32 * 32 * bytes_per_elem
    cb_total_bytes = tiles_per_cb * tile_size_bytes

    cb_in_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=a.dtype, page_size=tile_size_bytes),
        ],
    )
    cb_out_desc = ttnn.CBDescriptor(
        total_size=cb_total_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=16, data_format=out.dtype, page_size=tile_size_bytes),
        ],
    )

    # Lanczos constants + multigammaln constant (p=4 => 3*log(pi))
    c1 = _f32_to_u32_bits(76.18009172947146)
    c2 = _f32_to_u32_bits(-86.50532032941677)
    c3 = _f32_to_u32_bits(24.01409824083091)
    c4 = _f32_to_u32_bits(-1.231739572450155)
    c5 = _f32_to_u32_bits(0.1208650973866179e-2)
    c6 = _f32_to_u32_bits(-0.5395239384953e-5)
    c0 = _f32_to_u32_bits(0.918938531357171)     # 0.5*log(2*pi)
    cpi = _f32_to_u32_bits(3.434189657547)       # 3*log(pi)

    reader_ct_args = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    writer_ct_args = ttnn.TensorAccessorArgs(out).get_compile_time_args()
    compute_ct_args = []

    reader_common_rt_args = [
        a.buffer_address(),
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
    compute_common_rt_args = [
        base_tiles_per_core,
        extra_tile_range,
        grid_size.x,
        grid_size.y,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c0,
        cpi,
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
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=True,  # keep intermediates in fp32 regs for stability
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
        cbs=[cb_in_desc, cb_out_desc],
    )

    return ttnn.generic_op([a, out], program)