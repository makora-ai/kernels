// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Read per_core_tile_cnt via get_compile_time_arg_val(0)
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    // CB indices
    constexpr auto cb_in0 = tt::CBIndex::c_0;   // A input
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // B input
    constexpr auto cb_2 = tt::CBIndex::c_2;     // scratch for A²
    constexpr auto cb_3 = tt::CBIndex::c_3;     // scratch for B²
    constexpr auto cb_out = tt::CBIndex::c_16;  // output = A² + B²

    constexpr uint32_t dst_reg = 0;

    // Initialize unary operations for squares
    unary_op_init_common(cb_in0, cb_2);
    unary_op_init_common(cb_in1, cb_3);
    copy_tile_init(static_cast<uint32_t>(cb_in0));
    copy_tile_init(static_cast<uint32_t>(cb_in1));
    square_tile_init();

    // Initialize binary operation for add (will be called later)
    binary_op_init_common(cb_2, cb_3, cb_out);
    add_tiles_init(cb_2, cb_3);

    for (uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        // For A: cb_wait_front(in0,1) → acquire_dst → copy_tile(in0,0,0) → square_tile(0) → cb_reserve_back(c_2,1) → pack_tile(0,c_2) → cb_pop_front(in0,1) → cb_push_back(c_2,1) → release_dst
        acquire_dst();
        
        cb_wait_front(cb_in0, 1);
        cb_reserve_back(cb_2, 1);
        copy_tile(cb_in0, /*src_idx=*/0, /*dst_idx=*/0);
        square_tile(/*idst=*/0);
        pack_tile(/*dst_idx=*/0, cb_2);
        cb_pop_front(cb_in0, 1);
        cb_push_back(cb_2, 1);
        
        release_dst();

        // For B: same into c_3
        acquire_dst();
        
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_3, 1);
        copy_tile(cb_in1, /*src_idx=*/0, /*dst_idx=*/0);
        square_tile(/*idst=*/0);
        pack_tile(/*dst_idx=*/0, cb_3);
        cb_pop_front(cb_in1, 1);
        cb_push_back(cb_3, 1);
        
        release_dst();

        // Then cb_wait_front(c_2,1); cb_wait_front(c_3,1); acquire_dst; cb_reserve_back(cb_out,1); add_tiles(c_2,c_3,0,0,dst_reg); pack_tile(dst_reg, cb_out); cb_pop_front(c_2,1); cb_pop_front(c_3,1); cb_push_back(cb_out,1); release_dst;
        acquire_dst();
        
        cb_wait_front(cb_2, 1);
        cb_wait_front(cb_3, 1);
        cb_reserve_back(cb_out, 1);
        add_tiles(cb_2, cb_3, /*src0=*/0, /*src1=*/0, /*dst=*/dst_reg);
        pack_tile(dst_reg, cb_out);
        cb_pop_front(cb_2, 1);
        cb_pop_front(cb_3, 1);
        cb_push_back(cb_out, 1);
        
        release_dst();
    }
}
}  // namespace NAMESPACE
