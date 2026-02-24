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
    const uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // A
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // B
    constexpr auto cb_2   = tt::CBIndex::c_2;   // A^2 scratch
    constexpr auto cb_3   = tt::CBIndex::c_3;   // B^2 scratch
    constexpr auto cb_out = tt::CBIndex::c_16;  // A^2 + B^2

    constexpr uint32_t dst_reg = 0;

    for (uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        // ---- square(A) -> cb_2 ------------------------------------------
        // Bind unary path for (in0 -> cb_2) and init copy path from in0
        unary_op_init_common(cb_in0, cb_2);
        copy_tile_init(static_cast<uint32_t>(cb_in0));
        square_tile_init();

        acquire_dst();

        cb_wait_front(cb_in0, 1);
        cb_reserve_back(cb_2, 1);

        copy_tile(cb_in0, /*src_idx=*/0, /*dst_idx=*/0);
        square_tile(/*idst=*/0);
        pack_tile(/*dst_idx=*/0, cb_2);

        cb_pop_front(cb_in0, 1);
        cb_push_back(cb_2, 1);

        release_dst();

        // ---- square(B) -> cb_3 ------------------------------------------
        // Re-bind unary path for (in1 -> cb_3) and init copy path from in1
        unary_op_init_common(cb_in1, cb_3);
        copy_tile_init(static_cast<uint32_t>(cb_in1));
        square_tile_init();

        acquire_dst();

        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_3, 1);

        copy_tile(cb_in1, /*src_idx=*/0, /*dst_idx=*/0);
        square_tile(/*idst=*/0);
        pack_tile(/*dst_idx=*/0, cb_3);

        cb_pop_front(cb_in1, 1);
        cb_push_back(cb_3, 1);

        release_dst();

        // ---- add(A^2, B^2) -> cb_out ------------------------------------
        // Switch to binary family only now (after both squares are staged)
        binary_op_init_common(cb_2, cb_3, cb_out);
        add_tiles_init(cb_2, cb_3);

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
} // namespace NAMESPACE
