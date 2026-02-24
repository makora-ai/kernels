// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"  
#include "compute_kernel_api/eltwise_unary/activations.h"  // softsign_tile_init / softsign_tile

namespace NAMESPACE {
void MAIN {
    // CT arg 0: per_core_tile_cnt (passed from host)
    const uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr tt::CBIndex cb_in  = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Bind CBs, init copy path, and init SFPU softsign
    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(static_cast<uint32_t>(cb_in));
    softsign_tile_init();

    for (uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        acquire_dst();

        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        // Load → softsign → pack
        copy_tile(cb_in, /*src_idx=*/0, /*dst_idx=*/0);
        softsign_tile(/*idst=*/0);
        pack_tile(/*dst_idx=*/0, cb_out);

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);

        release_dst();
    }
}
}  // namespace NAMESPACE
