// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"  
#include "compute_kernel_api/eltwise_unary/activations.h"  // celu_tile_init / celu_tile

namespace NAMESPACE {

void MAIN {
    // CT arg 0: per_core_tile_cnt
    const uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    // RT args: [alpha_bits, alpha_recip_bits] (IEEE-754 float bit patterns)
    const uint32_t alpha_bits       = get_arg_val<uint32_t>(0);
    const uint32_t alpha_recip_bits = get_arg_val<uint32_t>(1);

    constexpr tt::CBIndex cb_in  = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(static_cast<uint32_t>(cb_in));
    celu_tile_init();

    for (uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        acquire_dst();

        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        // Load → CELU → pack
        // If your SDK only has the 2-arg copier, use: copy_tile_to_dst(cb_in, /*dst=*/0);
        copy_tile(cb_in, /*src_idx=*/0, /*dst_idx=*/0);
        celu_tile(/*idst=*/0, alpha_bits, alpha_recip_bits);
        pack_tile(/*dst_idx=*/0, cb_out);

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);

        release_dst();
    }
}

} // namespace NAMESPACE
