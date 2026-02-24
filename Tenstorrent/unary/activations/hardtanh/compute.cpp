// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/hardtanh.h"  // hardtanh_tile_init / hardtanh_tile
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"  

namespace NAMESPACE {
void MAIN {
    // CT arg 0: per_core_tile_cnt (from host)
    const uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    // RT args: [min_bits, max_bits] as IEEE-754 float bit patterns
    const uint32_t min_bits = get_arg_val<uint32_t>(0);
    const uint32_t max_bits = get_arg_val<uint32_t>(1);

    constexpr tt::CBIndex cb_in  = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(static_cast<uint32_t>(cb_in));
    hardtanh_tile_init();

    for (uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        acquire_dst();

        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        // Load → HardTanh → pack
        // If your drop only has 2-arg copier, use: copy_tile_to_dst(cb_in, /*dst=*/0);
        copy_tile(cb_in, /*src_idx=*/0, /*dst_idx=*/0);
        hardtanh_tile(/*idst=*/0, /*param0=*/min_bits, /*param1=*/max_bits);
        pack_tile(/*dst_idx=*/0, cb_out);

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);

        release_dst();
    }
}
}  // namespace NAMESPACE
