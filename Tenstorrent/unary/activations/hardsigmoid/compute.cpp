// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"  
#include "compute_kernel_api/eltwise_unary/activations.h"  // hardsigmoid_tile_init / hardsigmoid_tile
// #include "debug/dprint.h"  // (optional) enable for bring-up prints

namespace NAMESPACE {
void MAIN {
    // CT arg 0: per_core_tile_cnt (set by host)
    const uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr tt::CBIndex cb_in  = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Bind CBs & init
    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(static_cast<uint32_t>(cb_in));
    hardsigmoid_tile_init();

    for (uint32_t t = 0; t < per_core_tile_cnt; ++t) {
        acquire_dst();

        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        // Load front tile -> dst(0)
        // If your drop only has copy_tile_to_dst(cb, dst), use that instead.
        copy_tile(cb_in, /*src_idx=*/0, /*dst_idx=*/0);

        // (optional debug)
        // DPRINT_MATH({ DPRINT << "HSig in: " << TSLICE(cb_in, 0, SliceRange::hw0_32_16()) << ENDL(); });

        // Apply hardsigmoid in-place
        hardsigmoid_tile(/*idst=*/0);

        // Pack to output CB, then advance queues
        pack_tile(/*dst_idx=*/0, cb_out);
        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);

        release_dst();
    }
}
}  // namespace NAMESPACE
