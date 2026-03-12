
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kernel_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/ptx>

namespace ptx = cuda::ptx;

// ============================================================
// WGMMA inline PTX helpers
// ============================================================

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// 64-bit WGMMA shared memory descriptor builder
// leading_bytes: usually 0
// stride_bytes:  8 * BK * sizeof(half)
// swizzle_mode:  1 = SWIZZLE_B128
__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr,
    uint32_t leading_bytes,
    uint32_t stride_bytes,
    uint32_t swizzle_mode,
    uint32_t base_offset = 0)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)(addr >> 4) & 0x3FFFULL;
    desc |= ((uint64_t)(leading_bytes >> 4) & 0x3FFFULL) << 16;
    desc |= ((uint64_t)(stride_bytes >> 4) & 0x3FFFULL) << 32;
    desc |= ((uint64_t)base_offset) << 49;
    desc |= ((uint64_t)swizzle_mode) << 62;
    return desc;
}

// ============================================================
// WGMMA m64n128k16, f16 inputs, f32 output
// Each warpgroup (128 threads) holds d[8][8] = 64 floats
// ScaleD=1: accumulate into d;  ScaleD=0: zero then store
// TransA=0, TransB=0: (64,K) and (128,K) physical SMEM layout
// ============================================================
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n128k16_f16(
    float d[8][8], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63},"
        " %64,"
        " %65,"
        " %66, %67, %68, %69, %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
          "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b),
          "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
          "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* addr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
    asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(smem_addr) : "memory");
}

// ============================================================
// TMA tensor map helper
// ============================================================
static PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled_fn() {
    void* driver_ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr,
                            cudaEnableDefault, &status);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(driver_ptr);
}

// Creates tensor map for a row-major half matrix [ROWS x COLS]
// tile shape: [tile_rows x tile_cols]
// With 128B swizzle for WGMMA compatibility
CUtensorMap create_tmap_f16(void* ptr, uint64_t rows, uint64_t cols,
                             uint32_t tile_rows, uint32_t tile_cols) {
    CUtensorMap tmap{};
    auto fn = get_cuTensorMapEncodeTiled_fn();
    // Column-major ordering: innermost dimension (cols) first
    uint64_t size[2]   = {cols, rows};
    uint64_t stride[1] = {cols * sizeof(__half)};  // bytes per row
    uint32_t box[2]    = {tile_cols, tile_rows};
    uint32_t es[2]     = {1, 1};
    CUresult res = fn(&tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, ptr,
        size, stride, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        printf("cuTensorMapEncodeTiled failed: %d\n", (int)res);
    }
    return tmap;
}

// ============================================================
// Tile configuration
// BM=128, BN=128, BK=64
// 2 warpgroups (256 threads total)
// WG0 = threads [0,127]   -> computes rows [bm, bm+64)   using sA[0..64*BK]
// WG1 = threads [128,255] -> computes rows [bm+64,bm+128) using sA[64*BK..128*BK]
// Both use the same sB[BN*BK]
//
// 3-stage pipeline SMEM layout per stage:
//   sA[BM * BK] = 128*64*2 = 16384 bytes   (both WG A tiles packed)
//   sB[BN * BK] = 128*64*2 = 16384 bytes
// Per stage: 32768 bytes
// 3 stages: 98304 bytes
// Plus 3 barriers x 8 bytes = 24 bytes
// Total: ~98328 bytes (~96KB) -- fits comfortably in H100's 228KB
// ============================================================

constexpr int BM2 = 128;
constexpr int BN2 = 128;
constexpr int BK2 = 64;
constexpr int NUM_STAGES = 3;
constexpr int THREADS_PER_BLOCK = 256;  // 2 warpgroups
// Each warpgroup covers BM2/2 = 64 rows (half of BM2)
constexpr int WG_ROWS = 64;  // rows per warpgroup

// WGMMA stride for m64n128k16 with BK2 cols:
// stride_bytes = 8 * BK2 * sizeof(half) = 8 * 64 * 2 = 1024 bytes
constexpr uint32_t WGMMA_STRIDE = 8 * BK2 * (uint32_t)sizeof(__half);

// ============================================================
// Vectorized epilogue helper:
// Store WG's 64x128 f32 accumulators through SMEM as f16,
// then write 128-bit (float4 = 8xf16) vectors to global C.
// SMEM reuse: after WGMMA is done, the sA/sB tiles for the consumed
// stage are no longer needed, so we repurpose SMEM for the epilogue.
//
// Accumulator layout for m64n128k16 (per warpgroup local thread index wl):
//   warp_id  = wl / 32          (0..3)
//   groupID  = (wl % 32) / 4   (0..7)
//   tid_grp  = wl % 4          (0..3)
//   r = ri*8+rj (0..63):
//     row = warp_id*16 + ((r>>1)&1)*8 + groupID    (0..63)
//     col = (r>>2)*8   + tid_grp*2   + (r&1)       (0..127)
// ============================================================
__device__ __forceinline__ void store_wg_accum(
    const float d[8][8],
    __half* smem_tile,          // 64*128 halfs of SMEM staging area
    __half* __restrict__ C_row, // pointer to C[row_base][0]
    int col_base,
    int M, int N,
    int row_base,
    int wl)                     // local thread index within warpgroup (0..127)
{
    // Step 1: scatter f32 -> f16 into SMEM at (row, col) positions
    int warp_id = wl / 32;
    int groupID = (wl % 32) / 4;
    int tid_grp = wl % 4;

    #pragma unroll
    for (int ri = 0; ri < 8; ri++) {
        #pragma unroll
        for (int rj = 0; rj < 8; rj++) {
            int r   = ri * 8 + rj;
            int row = warp_id * 16 + ((r >> 1) & 1) * 8 + groupID;
            int col = (r >> 2) * 8 + tid_grp * 2 + (r & 1);
            // smem_tile is [64][128] halfs
            smem_tile[row * BN2 + col] = __float2half(d[ri][rj]);
        }
    }

    __syncwarp();
    // Wait for all 128 threads in this WG to finish writing
    // We use a warpgroup-level sync -- need full __syncthreads for the tile
    // This is called from a context with a block sync before and after
}

// ============================================================
// Main kernel: BM=128, BN=128, BK=64, 3-stage pipeline
// 2 warpgroups: WG0 (threads 0-127) and WG1 (threads 128-255)
// ============================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) gemm_dual_wg(
    const __grid_constant__ CUtensorMap tmap_a,  // A: (M, K), tile (BM2, BK2)
    const __grid_constant__ CUtensorMap tmap_b,  // B: (N, K), tile (BN2, BK2)
    __half* __restrict__ C,
    int M, int N, int K)
{
    const int Tx  = threadIdx.x;               // 0..255
    const int wg  = Tx / 128;                  // 0 or 1  (which warpgroup)
    const int wl  = Tx % 128;                  // 0..127  local index within WG
    const int bm  = blockIdx.y * BM2;          // row offset
    const int bn  = blockIdx.x * BN2;          // col offset
    const int row_base_wg = bm + wg * WG_ROWS; // absolute row base for this WG

    extern __shared__ char smem_raw[];

    // SMEM layout:
    //   [NUM_STAGES stages] each stage = sA[BM2*BK2] + sB[BN2*BK2] halfs
    //   sA[s] at offset: s * (BM2*BK2 + BN2*BK2) * sizeof(half)
    //   sB[s] at offset: sA[s] + BM2*BK2*sizeof(half)
    //   barriers at end: NUM_STAGES * sizeof(uint64_t), 8-byte aligned
    constexpr int STAGE_HALFS = (BM2 * BK2 + BN2 * BK2);  // 128*64+128*64=16384
    constexpr int STAGE_BYTES = STAGE_HALFS * (int)sizeof(__half); // 32768

    // Helper lambdas to get stage pointers
    // sA for stage s: full BM2*BK2 tile (both WG halves packed)
    // WG0 uses top half: sA[s][0 .. 64*BK2)
    // WG1 uses bottom half: sA[s][64*BK2 .. 128*BK2)
    auto get_sA = [&](int s) -> __half* {
        return (__half*)(smem_raw + s * STAGE_BYTES);
    };
    auto get_sB = [&](int s) -> __half* {
        return (__half*)(smem_raw + s * STAGE_BYTES + BM2 * BK2 * sizeof(__half));
    };

    // Barriers after all stage data
    size_t bar_offset = (size_t)NUM_STAGES * STAGE_BYTES;
    bar_offset = (bar_offset + 7) & ~size_t(7);
    uint64_t* bar = (uint64_t*)(smem_raw + bar_offset);

    // Init mbarriers: thread 0 of block only
    if (Tx == 0) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; s++) {
            ptx::mbarrier_init(&bar[s], 1);
        }
    }
    __syncthreads();
    ptx::fence_proxy_async(ptx::space_shared);

    const int num_k_tiles = K / BK2;
    // TMA bytes per stage: full A tile (BM2*BK2 halfs) + B tile (BN2*BK2 halfs)
    const uint32_t total_tx = (uint32_t)(BM2 * BK2 + BN2 * BK2) * (uint32_t)sizeof(__half);

    // Accumulators: m64n128k16 -> d[8][8] = 64 floats per warpgroup thread
    float d[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            d[i][j] = 0.0f;

    // Per-stage parity trackers
    uint32_t parity[NUM_STAGES];
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) parity[s] = 0;

    // -------------------------------------------------------
    // Prologue: fill (NUM_STAGES-1) stages with TMA loads
    // (overlapped: issue all prefetches before waiting on any)
    // -------------------------------------------------------
    {
        int prefill = (num_k_tiles < NUM_STAGES - 1) ? num_k_tiles : (NUM_STAGES - 1);
        if (Tx == 0) {
            for (int s = 0; s < prefill; s++) {
                int k_col = s * BK2;
                int32_t ca[2] = {k_col, bm};   // A tile: row=bm, col=k_col
                int32_t cb[2] = {k_col, bn};   // B tile: row=bn, col=k_col
                ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                                          get_sA(s), &tmap_a, ca, &bar[s]);
                ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                                          get_sB(s), &tmap_b, cb, &bar[s]);
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                               ptx::space_shared, &bar[s], total_tx);
            }
        }
    }

    // -------------------------------------------------------
    // Main K-tile loop with 3-stage pipelining
    // -------------------------------------------------------
    for (int k = 0; k < num_k_tiles; ++k) {
        int cur = k % NUM_STAGES;
        int nxt = (k + NUM_STAGES - 1) % NUM_STAGES; // stage to prefetch into

        // STEP 1: Prefetch tile k+(NUM_STAGES-1) into nxt stage
        //         Issue BEFORE waiting so TMA runs during compute
        int prefetch_k = k + (NUM_STAGES - 1);
        if (prefetch_k < num_k_tiles && Tx == 0) {
            int k_col = prefetch_k * BK2;
            int32_t ca[2] = {k_col, bm};
            int32_t cb[2] = {k_col, bn};
            ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                                      get_sA(nxt), &tmap_a, ca, &bar[nxt]);
            ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                                      get_sB(nxt), &tmap_b, cb, &bar[nxt]);
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                           ptx::space_shared, &bar[nxt], total_tx);
        }

        // STEP 2: Wait for current stage
        while (!ptx::mbarrier_try_wait_parity(&bar[cur], parity[cur])) {}
        parity[cur] ^= 1;
        __syncthreads();

        // STEP 3: WGMMA compute on current stage
        // WG0 uses sA[cur][0..WG_ROWS*BK2), WG1 uses sA[cur][WG_ROWS*BK2..)
        __half* pA_base = get_sA(cur) + wg * WG_ROWS * BK2;
        __half* pB_base = get_sB(cur);

        // Fence accumulators before issuing WGMMA
        #pragma unroll
        for (int i = 0; i < 8; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                asm volatile("" : "+f"(d[i][j]) :: "memory");

        warpgroup_arrive();

        // BK2/16 = 4 k-steps within this tile (BK2=64, each step k16)
        // Accumulate: ScaleD=1
        #pragma unroll
        for (int ks = 0; ks < BK2 / 16; ks++) {
            // Advance smem pointer by ks*16 elements along K
            __half* pA = pA_base + ks * 16;
            __half* pB = pB_base + ks * 16;
            uint64_t desc_a = make_smem_desc(pA, 0, WGMMA_STRIDE, 1 /*SWIZZLE_B128*/);
            uint64_t desc_b = make_smem_desc(pB, 0, WGMMA_STRIDE, 1 /*SWIZZLE_B128*/);
            wgmma_m64n128k16_f16<1, 1, 1, 0, 0>(d, desc_a, desc_b);
        }

        warpgroup_commit_batch();

        // Fence accumulators after commit
        #pragma unroll
        for (int i = 0; i < 8; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                asm volatile("" : "+f"(d[i][j]) :: "memory");

        warpgroup_wait<0>();
        __syncthreads();
    }

    // -------------------------------------------------------
    // Epilogue: scatter f32 accumulators to SMEM as f16,
    // then coalesced float4 stores to global C.
    //
    // Each WG writes a 64 x 128 f16 tile.
    // WG0 writes C[bm..bm+64][bn..bn+128]
    // WG1 writes C[bm+64..bm+128][bn..bn+128]
    //
    // SMEM staging: reuse stage-0 sA/sB as a 128x128 f16 scratch.
    // WG0 uses the first 64 rows (sA of stage 0: 64*128 halfs = 16384 bytes)
    // WG1 uses the next 64 rows (sB of stage 0: 64*128 halfs -- but BN2*BK2=128*64=8192 halfs)
    // Wait -- we need 64*128 = 8192 halfs per WG.
    // sA[0] = BM2*BK2 = 128*64 = 8192 halfs = 16384 bytes (fits both WG staging areas)
    // WG0 staging: sA[0][0..8192) = first 8192 halfs
    // WG1 staging: sA[0][8192..16384) = next 8192 halfs (offset 64*BN2 = 64*128)
    // This gives each WG a 64x128 staging tile.
    // -------------------------------------------------------
    __syncthreads();  // ensure all WGMMA done and SMEM is free to reuse

    // Each warpgroup's staging area in SMEM: 64 rows x 128 cols = 8192 halfs
    // WG0: stage0 sA[0..8191]
    // WG1: stage0 sA[8192..16383]
    __half* smem_stage = get_sA(0) + wg * WG_ROWS * BN2;

    // Scatter accumulators into SMEM staging area
    {
        int warp_id = wl / 32;
        int groupID = (wl % 32) / 4;
        int tid_grp = wl % 4;
        #pragma unroll
        for (int ri = 0; ri < 8; ri++) {
            #pragma unroll
            for (int rj = 0; rj < 8; rj++) {
                int r   = ri * 8 + rj;
                int row = warp_id * 16 + ((r >> 1) & 1) * 8 + groupID;  // 0..63
                int col = (r >> 2) * 8 + tid_grp * 2 + (r & 1);         // 0..127
                smem_stage[row * BN2 + col] = __float2half(d[ri][rj]);
            }
        }
    }

    __syncthreads();  // all threads in block done writing SMEM

    // Coalesced global stores: each thread writes consecutive elements via float4 (8xf16=128b)
    // Total elements per WG: 64 * 128 = 8192 halfs = 1024 float4's
    // 128 threads -> 8 float4's per thread = 16 consecutive halfs per thread
    // Use a simple linear mapping: thread wl in [0,128) writes float4s at indices wl, wl+128, ...
    {
        int grow_base = row_base_wg;
        int gcol_base = bn;
        // We write the 64*128 WG tile linearly: element idx -> (row = idx/128, col = idx%128)
        // Using float4 (8 halfs at a time):
        // float4 index: each float4 covers 8 consecutive halfs in the same row
        // Row has 128/8 = 16 float4s per row, 64 rows -> 1024 float4s total
        // Distribute among 128 threads: 8 float4s each
        int ldc = N;  // leading dim of C (columns)
        float4* smem_f4 = reinterpret_cast<float4*>(smem_stage);
        // 1024 float4s / 128 threads = 8 per thread
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            int f4_idx = wl + i * 128;     // float4 index in the 64x128 tile
            int local_row = f4_idx / 16;   // row within the 64x128 tile (16 float4s per row)
            int local_f4  = f4_idx % 16;   // float4 index within the row
            int local_col = local_f4 * 8;  // element column (8 halfs per float4)
            int grow = grow_base + local_row;
            int gcol = gcol_base + local_col;
            if (grow < M && gcol + 7 < N) {
                // Write 8 consecutive halfs (1 float4) to global
                float4* dst = reinterpret_cast<float4*>(&C[grow * ldc + gcol]);
                dst[0] = smem_f4[f4_idx];
            } else if (grow < M) {
                // Partial row -- scalar fallback
                __half* src = smem_stage + local_row * BN2 + local_col;
                for (int e = 0; e < 8 && (gcol + e) < N; e++) {
                    C[grow * ldc + gcol + e] = src[e];
                }
            }
        }
    }

    // Invalidate barriers
    if (Tx == 0) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_inval(&bar[s]);
        }
    }
}

// ============================================================
// Host wrapper
// ============================================================
torch::Tensor gemm_dual_wg_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(K == B.size(1), "K dimension mismatch");
    TORCH_CHECK(M % BM2 == 0, "M must be divisible by BM=128");
    TORCH_CHECK(N % BN2 == 0, "N must be divisible by BN=128");
    TORCH_CHECK(K % BK2 == 0, "K must be divisible by BK=64");

    auto C = torch::empty({M, N}, A.options());

    __half* A_ptr = reinterpret_cast<__half*>(A.data_ptr<at::Half>());
    __half* B_ptr = reinterpret_cast<__half*>(B.data_ptr<at::Half>());
    __half* C_ptr = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    // Create TMA tensor maps
    // A: M rows, K cols, tile (BM2, BK2) = (128, 64)
    CUtensorMap tmap_a = create_tmap_f16(A_ptr, (uint64_t)M, (uint64_t)K,
                                          (uint32_t)BM2, (uint32_t)BK2);
    // B: N rows, K cols, tile (BN2, BK2) = (128, 64)
    CUtensorMap tmap_b = create_tmap_f16(B_ptr, (uint64_t)N, (uint64_t)K,
                                          (uint32_t)BN2, (uint32_t)BK2);

    // SMEM: NUM_STAGES * (BM2*BK2 + BN2*BK2) * sizeof(half) + barriers
    // = 3 * (128*64 + 128*64) * 2 + 3*8
    // = 3 * 16384 * 2 + 24   -- wait: BM2*BK2 = 128*64 = 8192, BN2*BK2 = 128*64 = 8192
    // per stage = (8192 + 8192) * 2 = 32768 bytes
    // 3 stages = 98304 bytes
    // + barriers = 24 bytes -> aligned to next 8 = ok
    // Total = ~98328 bytes
    constexpr int STAGE_BYTES_H = (BM2 * BK2 + BN2 * BK2) * (int)sizeof(__half);
    size_t smem_size = (size_t)NUM_STAGES * STAGE_BYTES_H + NUM_STAGES * sizeof(uint64_t) + 8;
    // Round up to 128-byte alignment
    smem_size = (smem_size + 127) & ~size_t(127);

    cudaError_t err = cudaFuncSetAttribute(
        gemm_dual_wg,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size);
    if (err != cudaSuccess) {
        printf("cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(err));
    }

    // Grid: (N/BN2, M/BM2) = (4096/128, 2048/128) = (32, 16) = 512 blocks
    dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2);
    dim3 block(THREADS_PER_BLOCK);  // 256 threads = 2 warpgroups

    gemm_dual_wg<<<grid, block, smem_size>>>(tmap_a, tmap_b, C_ptr, M, N, K);

    return C;
}
"""

kernel_cpp = "torch::Tensor gemm_dual_wg_cuda(torch::Tensor A, torch::Tensor B);"

module = load_inline(
    name='gemm_dual_wg',
    cpp_sources=kernel_cpp,
    cuda_sources=kernel_source,
    functions=['gemm_dual_wg_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=[
        '-O3',
        '-gencode', 'arch=compute_90a,code=sm_90a',
        '--expt-relaxed-constexpr',
        '-std=c++17',
    ],
    extra_ldflags=[''],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return module.gemm_dual_wg_cuda(A, B)

