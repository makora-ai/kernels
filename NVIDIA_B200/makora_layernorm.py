import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

static __device__ __forceinline__ void warp_reduce(float &a, float &b) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

// Kernel 1: compute partial sums/sumsq packed into float2 for each (sample, block)
__global__ void ln_partial_sums_packed_kernel(const float* __restrict__ x,
                                              float2* __restrict__ partials,
                                              int B,
                                              long long N,
                                              int blocks_per_sample) {
    int global_block = blockIdx.x;
    int s = global_block / blocks_per_sample;          // sample index
    int lb = global_block % blocks_per_sample;         // local block within sample
    if (s >= B) return;

    const float* base = x + (long long)s * N;

    float lsum = 0.0f;
    float lsum2 = 0.0f;

    // Vectorized path
    long long N_vec = N / 4; // number of float4
    int stride_vec = blockDim.x * blocks_per_sample;
    long long j0 = (long long)threadIdx.x + (long long)lb * blockDim.x;

    const float4* base4 = reinterpret_cast<const float4*>(base);

    for (long long j = j0; j < N_vec; j += stride_vec) {
        float4 v4 = base4[j];
        float x0 = v4.x; float x1 = v4.y; float x2 = v4.z; float x3 = v4.w;
        lsum  += (x0 + x1 + x2 + x3);
        lsum2 += (x0*x0 + x1*x1 + x2*x2 + x3*x3);
    }

    // Handle tail elements if any (rare for typical shapes)
    int tail = (int)(N - N_vec * 4);
    if (tail && lb == 0 && threadIdx.x == 0) {
        for (int t = 0; t < tail; ++t) {
            float v = base[N_vec * 4 + t];
            lsum += v;
            lsum2 += v * v;
        }
    }

    // Warp-level reduction then cross-warp reduction
    int lane = threadIdx.x & (WARP_SIZE-1);
    int warp = threadIdx.x >> 5;
    warp_reduce(lsum, lsum2);

    __shared__ float sh_sum[8];
    __shared__ float sh_sumsq[8];
    if (lane == 0) {
        sh_sum[warp] = lsum;
        sh_sumsq[warp] = lsum2;
    }
    __syncthreads();

    if (warp == 0) {
        float bsum = (lane < 8) ? sh_sum[lane] : 0.0f;
        float bsum2 = (lane < 8) ? sh_sumsq[lane] : 0.0f;
        warp_reduce(bsum, bsum2);
        if (lane == 0) {
            partials[global_block] = make_float2(bsum, bsum2);
        }
    }
}

// Kernel 2: per-block computes row stats by reducing its row partials, then applies normalization+affine
__global__ void ln_fused_apply_kernel(const float* __restrict__ x,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      const float2* __restrict__ partials,
                                      float* __restrict__ y,
                                      int B,
                                      long long N,
                                      int blocks_per_sample,
                                      float eps) {
    int global_block = blockIdx.x;
    int s = global_block / blocks_per_sample;          // sample index
    int lb = global_block % blocks_per_sample;         // local block within sample
    if (s >= B) return;

    // 1) Recompute mean and inv_std locally from partials of this row (duplicated across the row's blocks)
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < blocks_per_sample; i += blockDim.x) {
        float2 p = partials[s * blocks_per_sample + i];
        sum   += p.x;
        sumsq += p.y;
    }

    int lane = threadIdx.x & (WARP_SIZE-1);
    int warp = threadIdx.x >> 5;
    warp_reduce(sum, sumsq);

    __shared__ float sh_sum[8];
    __shared__ float sh_sumsq[8];
    if (lane == 0) {
        sh_sum[warp] = sum;
        sh_sumsq[warp] = sumsq;
    }
    __syncthreads();

    float mean, inv_std;
    if (warp == 0) {
        float bsum = (lane < 8) ? sh_sum[lane] : 0.0f;
        float bsum2 = (lane < 8) ? sh_sumsq[lane] : 0.0f;
        warp_reduce(bsum, bsum2);
        if (lane == 0) {
            mean = bsum / (float)N;
            float var = bsum2 / (float)N - mean * mean;
            inv_std = rsqrtf(var + eps);
            sh_sum[0] = mean;     // reuse shared to broadcast
            sh_sumsq[0] = inv_std;
        }
    }
    __syncthreads();
    mean = sh_sum[0];
    inv_std = sh_sumsq[0];

    // 2) Apply normalization + affine on this block's strided tile with float4 vectorization
    const float* base_x = x + (long long)s * N;
    float* base_y = y + (long long)s * N;

    long long N_vec = N / 4;
    int stride_vec = blockDim.x * blocks_per_sample;
    long long j0 = (long long)threadIdx.x + (long long)lb * blockDim.x;

    const float4* x4 = reinterpret_cast<const float4*>(base_x);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    float4* y4 = reinterpret_cast<float4*>(base_y);

    for (long long j = j0; j < N_vec; j += stride_vec) {
        float4 xv = x4[j];
        float4 gv = g4[j];
        float4 bv = b4[j];
        float4 ov;
        ov.x = (xv.x - mean) * inv_std * gv.x + bv.x;
        ov.y = (xv.y - mean) * inv_std * gv.y + bv.y;
        ov.z = (xv.z - mean) * inv_std * gv.z + bv.z;
        ov.w = (xv.w - mean) * inv_std * gv.w + bv.w;
        y4[j] = ov;
    }

    // Tail handling: let one block and one thread handle leftover scalars
    int tail = (int)(N - N_vec * 4);
    if (tail && lb == 0 && threadIdx.x == 0) {
        long long base_tail = N_vec * 4;
        for (int t = 0; t < tail; ++t) {
            float v = base_x[base_tail + t];
            float n = (v - mean) * inv_std;
            base_y[base_tail + t] = n * gamma[base_tail + t] + beta[base_tail + t];
        }
    }
}

// Host wrapper
torch::Tensor layernorm_fused_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma and beta must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    int B = (int)x.size(0);
    long long N = x.numel() / x.size(0);

    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int blocks_per_sample = 64;
    int grid_size = B * blocks_per_sample;

    auto opts = x.options();
    // Packed partials as float2 stored in a float tensor
    auto partials_buf = torch::empty({grid_size, 2}, opts);

    ln_partial_sums_packed_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        reinterpret_cast<float2*>(partials_buf.data_ptr<float>()),
        B,
        N,
        blocks_per_sample);

    ln_fused_apply_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        reinterpret_cast<const float2*>(partials_buf.data_ptr<float>()),
        y.data_ptr<float>(),
        B,
        N,
        blocks_per_sample,
        static_cast<float>(eps));

    return y;
}
"""

cpp_src = "torch::Tensor layernorm_fused_forward(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps);"

ln_fused_ext = load_inline(
    name="custom_layernorm_fused_vec",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["layernorm_fused_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        # Keep LayerNorm module to manage parameters and epsilon
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self._ext = ln_fused_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ext.layernorm_fused_forward(x, self.ln.weight, self.ln.bias, float(self.ln.eps))

