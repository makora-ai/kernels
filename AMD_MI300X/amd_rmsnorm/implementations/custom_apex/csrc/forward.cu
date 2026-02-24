#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"


template <typename T> __device__
T rsqrt(T v) {
    return T(1) / sqrt(v);
}

__device__ float rsqrt(float v) {
    return rsqrtf(v);
}


template<typename T, typename U, bool Residual, int GpuWarpSize, bool Half> __device__
void cuWelfordMuSigma2(
    const T* __restrict__ vals,
    const T* __restrict__ residual,
    const int n1,
    const int n2,
    const int i1,
    U& mu,
    U& sigma2,
    U* shared)
{
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensor is contiguous
    // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
    //
    // compute variance and mean over n2
    // printf("(%i) Hi! %i %i %i %i\n", threadIdx.x, n1, n2, i1, (int)Half);
    U count = U(0);
    mu = U(0);
    sigma2 = U(0);
    if (i1 < n1) {
        // one warp normalizes one n1 index,
        // synchronization is implicit
        // initialize with standard Welford algorithm
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        const T* lvals = vals + i1*n2;
        const T* lres = nullptr;
        if constexpr (Residual) {
            lres = residual + i1*n2;
        }

        int l = 0;
        if constexpr (Half) {
            l = 8*thrx;
            if ((((size_t)lvals)&3) != 0) {
                // 16 bit alignment
                // first thread consumes first point
                if (thrx == 0) {
                    U curr = static_cast<U>(lvals[0]);
                    sigma2 += curr*curr;
                }
                ++l;
            }
        }
        else {
            l = 4*thrx;
        }

        if constexpr (Half) {
            for (; l+7 < n2; l += 8*numx) {
                for (int k = 0; k < 8; k+=2) {
                    float2 curr;
                    if constexpr (Residual) {
                        curr = __half22float2(*((__half2*)(lvals+l+k))) + __half22float2(*((__half2*)(lres+l+k)));
                    } else {
                        curr = __half22float2(*((__half2*)(lvals+l+k)));
                    }
                    curr = curr*curr;
                    sigma2 += curr.x + curr.y;
                }
            }
        } else {
            for (; l+3 < n2; l+=4*numx) {
                for (int k = 0; k < 4; ++k) {
                    U curr;
                    if constexpr (Residual) {
                        curr = static_cast<U>(lvals[l+k] + lres[l+k]);
                    } else {
                        curr = static_cast<U>(lvals[l+k]);
                    }
                    sigma2 += curr*curr;
                    // printf("(%i) curr=%f, sigma2=%f\n", threadIdx.x, (float)curr, (float)sigma2);
                }
            }
        }

        for (; l < n2; ++l) {
            U curr;
            if constexpr (Residual) {
                curr = static_cast<U>(lvals[l] + lres[l]);
            } else {
                curr = static_cast<U>(lvals[l]);
            }
            sigma2 += curr*curr;
            // printf("(%i) curr=%f, sigma2=%f\n", threadIdx.x, (float)curr, (float)sigma2);
        }

        // intra-warp reductions
        for (int stride = GpuWarpSize / 2; stride > 0; stride /= 2) {
            U sigma2B = WARP_SHFL_DOWN(sigma2, stride);
            sigma2 += sigma2B;
        }

        // threadIdx.x == 0 has correct values for each warp
        // inter-warp reductions
        if (blockDim.y > 1) {
            for (int offset = blockDim.y/2; offset > 0; offset /= 2) {
                // upper half of warps write to shared
                if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
                    const int wrt_y = threadIdx.y - offset;
                    shared[2*wrt_y+1] = sigma2;
                }
                __syncthreads();
                // lower half merges
                if (threadIdx.x == 0 && threadIdx.y < offset) {
                    U sigma2B = shared[2*threadIdx.y+1];
                    sigma2 += sigma2B;
                }
                __syncthreads();
            }
            // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                shared[1] = sigma2;
            }
            __syncthreads();
            sigma2 = shared[1]/U(n2);
            // don't care about final value of count, we know count == n2
        } else {
            sigma2 = WARP_SHFL(sigma2/U(n2), 0);
        }
    }
}


template<typename T, typename U, typename V, bool Training, bool Residual, int GpuWarpSize, bool Half> __global__
void cuApplyRMSNorm_(
    V* __restrict__ output_vals,
    U* __restrict__ invvar,
    const T* __restrict__ vals,
    const T* __restrict__ residual,
    T* __restrict__ inter_out,
    const int n1,
    const int n2,
    const U epsilon,
    const V* __restrict__ gamma)
{
    // Assumptions:
    // 1) blockDim.x == warpSize
    // 2) Tensors are contiguous
    //
    extern __shared__ U _shared[];
    U* shared = _shared;

    U mu, sigma2;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int numx = blockDim.x * blockDim.y;

    for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
        cuWelfordMuSigma2<T, U, Residual, GpuWarpSize, Half>(vals, residual, n1, n2, i1, mu, sigma2, shared);
        // printf("(%i) %f %f\n", threadIdx.x, (float)mu, (float)sigma2);

        const T* lvals = vals + i1*n2;
        const T* lresidual = nullptr;
        T* lvals_inter = nullptr;
        if constexpr (Residual) {
            lresidual = residual + i1*n2;
            if constexpr (Training)
                lvals_inter = inter_out + i1*n2;
        }

        V* ovals = output_vals + i1*n2;
        U c_invvar = rsqrt(sigma2 + epsilon);

        for (int i = thrx; i < n2; i += numx) {
            U curr = static_cast<U>(lvals[i]);

            if constexpr (Residual) {
                curr += static_cast<U>(lresidual[i]);

                if constexpr (Training)
                    lvals_inter[i] = static_cast<T>(curr);
            }

            ovals[i] = gamma[i] * static_cast<V>(curr * c_invvar);
        }

        if constexpr (Training) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                invvar[i1] = c_invvar;
            }
        }
        __syncthreads();
    }
}


template<typename T, typename U, typename V>
void dispatch_cuda_rms_norm(
    V* output,
    U* invvar,
    const T* input,
    const T* residual,
    T* inter_out,
    int n1,
    int n2,
    double epsilon,
    const V* gamma,
    bool half)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);

    int block_dim = 0;
    int shared = 0;
    block_dim = std::max(1, std::round(n2 / 1024.0));
    block_dim = block_dim != 1 ? std::round(block_dim * 0.5) * 2 : block_dim;
    block_dim = std::min(block_dim, 4);
    dim3 threads(warp_size, block_dim, 1);

    if (threads.y > 1)
        shared = threads.y*sizeof(U) + (threads.y/2)*sizeof(U);

#define _DISPATCH(Training, Residual, Warp, Half) \
    /* std::cout << "(" << threads.x << ", " << threads.y << ") (" << blocks.x << ", " << blocks.y << ") " << shared \
        << " " << sizeof(T) << " " << sizeof(U) << " " << sizeof(V) << " " << Training << " " << Residual << " " << Warp << " " << Half \
        << "\n    0x" << std::hex << (std::uint64_t)output \
        << "\n    0x" << std::hex << (std::uint64_t)invvar \
        << "\n    0x" << std::hex << (std::uint64_t)input \
        << "\n    0x" << std::hex << (std::uint64_t)residual \
        << "\n    0x" << std::hex << (std::uint64_t)inter_out \
        << "\n    0x" << std::hex << (std::uint64_t)gamma \
        << std::dec \
        << std::endl; */ \
    auto kernel = &cuApplyRMSNorm_<T, U, V, Training, Residual, Warp, Half>; \
    kernel<<<blocks, threads, shared, stream>>>( \
        output, \
        invvar, \
        input, \
        residual, \
        inter_out, \
        n1, \
        n2, \
        U(epsilon), \
        gamma)

#define DISPATCH_KERNEL_(Training, Residual, Warp) { \
    if (half) { \
        _DISPATCH(Training, Residual, Warp, true); \
    } else { \
        _DISPATCH(Training, Residual, Warp, false); \
    } \
}

#define DISPATCH_KERNEL(Training, Residual) \
    switch (warp_size) { \
        case 32: \
            DISPATCH_KERNEL_(Training, Residual, 32) \
            break; \
        case 64: \
            DISPATCH_KERNEL_(Training, Residual, 64) \
            break; \
        default: \
            throw std::runtime_error("Unsupported warp size"); \
    }

    if (residual != NULL) {
        if (invvar != NULL) {
            DISPATCH_KERNEL(true, true)
        } else {
            DISPATCH_KERNEL(false, true)
        }
    }
    else {
        if (invvar != NULL) {
            DISPATCH_KERNEL(true, false)
        } else {
            DISPATCH_KERNEL(false, false)
        }
    }

#undef _DISPATCH
#undef DISPATCH_KERNEL
#undef DISPATCH_KERNEL_
}


void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* invvar,
    at::Tensor* input,
    at::Tensor* residual,
    at::Tensor* inter_out,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon)
{
    using namespace at;
    bool half = input->scalar_type() == at::ScalarType::Half; // || input->scalar_type() == at::ScalarType::BFloat16;

    DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "cuda_rms_norm",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        dispatch_cuda_rms_norm<scalar_t_in, accscalar_t, scalar_t_out>(
            output->data_ptr<scalar_t_out>(),
            invvar != NULL ? invvar->data_ptr<accscalar_t>() : NULL,
            input->data_ptr<scalar_t_in>(),
            residual != NULL ? residual->data_ptr<scalar_t_in>() : NULL,
            inter_out != NULL ? inter_out->data_ptr<scalar_t_in>() : NULL,
            n1,
            n2,
            epsilon,
            gamma != NULL ? gamma->data_ptr<scalar_t_out>() : NULL,
            half
        );
    )
}