#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"


template <typename T, typename U, typename V> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* inter_out,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const double eps)
{
    int i1 = i1_block+thr_load_row_off;
    if (i1 < i1_end) {
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1*n2+i2;
            int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
            if (i2<n2) {
                U c_h = static_cast<U>(inter_out[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);
                warp_buf2[write_idx] = curr_dout * c_h * invvar[i1];
            } else {
                warp_buf2[write_idx] = U(0);
            }
        }
    } else {
        for (int k = 0; k < blockDim.y; ++k) {
            int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
            warp_buf2[write_idx] = U(0);
        }
    }
}


template <typename T, typename U, typename V> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* inter_out,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const double eps)
{
    int i1 = i1_block+thr_load_row_off;
    if (i1 < i1_end) {
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1*n2+i2;
            int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
            if (i2<n2) {
                U c_h = static_cast<U>(inter_out[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);
                warp_buf2[write_idx] += curr_dout * c_h * invvar[i1];
            }
        }
    }
}


template <typename T, typename U, typename V> __global__
void cuComputePartGradGammaBeta(
    const V* __restrict__ dout,
    const T* __restrict__ inter_out,
    const int n1,
    const int n2,
    const U* __restrict__ invvar,
    U epsilon,
    const V* __restrict__ gamma,
    U* part_grad_gamma,
    const double eps)
{
    const int numsegs_n1 = (n1+blockDim.y*blockDim.y-1) / (blockDim.y*blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y+1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x+1;
    const int thr_load_col_off = (threadIdx.x*blockDim.y)&(blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x*blockDim.y)/blockDim.x + threadIdx.y*blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    extern __shared__ U _shared[];
    U* buf = &_shared[0]; // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    // printf("(%i, %i), (%i, %i) A\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    cuLoadWriteStridedInputs<T, U, V>(i1_beg, thr_load_row_off, thr_load_col_off, i2_off, row_stride, warp_buf1, warp_buf2, inter_out, dout, i1_end, n2, invvar, gamma, eps);
    for (int i1_block = i1_beg+blockDim.y*blockDim.y; i1_block < i1_end; i1_block+=blockDim.y*blockDim.y) {
        cuLoadAddStridedInputs<T, U, V>(i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride, warp_buf1, warp_buf2, inter_out, dout, i1_end, n2, invvar, gamma, eps);
    }
    // printf("(%i, %i), (%i, %i) B\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0; k < blockDim.y; ++k) {
        int row1 = threadIdx.y + k*blockDim.y;
        int idx1 = row1*row_stride + threadIdx.x;
        acc2 += warp_buf2[idx1];
    }
    warp_buf2[threadIdx.y*row_stride+threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y/2; offset > 1; offset /= 2) {
        if (threadIdx.y < offset) {
            int row1 = threadIdx.y;
            int row2 = threadIdx.y + offset;
            int idx1 = row1*row_stride + threadIdx.x;
            int idx2 = row2*row_stride + threadIdx.x;
            warp_buf2[idx1] += warp_buf2[idx2];
        }
        __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + 1;
        int idx1 = row1*row_stride + threadIdx.x;
        int idx2 = row2*row_stride + threadIdx.x;
        part_grad_gamma[blockIdx.y*n2+i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}



template <typename U, typename V> __global__
void cuComputeGradGammaBeta(
    const U* part_grad_gamma,
    const int part_size,
    const int n1,
    const int n2,
    V* grad_gamma)
{
    // sum partial gradients for gamma and beta
    extern __shared__ U shared[];
    U* buf = &shared[0];
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        // each warp does sequential reductions until reduced part_size is num_warps
        const int num_warp_reductions = part_size / blockDim.y;
        U sum_gamma = U(0);
        U sum_beta = U(0);
        const U* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;

        for (int warp_offset = 0;  warp_offset < num_warp_reductions;  ++warp_offset) {
            sum_gamma += part_grad_gamma_ptr[warp_offset*n2];
        }
        // inter-warp reductions
        for (int offset = blockDim.y/2;  offset >= 1;  offset /= 2) {
            // top half write to shared memory
            if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
                const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_gamma;
            }
            __syncthreads();
            // bottom half sums
            if (threadIdx.y < offset) {
                const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_gamma += buf[read_idx];
            }
            __syncthreads();
        }
         // write out fully summed gradients
        if (threadIdx.y == 0) {
            grad_gamma[i2] = sum_gamma;
        }
    }
}


template<typename T, typename U, typename V, bool Residual> __global__
void cuComputeGradInput(
    const V* __restrict__ dout,
    const T* __restrict__ inter_out,
    const int n1,
    const int n2,
    const U* __restrict__ invvar,
    U epsilon,
    const V* gamma,
    T* grad_input,
    T* grad_residual)
{
    extern __shared__ U shared[];

    for (int i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
        U sum_loss1 = U(0);
        U sum_loss2 = U(0);
        const U c_invvar = invvar[i1];
        const T* k_inter_out = inter_out + i1*n2;
        const V* k_dout = dout + i1*n2;
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        
        // Optimization for ROCm MI100
        for( int l = 0; l < n2 ; l += numx) {
            int idx = l + thrx;
            const U gamma_idx = static_cast<U>((idx<n2) ? gamma[idx] : V(0));
            const U c_h = static_cast<U>((idx<n2) ? k_inter_out[idx] : T(0));
            const U c_loss = static_cast<U>((idx<n2) ? k_dout[idx] : V(0));
            sum_loss2 += c_loss * gamma_idx * c_h * c_invvar;
        }
        
        // intra-warp reductions
        for (int mask = blockDim.x/2;    mask > 0;    mask /= 2) {
            sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
        }
        // inter-warp reductions
        if (blockDim.y > 1) {
            U* buf = &shared[0];
            for (int offset = blockDim.y/2; offset > 0; offset /= 2) {
                // upper half of warps write to shared
                if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
                    const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                    buf[2*wrt_i+1] = sum_loss2;
                }
                __syncthreads();
                // lower half merges
                if (threadIdx.y < offset) {
                    const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
                    sum_loss2 += buf[2*read_i+1];
                }
                __syncthreads();
            }
            if (threadIdx.y == 0) {
                buf[2*threadIdx.x+1] = sum_loss2;
            }
            __syncthreads();
            if (threadIdx.y !=0) {
                sum_loss2 = buf[2*threadIdx.x+1];
            }
        }
        // all threads now have the two sums over l
        U fH = (U)n2;
        U term1 = (U(1) / fH) * c_invvar;
        T* k_grad_input = grad_input + i1*n2;
        T* k_grad_resdiaul = nullptr;
        if constexpr (Residual)
            k_grad_resdiaul = grad_residual + i1*n2;

        for (int l = thrx; l < n2; l += numx) {
            const U c_h = static_cast<U>(k_inter_out[l]);
            const U c_loss = static_cast<U>(k_dout[l]);
            U f_grad_input = fH * c_loss * gamma[l];
            f_grad_input -= c_h * c_invvar * sum_loss2;
            f_grad_input *= term1;
            k_grad_input[l] = static_cast<T>(f_grad_input);
            if constexpr (Residual)
                k_grad_resdiaul[l] = static_cast<T>(f_grad_input);
        }
        
        // prevent race where buf is written again before reads are done
        __syncthreads();

    }
}


template <typename T, typename U, typename V>
void dispatch_cuda_rms_norm_backward(
    const V* dout,
    const U* invvar,
    at::Tensor* inter_out,
    int n1,
    int n2,
    const V* gamma,
    double epsilon,
    T* grad_input,
    T* grad_residual,
    V* grad_gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();
    if (gamma != nullptr) {
        const int part_size = warp_size;
        const int BLOCK_DIM = 4;
        const dim3 threads2(warp_size, BLOCK_DIM, 1);
        const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
        const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
        const int nshared2_b = threads2.x * threads2.y * sizeof(U);
        const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;

        const auto part_grad_dtype =
          (inter_out->scalar_type() == at::ScalarType::Half || inter_out->scalar_type() == at::ScalarType::BFloat16) ?
          at::ScalarType::Float :
          inter_out->scalar_type();
        at::Tensor part_grad_gamma = at::empty({part_size,n2}, inter_out->options().dtype(part_grad_dtype));
        auto kernel = &cuComputePartGradGammaBeta<T, U, V>;
        // std::cout << ":) ("
        //     << threads2.x << ", " 
        //     << threads2.y << ", " 
        //     << threads2.z << ") ("
        //     << blocks2.x << ", "
        //     << blocks2.y << ", "
        //     << blocks2.z << ") "
        //     << sizeof(T) << " " << sizeof(U) << " " << sizeof(V)
        //     << " " << nshared2
        //     << "\n   0x" << std::hex << (std::uint64_t)dout
        //     << "\n   0x" << std::hex << (std::uint64_t)invvar
        //     << "\n   0x" << std::hex << (std::uint64_t)inter_out->data_ptr<T>()
        //     << "\n   0x" << std::hex << (std::uint64_t)gamma
        //     << "\n   0x" << std::hex << (std::uint64_t)grad_input
        //     << "\n   0x" << std::hex << (std::uint64_t)grad_residual
        //     << "\n   0x" << std::hex << (std::uint64_t)grad_gamma
        //     << std::dec
        //     << std::endl;
        kernel<<<blocks2, threads2, nshared2, stream>>>(
            dout,
            inter_out->data_ptr<T>(),
            n1,
            n2,
            invvar,
            U(epsilon),
            gamma,
            part_grad_gamma.data_ptr<U>(),
            epsilon);
        // cudaStreamSynchronize(stream);

        const int BLOCK_DIM_GAMMA = 8;
        const dim3 threads3(warp_size, BLOCK_DIM_GAMMA, 1);
        const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
        const int nshared3 = threads3.x * threads3.y * sizeof(U);
        auto kernel2 = &cuComputeGradGammaBeta<U, V>;
        kernel2<<<blocks3, threads3, nshared3, stream>>>(
            part_grad_gamma.data_ptr<U>(),
            part_size,
            n1,
            n2,
            grad_gamma);
    }

    // compute grad_input
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
    const int  BLOCK_DIM_INPUT = std::min(std::round(n2 / 512.0), 4);
    const dim3 threads1(warp_size,BLOCK_DIM_INPUT,1);
    int nshared =
            threads1.y > 1 ?
            threads1.y*threads1.x*sizeof(U) :
            0;

    if (grad_residual) {
        auto kernel = &cuComputeGradInput<T, U, V, true>;
        kernel<<<blocks1, threads1, nshared, stream>>>(
            dout,
            inter_out->data_ptr<T>(),
            n1,
            n2,
            invvar,
            U(epsilon),
            gamma,
            grad_input,
            grad_residual);
    } else {
        auto kernel = &cuComputeGradInput<T, U, V, false>;
        kernel<<<blocks1, threads1, nshared, stream>>>(
            dout,
            inter_out->data_ptr<T>(),
            n1,
            n2,
            invvar,
            U(epsilon),
            gamma,
            grad_input,
            grad_residual);
    }
}


void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* inter_out,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_residual,
    at::Tensor* grad_gamma)
{
    using namespace at;

    DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        inter_out->scalar_type(), gamma == NULL ? inter_out->scalar_type() : gamma->scalar_type(), "cuda_rms_norm_backward",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        dispatch_cuda_rms_norm_backward<scalar_t_in, accscalar_t, scalar_t_out>(
            dout->data_ptr<scalar_t_out>(),
            invvar->data_ptr<accscalar_t>(),
            inter_out,
            n1,
            n2,
            gamma != NULL ? gamma->data_ptr<scalar_t_out>() : NULL,
            epsilon,
            grad_input->data_ptr<scalar_t_in>(),
            grad_residual != NULL ? grad_residual->data_ptr<scalar_t_in>() : NULL,
            gamma != NULL ? grad_gamma->data_ptr<scalar_t_out>() : NULL);
    )
}
