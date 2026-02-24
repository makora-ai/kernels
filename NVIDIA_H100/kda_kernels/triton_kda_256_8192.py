import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kda_recurrent_forward_kernel(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    # Strides for memory access
    q_stride_b, q_stride_t, q_stride_h, q_stride_k,
    k_stride_b, k_stride_t, k_stride_h, k_stride_k,
    v_stride_b, v_stride_t, v_stride_h, v_stride_v,
    g_stride_b, g_stride_t, g_stride_h, g_stride_k,
    beta_stride_b, beta_stride_t, beta_stride_h,
    o_stride_b, o_stride_t, o_stride_h, o_stride_v,
    # Dimensions
    B, T, H, K, V,
    # Scale factor for q
    scale: tl.constexpr,
    # Block sizes for tiling
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Optimized Triton kernel for the recurrent KDA forward pass.
    
    This kernel is parallelized across the batch, head, and V dimensions.
    Each program instance processes a tile of size [K, BLOCK_V] for a single batch/head pair.
    The time-series recurrence is handled by a sequential loop within each program instance,
    which is highly efficient as the recurrent state `S` is kept in fast SRAM throughout the loop.
    """
    # Get program IDs for the grid: (B*H, V / BLOCK_V)
    pid_bh = tl.program_id(0)  # Program ID for the flattened batch-head dimension
    pid_v = tl.program_id(1)   # Program ID for the V dimension tile

    # Decompose the flattened batch-head index to get b and h
    b = pid_bh // H
    h = pid_bh % H

    # --- Set up pointers for the current batch and head ---
    # This pre-calculation avoids recomputing base addresses inside the time loop
    q_ptr_bh = q_ptr + b * q_stride_b + h * q_stride_h
    k_ptr_bh = k_ptr + b * k_stride_b + h * k_stride_h
    v_ptr_bh = v_ptr + b * v_stride_b + h * v_stride_h
    g_ptr_bh = g_ptr + b * g_stride_b + h * g_stride_h
    beta_ptr_bh = beta_ptr + b * beta_stride_b + h * beta_stride_h
    o_ptr_bh = o_ptr + b * o_stride_b + h * o_stride_h

    # --- Set up offsets for K and V blocks ---
    k_offsets = tl.arange(0, BLOCK_K)
    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    # --- Initialize recurrent state S in SRAM ---
    # The state is declared as a Triton variable, ensuring it resides in fast on-chip memory (SRAM/registers).
    # Its size is [BLOCK_K, BLOCK_V], and it's kept in float32 for numerical stability.
    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    # --- Main loop over the time dimension ---
    # This loop is executed sequentially within each kernel instance.
    for t in range(T):
        # Create masks for boundary conditions, preventing out-of-bounds access
        k_mask = k_offsets < K
        v_mask = v_offsets < V
        
        # --- Load inputs for the current timestep t ---
        # Vector loads are used for q, k, g, and v, which is efficient.
        # Data is converted to float32 on-the-fly to ensure high-precision computation.
        q_t = tl.load(q_ptr_bh + t * q_stride_t + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        k_t = tl.load(k_ptr_bh + t * k_stride_t + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        g_t = tl.load(g_ptr_bh + t * g_stride_t + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        v_t = tl.load(v_ptr_bh + t * v_stride_t + v_offsets, mask=v_mask, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_ptr_bh + t * beta_stride_t).to(tl.float32)

        q_t *= scale
        
        # --- Perform the recurrent update logic on the SRAM state tile ---
        # All operations for a single timestep are fused within this loop body,
        # maximizing data locality and minimizing SRAM traffic.
        
        # 1. Forget/Decay: S <- exp(g_t) * S
        state *= tl.exp(g_t)[:, None]
        
        # 2. Predict value: v_hat = k_t^T S
        v_hat = tl.sum(k_t[:, None] * state, axis=0)
        
        # 3. Write (delta rule): S += (beta * k_t) ⊗ (v_t - v_hat)
        delta_v = v_t - v_hat
        state += (beta_t * k_t)[:, None] * delta_v[None, :]
        
        # 4. Read: o_t = q_t^T S
        o_t = tl.sum(q_t[:, None] * state, axis=0)
        
        # --- Store the output for timestep t ---
        o_t_ptr = o_ptr_bh + t * o_stride_t + v_offsets
        # The result is cast back to the original datatype upon storing to global memory.
        tl.store(o_t_ptr, o_t, mask=v_mask)


def triton_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    PyTorch wrapper for the optimized Triton-based recurrent KDA forward pass.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    assert all(t.is_cuda for t in [q, k, v, g, beta]), "All input tensors must be on a CUDA device."
    
    # Ensure input tensors are contiguous in memory for optimal coalesced access in the kernel.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    
    o = torch.empty_like(v)
    
    if scale is None:
        scale = K ** -0.5
        
    # --- Kernel Configuration and Launch ---
    # For this workload, covering the entire K dimension in one block is optimal as it fits in SRAM.
    BLOCK_K = triton.next_power_of_2(K)
    # A larger BLOCK_V increases parallelism and arithmetic intensity per thread block.
    BLOCK_V = 128

    grid = (B * H, triton.cdiv(V, BLOCK_V))

    kda_recurrent_forward_kernel[grid](
        q, k, v, g, beta, o,
        # Pass strides to handle any memory layout.
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Dimensions
        B, T, H, K, V,
        # Constants passed at compile time
        scale=scale,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        # Performance tuning hints for the Triton compiler
        num_warps=8,
        num_stages=3,
    )
    return o


class ModelNew(nn.Module):
    def __init__(self, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
        # Replace the naive PyTorch loop with the highly optimized Triton kernel.
        return triton_recurrent_kda(q, k, v, g, beta, scale=self.scale)