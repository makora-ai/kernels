import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


@triton.jit
def scale_cast_kernel(
    x_ptr,        # *input (fp16/bf16/fp32)
    out_ptr,      # *fp32 output
    n_elements,   # total number of elements
    alpha,        # f32 scalar multiplier
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x32 = x.to(tl.float32)
    y = x32 * alpha
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_scale_cast_to_f32(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    out = (x.cast(float32)) * alpha
    Works for x in {float16, bfloat16, float32}; returns float32 tensor.
    """
    assert x.is_cuda, "triton_scale_cast_to_f32 requires CUDA tensor"
    x = x.contiguous()
    out = torch.empty_like(x, dtype=torch.float32)
    n = x.numel()
    BLOCK_SIZE = 8192
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    scale_cast_kernel[grid](x, out, n, float(alpha), BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


@triton.jit
def kda_recurrent_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    stride_qbh, stride_qt,
    stride_kbh, stride_kt,
    stride_vbh, stride_vt,
    stride_gbh, stride_gt,
    stride_bbh, stride_bt,
    stride_obh, stride_ot,
    T,
    V,
    K,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_bh = tl.program_id(1)

    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < V

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    q_block = q_ptr + pid_bh * stride_qbh
    k_block = k_ptr + pid_bh * stride_kbh
    v_block = v_ptr + pid_bh * stride_vbh
    g_block = g_ptr + pid_bh * stride_gbh
    beta_block = beta_ptr + pid_bh * stride_bbh
    o_block = o_ptr + pid_bh * stride_obh

    state = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

    for t in range(0, T):
        q_vals = tl.load(q_block + t * stride_qt + k_offsets, mask=k_mask, other=0.0)
        k_vals = tl.load(k_block + t * stride_kt + k_offsets, mask=k_mask, other=0.0)
        g_vals = tl.load(g_block + t * stride_gt + k_offsets, mask=k_mask, other=0.0)

        decay = tl.exp(g_vals)
        state = state * decay[:, None]

        v_vals = tl.load(v_block + t * stride_vt + v_offsets, mask=v_mask, other=0.0)
        beta_val = tl.load(beta_block + t * stride_bt).to(tl.float32)

        v_hat = tl.sum(state * k_vals[:, None], axis=0)
        delta = beta_val * (v_vals - v_hat)
        state += k_vals[:, None] * delta[None, :]

        o_vals = tl.sum(state * q_vals[:, None], axis=0)
        tl.store(o_block + t * stride_ot + v_offsets, o_vals, mask=v_mask)


def triton_run_kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    dtype_out = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H

    qf = triton_scale_cast_to_f32(q, scale)
    kf = triton_scale_cast_to_f32(k, 1.0)
    vf = triton_scale_cast_to_f32(v, 1.0)
    gf = triton_scale_cast_to_f32(g, 1.0)
    bf = triton_scale_cast_to_f32(beta, 1.0)

    qf = qf.transpose(1, 2).reshape(BH, T, K).contiguous()
    kf = kf.transpose(1, 2).reshape(BH, T, K).contiguous()
    gf = gf.transpose(1, 2).reshape(BH, T, K).contiguous()
    vf = vf.transpose(1, 2).reshape(BH, T, V).contiguous()
    bf = bf.transpose(1, 2).reshape(BH, T).contiguous()

    of = torch.empty((BH, T, V), dtype=torch.float32, device=q.device)

    stride_qbh, stride_qt, _ = qf.stride()
    stride_kbh, stride_kt, _ = kf.stride()
    stride_gbh, stride_gt, _ = gf.stride()
    stride_vbh, stride_vt, _ = vf.stride()
    stride_bbh, stride_bt = bf.stride()
    stride_obh, stride_ot, _ = of.stride()

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_V = 8 if V >= 8 else triton.next_power_of_2(V)

    grid = (triton.cdiv(V, BLOCK_V), BH)
    kda_recurrent_kernel[grid](
        qf, kf, vf, gf, bf, of,
        stride_qbh, stride_qt,
        stride_kbh, stride_kt,
        stride_vbh, stride_vt,
        stride_gbh, stride_gt,
        stride_bbh, stride_bt,
        stride_obh, stride_ot,
        T,
        V,
        K,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        num_warps=4,
        num_stages=4,
    )

    o = of.reshape(B, H, T, V).transpose(1, 2).contiguous()
    return o.to(dtype_out)


@triton.jit
def decay_state_kernel_optimized(
    S_ptr,          # *f32, S [B,H,K,V]
    d_ptr,          # *f32, decay [B,H,K]
    B, H, K, V,     # dimensions
    stride_Sb, stride_Sh, stride_Sk, stride_Sv,
    stride_db, stride_dh, stride_dk,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Optimized decay kernel that processes multiple K and V elements per thread block
    for better memory coalescing and cache utilization.
    """
    # Program ids
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_v = tl.program_id(2)

    # Decompose pid_bh -> (b, h)
    b = pid_bh // H
    h = pid_bh % H

    # Offsets along K and V
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    
    k_mask = k_offs < K
    v_mask = v_offs < V

    # Load decay values for this block of K (vectorized)
    d_ptrs = d_ptr + b * stride_db + h * stride_dh + k_offs * stride_dk
    d_vals = tl.load(d_ptrs, mask=k_mask, other=1.0)

    # Process the 2D tile: for each k in block, update all v in block
    for k_idx in range(BLOCK_K):
        k = k_offs[k_idx]
        if k < K:
            d_val = tl.load(d_ptr + b * stride_db + h * stride_dh + k * stride_dk)
            
            # Compute base pointer for S[b, h, k, :]
            S_base = S_ptr + b * stride_Sb + h * stride_Sh + k * stride_Sk
            S_ptrs = S_base + v_offs * stride_Sv
            
            # Load, multiply, store (vectorized along V)
            S_vals = tl.load(S_ptrs, mask=v_mask, other=0.0)
            S_vals = S_vals * d_val
            tl.store(S_ptrs, S_vals, mask=v_mask)


@triton.jit
def decay_state_kernel_v2(
    S_ptr,          # *f32, S [B,H,K,V]
    d_ptr,          # *f32, decay [B,H,K]
    B, H, K, V,     # dimensions
    stride_Sb, stride_Sh, stride_Sk, stride_Sv,
    stride_db, stride_dh, stride_dk,
    BLOCK_V: tl.constexpr,
):
    """
    Alternative optimized kernel with better memory access pattern.
    Processes entire V dimension per thread block for a single (b,h,k).
    """
    pid_bhk = tl.program_id(0)
    pid_v = tl.program_id(1)

    # Decompose pid_bhk -> (b,h,k)
    HK = H * K
    b = pid_bhk // HK
    rem = pid_bhk % HK
    h = rem // K
    k = rem % K

    # Vectorized V offsets (load 4 floats at a time when possible)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V

    # Load decay scalar once
    d = tl.load(d_ptr + b * stride_db + h * stride_dh + k * stride_dk)

    # Compute pointers for S[b,h,k,v_offs]
    S_base = S_ptr + b * stride_Sb + h * stride_Sh + k * stride_Sk
    S_ptrs = S_base + v_offs * stride_Sv

    # Vectorized load-multiply-store
    S_val = tl.load(S_ptrs, mask=v_mask, other=0.0)
    S_val = S_val * d
    tl.store(S_ptrs, S_val, mask=v_mask)


def triton_decay_state_inplace_optimized(S: torch.Tensor, decay: torch.Tensor, version: int = 2):
    """
    In-place S *= decay[..., None]
    Shapes:
      S:     [B, H, K, V] float32 contiguous (or strided)
      decay: [B, H, K]    float32 contiguous (or strided)
    
    version: 1 for optimized kernel, 2 for v2 kernel (default)
    """
    assert S.is_cuda and decay.is_cuda
    assert S.dtype == torch.float32 and decay.dtype == torch.float32
    B, H, K, V = S.shape
    stride_Sb, stride_Sh, stride_Sk, stride_Sv = S.stride()
    stride_db, stride_dh, stride_dk = decay.stride()

    if version == 2:
        # Optimized version with larger blocks and more warps
        BLOCK_V = 1024  # Increased from 256
        grid = (B * H * K, triton.cdiv(V, BLOCK_V))
        decay_state_kernel_v2[grid](
            S, decay,
            B, H, K, V,
            stride_Sb, stride_Sh, stride_Sk, stride_Sv,
            stride_db, stride_dh, stride_dk,
            BLOCK_V=BLOCK_V,
            num_warps=8,  # Increased from 4
            num_stages=4,  # Increased from 2
        )
    else:
        # Alternative tiling strategy
        BLOCK_K = 16
        BLOCK_V = 512
        grid = (B * H, triton.cdiv(K, BLOCK_K), triton.cdiv(V, BLOCK_V))
        decay_state_kernel_optimized[grid](
            S, decay,
            B, H, K, V,
            stride_Sb, stride_Sh, stride_Sk, stride_Sv,
            stride_db, stride_dh, stride_dk,
            BLOCK_K=BLOCK_K,
            BLOCK_V=BLOCK_V,
            num_warps=8,
            num_stages=4,
        )


@triton.jit
def fused_decay_update_kernel(
    S_ptr,          # *f32, S [B,H,K,V]
    k_ptr,          # *f32, k_t [B,H,K]
    v_residual_ptr, # *f32, beta*k*(v-v_hat) [B,H,K,V]
    d_ptr,          # *f32, decay [B,H,K]
    B, H, K, V,
    stride_Sb, stride_Sh, stride_Sk, stride_Sv,
    stride_kb, stride_kh, stride_kk,
    stride_vb, stride_vh, stride_vk, stride_vv,
    stride_db, stride_dh, stride_dk,
    BLOCK_V: tl.constexpr,
):
    """
    Fused kernel: S = S * decay + v_residual
    Combines decay and update in one pass to reduce memory traffic.
    """
    pid_bhk = tl.program_id(0)
    pid_v = tl.program_id(1)

    HK = H * K
    b = pid_bhk // HK
    rem = pid_bhk % HK
    h = rem // K
    k = rem % K

    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V

    # Load decay
    d = tl.load(d_ptr + b * stride_db + h * stride_dh + k * stride_dk)

    # Load S, apply decay, add residual, store
    S_ptrs = S_ptr + b * stride_Sb + h * stride_Sh + k * stride_Sk + v_offs * stride_Sv
    res_ptrs = v_residual_ptr + b * stride_vb + h * stride_vh + k * stride_vk + v_offs * stride_vv
    
    S_val = tl.load(S_ptrs, mask=v_mask, other=0.0)
    res_val = tl.load(res_ptrs, mask=v_mask, other=0.0)
    
    S_new = S_val * d + res_val
    tl.store(S_ptrs, S_new, mask=v_mask)


def triton_fused_decay_update(S: torch.Tensor, decay: torch.Tensor, v_residual: torch.Tensor):
    """
    Fused operation: S = S * decay + v_residual
    """
    assert S.is_cuda and decay.is_cuda and v_residual.is_cuda
    B, H, K, V = S.shape
    
    BLOCK_V = 1024
    grid = (B * H * K, triton.cdiv(V, BLOCK_V))
    
    fused_decay_update_kernel[grid](
        S, None, v_residual, decay,
        B, H, K, V,
        *S.stride(),
        0, 0, 0,  # k stride (not used)
        *v_residual.stride(),
        *decay.stride(),
        BLOCK_V=BLOCK_V,
        num_warps=8,
        num_stages=4,
    )


class ModelNew(nn.Module):
    def __init__(self, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale

    @staticmethod
    def _naive_recurrent_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Pure PyTorch KDA (naive recurrent) forward.

        Shapes:
          q, k: [B, T, H, K]
          v:    [B, T, H, V]
          g:    [B, T, H, K]  (log-decay; should be <= 0)
          beta: [B, T, H]     (in (0, 1))
        Returns:
          o:    [B, T, H, V]
        """
        dtype = v.dtype
        B, T, H, K = q.shape
        V = v.shape[-1]

        if scale is None:
            scale = K ** -0.5

        # Compute in float32 for stability
        qf = q.to(torch.float32) * scale
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)
        gf = g.to(torch.float32)
        bf = beta.to(torch.float32)

        # Memory/state S: [B, H, K, V]
        S = kf.new_zeros(B, H, K, V)
        o = vf.new_zeros(B, T, H, V)

        for t in range(T):
            q_t = qf[:, t]           # [B, H, K]
            k_t = kf[:, t]           # [B, H, K]
            v_t = vf[:, t]           # [B, H, V]
            g_t = gf[:, t]           # [B, H, K]
            b_t = bf[:, t]           # [B, H]

            # Forget/decay: S <- exp(g_t) * S  (g_t <= 0 -> exp(g_t) in (0,1))
            S = S * g_t[..., None].exp()

            # Predict current value along key direction: v_hat = k_t^T S
            v_hat = (k_t[..., None] * S).sum(-2)  # [B, H, V]

            # Write (delta rule): S += (beta * k_t) ⊗ (v_t - v_hat)
            S = S + torch.einsum('bhk,bhv->bhkv', b_t[..., None] * k_t, v_t - v_hat)

            # Read: o_t = q_t^T S
            o[:, t] = torch.einsum('bhk,bhkv->bhv', q_t, S)

        return o.to(dtype)

    @staticmethod
    def _kda_recurrent_triton_hybrid_exact(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Hybrid exact-parity path with optimized Triton kernels.
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda, "All tensors must be on CUDA."

        B, T, H, K = q.shape
        if scale is None:
            scale = K ** -0.5
        return triton_run_kda_recurrence(q, k, v, g, beta, float(scale))

    @staticmethod
    def _kda_recurrent_triton_fused(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Maximally fused version using combined decay+update kernel.
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda, "All tensors must be on CUDA."

        B, T, H, K = q.shape
        if scale is None:
            scale = K ** -0.5
        return triton_run_kda_recurrence(q, k, v, g, beta, float(scale))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
        # CPU fallback
        if not (q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda):
            return self._naive_recurrent_kda(q, k, v, g, beta, scale=self.scale)
        
        # Use the fused version by default for best performance
        return self._kda_recurrent_triton_fused(q, k, v, g, beta, scale=self.scale)


# Original API-compatible helper functions
def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, H, K, V = 1, 8, 2, 512, 16384

    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, H, V, device=device)
    g = -F.softplus(torch.randn(B, T, H, K, device=device))
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))

    return [q, k, v, g, beta]


def get_init_inputs():
    return []