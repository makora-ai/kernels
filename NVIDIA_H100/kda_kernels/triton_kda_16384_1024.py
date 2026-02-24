import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def s_qk_rowwise_dot_kernel(
    q_ptr,  # [T, BH, K], float32 (scaled)
    k_ptr,  # [T, BH, K], float32
    out_ptr,  # [T, BH], float32
    stride_q_t, stride_q_bh, stride_q_k,
    stride_k_t, stride_k_bh, stride_k_k,
    stride_o_t, stride_o_bh,
    K,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_row = q_ptr + pid_t * stride_q_t + pid_bh * stride_q_bh
    k_row = k_ptr + pid_t * stride_k_t + pid_bh * stride_k_bh

    acc = tl.zeros([], dtype=tl.float32)
    k_range = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + k_range
        mask = k_offsets < K
        q_vec = tl.load(q_row + k_offsets * stride_q_k, mask=mask, other=0.0, eviction_policy="evict_last")
        k_vec = tl.load(k_row + k_offsets * stride_k_k, mask=mask, other=0.0, eviction_policy="evict_last")
        acc += tl.sum(q_vec * k_vec, axis=0)
    tl.store(out_ptr + pid_t * stride_o_t + pid_bh * stride_o_bh, acc)


@triton.jit
def kda_recurrent_fused_precomputed_sqk_kernel(
    S_ptr,        # In/Out: [BH, K, V], float32
    q_ptr,        # In: [T, BH, K], float32 (scaled)
    k_ptr,        # In: [T, BH, K], float32
    v_ptr,        # In: [T, BH, V], float32
    ge_ptr,       # In: [T, BH, K], float32 (precomputed exp(g))
    beta_ptr,     # In: [T, BH], float32
    s_qk_ptr,     # In: [T, BH], float32 (precomputed q^T @ k)
    o_ptr,        # Out: [T, BH, V], float32
    # Strides
    stride_s_bh, stride_s_k, stride_s_v,
    stride_q_t, stride_q_bh, stride_q_k,
    stride_k_t, stride_k_bh, stride_k_k,
    stride_v_t, stride_v_bh, stride_v_v,
    stride_ge_t, stride_ge_bh, stride_ge_k,
    stride_beta_t, stride_beta_bh,
    stride_s_qk_t, stride_s_qk_bh,
    stride_o_t, stride_o_bh, stride_o_v,
    # sizes
    T, K, V,
    # meta
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v_tile = tl.program_id(1)

    v_offsets = pid_v_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < V
    tl.multiple_of(v_offsets, BLOCK_V)

    S_base_ptr = S_ptr + pid_bh * stride_s_bh
    k_range = tl.arange(0, BLOCK_K)
    tl.multiple_of(k_range, BLOCK_K)

    for t in range(0, T):
        q_base_ptr = q_ptr + t * stride_q_t + pid_bh * stride_q_bh
        k_base_ptr = k_ptr + t * stride_k_t + pid_bh * stride_k_bh
        v_base_ptr = v_ptr + t * stride_v_t + pid_bh * stride_v_bh
        ge_base_ptr = ge_ptr + t * stride_ge_t + pid_bh * stride_ge_bh
        o_base_ptr = o_ptr + t * stride_o_t + pid_bh * stride_o_bh

        beta_t = tl.load(beta_ptr + t * stride_beta_t + pid_bh * stride_beta_bh)
        s_qk_t = tl.load(s_qk_ptr + t * stride_s_qk_t + pid_bh * stride_s_qk_bh)

        v_hat_acc = tl.zeros([BLOCK_V], dtype=tl.float32)
        o_decay_acc = tl.zeros([BLOCK_V], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + k_range
            k_mask = k_offsets < K

            q_vec = tl.load(q_base_ptr + k_offsets * stride_q_k, mask=k_mask, other=0.0, eviction_policy="evict_last")
            k_vec = tl.load(k_base_ptr + k_offsets * stride_k_k, mask=k_mask, other=0.0, eviction_policy="evict_last")
            ge_vec = tl.load(ge_base_ptr + k_offsets * stride_ge_k, mask=k_mask, other=0.0, eviction_policy="evict_last")

            s_ptrs = S_base_ptr + k_offsets[:, None] * stride_s_k + v_offsets[None, :] * stride_s_v
            s_mask = k_mask[:, None] & v_mask[None, :]
            s_tile_old = tl.load(s_ptrs, mask=s_mask, other=0.0)

            s_tile_decayed = s_tile_old * ge_vec[:, None]

            v_hat_acc += tl.sum(s_tile_decayed * k_vec[:, None], axis=0)
            o_decay_acc += tl.sum(s_tile_decayed * q_vec[:, None], axis=0)

        v_tile = tl.load(v_base_ptr + v_offsets * stride_v_v, mask=v_mask, other=0.0)
        delta_v = v_tile - v_hat_acc
        o_t_final = tl.fma(beta_t * s_qk_t, delta_v, o_decay_acc)
        tl.store(o_base_ptr + v_offsets * stride_o_v, o_t_final, mask=v_mask)

        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + k_range
            k_mask = k_offsets < K

            k_vec = tl.load(k_base_ptr + k_offsets * stride_k_k, mask=k_mask, other=0.0, eviction_policy="evict_last")
            ge_vec = tl.load(ge_base_ptr + k_offsets * stride_ge_k, mask=k_mask, other=0.0, eviction_policy="evict_last")
            s_ptrs = S_base_ptr + k_offsets[:, None] * stride_s_k + v_offsets[None, :] * stride_s_v
            s_mask = k_mask[:, None] & v_mask[None, :]
            s_tile_old = tl.load(s_ptrs, mask=s_mask, other=0.0)

            s_tile_new = s_tile_old * ge_vec[:, None] + (beta_t * k_vec)[:, None] * delta_v[None, :]
            tl.store(s_ptrs, s_tile_new, mask=s_mask)


class ModelNew(nn.Module):
    def __init__(self, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale

    @staticmethod
    def _torch_recurrent_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        dtype = v.dtype
        B, T, H, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K ** -0.5
        qf = (q.to(torch.float32) * scale)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)
        gf = g.to(torch.float32)
        bf = beta.to(torch.float32)

        S = kf.new_zeros(B, H, K, V)
        o = vf.new_zeros(B, T, H, V)
        for t in range(T):
            q_t = qf[:, t]
            k_t = kf[:, t]
            v_t = vf[:, t]
            g_t = gf[:, t]
            b_t = bf[:, t]
            S = S * g_t[..., None].exp()
            v_hat = (k_t[..., None] * S).sum(-2)
            S = S + torch.einsum('bhk,bhv->bhkv', b_t[..., None] * k_t, v_t - v_hat)
            o[:, t] = torch.einsum('bhk,bhkv->bhv', q_t, S)
        return o.to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
        # CPU fallback
        if not (q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda):
            return self._torch_recurrent_kda(q, k, v, g, beta, scale=self.scale)

        dtype = v.dtype
        B, T, H, K = q.shape
        V = v.shape[-1]
        BH = B * H
        scale = self.scale if self.scale is not None else (K ** -0.5)

        # Prepare compute tensors in float32
        qf = (q.to(torch.float32) * scale)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)
        ge = g.to(torch.float32).exp_()  # precompute exp(g)
        bf = beta.to(torch.float32)

        # Layout: [B, T, H, D] -> [T, BH, D]
        q_flat = qf.permute(1, 0, 2, 3).contiguous().view(T, BH, K)
        k_flat = kf.permute(1, 0, 2, 3).contiguous().view(T, BH, K)
        v_flat = vf.permute(1, 0, 2, 3).contiguous().view(T, BH, V)
        ge_flat = ge.permute(1, 0, 2, 3).contiguous().view(T, BH, K)
        beta_flat = bf.permute(1, 0, 2).contiguous().view(T, BH)

        # Compute s_qk with Triton (rowwise dot over K) to avoid PyTorch einsum overhead
        s_qk_flat = torch.empty((T, BH), device=q.device, dtype=torch.float32)
        BLOCK_K_DOT = 1024
        grid_dot = (T, BH)
        s_qk_rowwise_dot_kernel[grid_dot](
            q_flat, k_flat, s_qk_flat,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            s_qk_flat.stride(0), s_qk_flat.stride(1),
            K,
            BLOCK_K=BLOCK_K_DOT,
            num_warps=4,
            num_stages=2,
        )

        # Persistent state and output buffers
        S_flat = torch.zeros((BH, K, V), device=q.device, dtype=torch.float32)
        o_flat = torch.empty((T, BH, V), device=q.device, dtype=torch.float32)

        # Kernel launch config
        BLOCK_K = 256
        BLOCK_V = 64
        grid = (BH, triton.cdiv(V, BLOCK_V))

        kda_recurrent_fused_precomputed_sqk_kernel[grid](
            S_flat, q_flat, k_flat, v_flat, ge_flat, beta_flat, s_qk_flat, o_flat,
            S_flat.stride(0), S_flat.stride(1), S_flat.stride(2),
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            ge_flat.stride(0), ge_flat.stride(1), ge_flat.stride(2),
            beta_flat.stride(0), beta_flat.stride(1),
            s_qk_flat.stride(0), s_qk_flat.stride(1),
            o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
            T, K, V,
            BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
            num_warps=8,
            num_stages=4,
        )

        # Reshape back to [B, T, H, V]
        o = o_flat.view(T, B, H, V).permute(1, 0, 2, 3).contiguous()
        return o.to(dtype)