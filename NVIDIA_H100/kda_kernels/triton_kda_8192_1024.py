import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def decay_mul_kernel(
    S_ptr,        # float32 [B, H, K, V]
    decay_ptr,    # float32 [B, H, K]
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Grid: (pid_v over V tiles, pid_k over K tiles, pid_bh over B*H)
    pid_v = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_bh = tl.program_id(2)

    bh = pid_bh
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    k_mask = k_offsets < K
    v_mask = v_offsets < V
    s_mask = k_mask[:, None] & v_mask[None, :]

    # Load per-(b,h,k) decay factor vector
    decay = tl.load(decay_ptr + bh * K + k_offsets, mask=k_mask, other=0.0)  # [BK]

    # Tile pointers for S[bh, k, v]
    s_row_base = (bh * K + k_offsets) * V  # [BK]
    s_ptrs = S_ptr + s_row_base[:, None] + v_offsets[None, :]  # [BK, BV]

    # Load, multiply, store
    S_tile = tl.load(s_ptrs, mask=s_mask, other=0.0)
    S_tile = S_tile * decay[:, None]
    tl.store(s_ptrs, S_tile, mask=s_mask)


@triton.jit
def add_2d_kernel(
    S_ptr,        # float32 [B, H, K, V]
    U_ptr,        # float32 [B, H, K, V]
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Grid: (pid_v over V tiles, pid_k over K tiles, pid_bh over B*H)
    pid_v = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_bh = tl.program_id(2)

    bh = pid_bh
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    k_mask = k_offsets < K
    v_mask = v_offsets < V
    mask = k_mask[:, None] & v_mask[None, :]

    row_base = (bh * K + k_offsets) * V
    s_ptrs = S_ptr + row_base[:, None] + v_offsets[None, :]
    u_ptrs = U_ptr + row_base[:, None] + v_offsets[None, :]

    S_tile = tl.load(s_ptrs, mask=mask, other=0.0)
    U_tile = tl.load(u_ptrs, mask=mask, other=0.0)
    S_tile = S_tile + U_tile
    tl.store(s_ptrs, S_tile, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale

    @staticmethod
    def _kda_with_triton_elementwise(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Hybrid implementation that is numerically identical to the naive PyTorch reference:
          - All reductions/einsums are computed with PyTorch to preserve exact numeric order.
          - Triton is used to accelerate large elementwise operations (decay multiply and state add),
            which are mathematically identical per-element and hence bitwise-match PyTorch.
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda, "All tensors must be CUDA tensors."

        dtype = v.dtype
        B, T, H, K = q.shape
        V = v.shape[-1]

        if scale is None:
            scale = K ** -0.5

        device = q.device

        # State and outputs in float32 for stability
        S = torch.zeros((B, H, K, V), dtype=torch.float32, device=device)
        o = torch.empty((B, T, H, V), dtype=torch.float32, device=device)

        # Triton launch configuration
        BLOCK_K = 128
        BLOCK_V = 128
        grid = lambda META: (
            triton.cdiv(V, META["BLOCK_V"]),
            triton.cdiv(K, META["BLOCK_K"]),
            B * H,
        )

        for t in range(T):
            # Casts to float32 to match reference
            q_t = q[:, t].to(torch.float32) * float(scale)  # [B,H,K]
            k_t = k[:, t].to(torch.float32)                 # [B,H,K]
            v_t = v[:, t].to(torch.float32)                 # [B,H,V]
            g_t = g[:, t].to(torch.float32)                 # [B,H,K]
            beta_t = beta[:, t].to(torch.float32)           # [B,H]

            # 1) Decay: S <- exp(g_t) * S
            decay = g_t.exp().contiguous()  # [B,H,K]
            decay_mul_kernel[grid](
                S, decay,
                B, H, K, V,
                BLOCK_K=BLOCK_K,
                BLOCK_V=BLOCK_V,
            )

            # 2) Predict: v_hat = (k_t[..., None] * S).sum(-2)
            v_hat = (k_t[..., None] * S).sum(dim=-2)  # [B,H,V]

            # 3) Compute update tensor U exactly as in reference using PyTorch einsum
            dv = (v_t - v_hat)  # [B,H,V]
            bk = (beta_t[..., None] * k_t)  # [B,H,K]
            U = torch.einsum('bhk,bhv->bhkv', bk, dv).contiguous()  # [B,H,K,V]

            # 4) In-place add S += U via Triton (elementwise add)
            add_2d_kernel[grid](
                S, U,
                B, H, K, V,
                BLOCK_K=BLOCK_K,
                BLOCK_V=BLOCK_V,
            )
            del U  # free early

            # 5) Read: o_t = q_t^T S (use einsum like reference)
            o[:, t] = torch.einsum('bhk,bhkv->bhv', q_t, S)

        return o.to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
        scale = self.scale if self.scale is not None else (q.shape[-1] ** -0.5)
        return self._kda_with_triton_elementwise(q, k, v, g, beta, scale=scale)