import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
        return self._naive_recurrent_kda(q, k, v, g, beta, scale=self.scale)


def get_inputs():
    # randomly generate input tensors based on the model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Keep T modest for speed; K=256, V=8192 as requested
    B, T, H, K, V = 1, 8, 2, 256, 8192

    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, H, V, device=device)

    # Log-decay g <= 0 ensures exp(g) in (0, 1)
    g = -F.softplus(torch.randn(B, T, H, K, device=device))
    # Write strength beta in (0, 1)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))

    return [q, k, v, g, beta]


def get_init_inputs():
    # No extra initialization tensors required
    return []

model = Model()
inputs = get_inputs()
outputs = model(*inputs)
print(outputs.shape)