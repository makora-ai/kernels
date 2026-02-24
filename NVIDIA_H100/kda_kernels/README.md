# Kimi Delta Attention (KDA) — Triton Kernels (LLM-Generated)

This repository contains **MakoraGenerate-generated Triton implementations of Kimi Delta Attention (KDA)**.  

The blog post with more detail can be found here: https://makora.com/blog/generating-kda-kernels

KDA is a simplified linear-time attention variant introduced in the *Kimi* model family.  
These Triton kernels implement the core Δ-attention update in a fused and memory-efficient way.

---

## 📈 Benchmark Results

| Query × KeyLen Shape | Speedup vs `torch.compile` | Notes |
|----------------------|-----------------------------|-------|
| **256 × 8192**       | **5.5×**                    | Matches hand-optimized |
| **512 × 16384**      | **7.8×**                    | Matches hand-optimized |
| **8192 × 1024**      | **0.34×**                   | Hand-optimized kernel fails |
| **16384 × 1024**     | **0.7×**                    | Hand-optimized kernel fails |

All benchmarks run on the same backend using identical inputs and precision settings.
