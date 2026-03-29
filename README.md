# autoresearch-jax-tpu

JAX/Flax port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for Google Colab TPU.

## Setup

1. Open a Colab notebook with TPU runtime (dependencies are pre-installed)
2. Clone this repo and run data prep:
```bash
git clone https://github.com/dingkwang/autoresearch-jax-tpu
cd autoresearch-jax-tpu
python prepare.py --num-shards 2
```
3. Train:
```bash
python train_tpu.py
```

## Key differences from original

| Component | Original (PyTorch/CUDA) | This port (JAX/TPU) |
|-----------|------------------------|---------------------|
| Framework | PyTorch + torch.compile | JAX + jax.jit |
| Attention | Flash Attention 3 | jax.nn.dot_product_attention |
| Optimizer | Muon + AdamW | Muon + AdamW (optax ≥0.2.8, toggle `USE_MUON`) |
| Precision | bfloat16 autocast | bfloat16 native |
| Hardware | H100 GPU (80GB) | Colab TPU v2 (16.9GB) |
| Seq length | 2048 | 512 |
| Batch size | 128 | 8 |

## Model Architecture

```
Input tokens (B, T)
        │
        ▼
┌───────────────────┐
│  Token Embedding  │  wte: vocab_size → n_embd (512)
└───────────────────┘
        │
        ▼
   RMSNorm ──────────────────────────────── x0 (skip anchor)
        │                                        │
        │     × resid_lambda[i]  +  x0_lambda[i] × ──┘  (per layer)
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Block × 8                                                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  CausalSelfAttention  (pre-norm)                         │   │
│  │                                                          │   │
│  │   RMSNorm(x)                                             │   │
│  │       │                                                  │   │
│  │   ┌───┼───────────┐                                      │   │
│  │   Q   K           V                                      │   │
│  │   │   │           │  ← Value Embedding (even layers*)    │   │
│  │  RoPE RoPE        │    ve = Embed(vocab → kv_dim)        │   │
│  │  RMS  RMS         │    v += sigmoid_gate × ve            │   │
│  │   └───┴───────────┘                                      │   │
│  │   dot_product_attention (causal mask)                    │   │
│  │   Linear proj → n_embd                                   │   │
│  │   x = x + attn_out                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MLP  (pre-norm)                                         │   │
│  │                                                          │   │
│  │   RMSNorm(x)                                             │   │
│  │   Linear(n_embd → 4×n_embd)                              │   │
│  │   ReLU²  (relu-squared)                                  │   │
│  │   Linear(4×n_embd → n_embd)                              │   │
│  │   x = x + mlp_out                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
   RMSNorm
        │
        ▼
┌───────────────────┐
│  lm_head          │  Linear(n_embd → vocab_size), no bias
└───────────────────┘
        │
        ▼
  Softcap: 15 × tanh(logits / 15)
        │
        ▼
  Cross-entropy loss
```

*Value Embedding is present on even-indexed layers (`layer_idx % 2 == (n_layer-1) % 2`),
i.e. layers 1, 3, 5, 7 for depth=8.*

### Design highlights vs vanilla nanoGPT

| Feature | Purpose |
|---------|---------|
| **Value Embedding** (ResFormer) | Injects raw token semantics into V; mitigates over-smoothing in deep layers |
| **resid_lambda + x0_lambda** | Learnable per-layer residual scale + x0 skip connection; stabilises deep training |
| **RoPE** | Rotary positional encoding; replaces absolute position embeddings |
| **QK RMSNorm** | Normalises Q and K inside attention; prevents attention collapse |
| **ReLU²** | Sparser than GeLU; TPU-friendly |
| **Softcap logits** | Caps logits at ±15 via tanh; prevents logit explosion without gradient clipping |

## Results (300s training budget, TPU v2)

```
val_bpb:          1.500
training_seconds: 300
num_steps:        17338
num_params_M:     50.3
throughput:       ~255K tok/s
```
