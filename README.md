# autoresearch-jax-tpu

JAX/Flax port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for Google Colab TPU.

## Setup

1. Open a Colab notebook with TPU runtime
2. Install dependencies:
```bash
pip install jax[tpu] flax optax rustbpe tiktoken pyarrow requests
```
3. Clone this repo and run data prep:
```bash
git clone https://github.com/dingkwang/autoresearch-jax-tpu
cd autoresearch-jax-tpu
python prepare.py --num-shards 2
```
4. Train:
```bash
python train_tpu.py
```

## Key differences from original

| Component | Original (PyTorch/CUDA) | This port (JAX/TPU) |
|-----------|------------------------|---------------------|
| Framework | PyTorch + torch.compile | JAX + jax.jit |
| Attention | Flash Attention 3 | jax.nn.dot_product_attention |
| Optimizer | Muon + AdamW | AdamW (optax) |
| Precision | bfloat16 autocast | bfloat16 native |
| Hardware | H100 GPU (80GB) | Colab TPU v2 (16.9GB) |
| Seq length | 2048 | 512 |
| Batch size | 128 | 8 |

## Results (120s training budget, TPU v2)

```
val_bpb:          1.515
training_seconds: 120
num_steps:        9768
num_params_M:     33.0
throughput:       ~370K tok/s
```
