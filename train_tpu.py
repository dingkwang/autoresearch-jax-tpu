"""
Autoresearch TPU training script. Single-TPU, single-file, JAX/Flax.
Ported from karpathy/autoresearch (PyTorch/CUDA) to run on free Colab TPU.
Usage: python train_tpu.py
"""

import os
import gc
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, EVAL_TOKENS, TOKENIZER_DIR

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(jnp.bfloat16)[None, :, None, :]
    sin = jnp.sin(freqs).astype(jnp.bfloat16)[None, :, None, :]
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=3)

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 512
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 384
    window_pattern: str = "SSSL"


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(nn.Module):
    n_head: int
    n_kv_head: int
    n_embd: int
    has_ve: bool
    ve_gate_channels: int = 32

    @nn.compact
    def __call__(self, x, ve, cos, sin):
        B, T, C = x.shape
        head_dim = self.n_embd // self.n_head

        q = nn.Dense(self.n_head * head_dim, use_bias=False, name='c_q')(x)
        k = nn.Dense(self.n_kv_head * head_dim, use_bias=False, name='c_k')(x)
        v = nn.Dense(self.n_kv_head * head_dim, use_bias=False, name='c_v')(x)

        q = q.reshape(B, T, self.n_head, head_dim)
        k = k.reshape(B, T, self.n_kv_head, head_dim)
        v = v.reshape(B, T, self.n_kv_head, head_dim)

        # Value residual (ResFormer)
        if self.has_ve and ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, head_dim)
            gate = 2 * jax.nn.sigmoid(
                nn.Dense(self.n_kv_head, use_bias=False, name='ve_gate')(x[..., :self.ve_gate_channels])
            )
            v = v + gate[..., None] * ve

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        y = y.reshape(B, T, -1)
        y = nn.Dense(self.n_embd, use_bias=False, name='c_proj')(y)
        return y


class MLP(nn.Module):
    n_embd: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.n_embd, use_bias=False, name='c_fc')(x)
        x = jax.nn.relu(x) ** 2  # relu-squared
        x = nn.Dense(self.n_embd, use_bias=False, name='c_proj')(x)
        return x


class Block(nn.Module):
    n_head: int
    n_kv_head: int
    n_embd: int
    has_ve: bool

    @nn.compact
    def __call__(self, x, ve, cos, sin):
        x = x + CausalSelfAttention(
            n_head=self.n_head, n_kv_head=self.n_kv_head,
            n_embd=self.n_embd, has_ve=self.has_ve, name='attn'
        )(rms_norm(x), ve, cos, sin)
        x = x + MLP(n_embd=self.n_embd, name='mlp')(rms_norm(x))
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, targets=None):
        cfg = self.config
        B, T = idx.shape
        head_dim = cfg.n_embd // cfg.n_head

        cos, sin = precompute_rotary_embeddings(T, head_dim)

        x = nn.Embed(cfg.vocab_size, cfg.n_embd, name='wte')(idx)
        x = rms_norm(x)
        x0 = x

        resid_lambdas = self.param('resid_lambdas', lambda rng, shape: jnp.ones(shape), (cfg.n_layer,))
        x0_lambdas = self.param('x0_lambdas', lambda rng, shape: jnp.full(shape, 0.1), (cfg.n_layer,))

        for i in range(cfg.n_layer):
            x = resid_lambdas[i] * x + x0_lambdas[i] * x0

            ve = None
            if has_ve(i, cfg.n_layer):
                kv_dim = cfg.n_kv_head * head_dim
                ve = nn.Embed(cfg.vocab_size, kv_dim, name=f've_{i}')(idx)

            x = Block(
                n_head=cfg.n_head, n_kv_head=cfg.n_kv_head,
                n_embd=cfg.n_embd, has_ve=has_ve(i, cfg.n_layer),
                name=f'block_{i}'
            )(x, ve, cos, sin)

        x = rms_norm(x)

        softcap = 15.0
        logits = nn.Dense(cfg.vocab_size, use_bias=False, name='lm_head')(x)
        logits = logits.astype(jnp.float32)
        logits = softcap * jnp.tanh(logits / softcap)

        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            mask = (targets != -1).astype(jnp.float32)
            loss = jnp.sum(loss * mask) / jnp.sum(mask)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Dataloader (JAX-compatible, no CUDA)
# ---------------------------------------------------------------------------

from prepare import _document_batches

def make_dataloader_jax(tokenizer, B, T, split, buffer_size=1000):
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = np.empty((B, row_capacity), dtype=np.int32)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx, best_len = i, doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        inputs = jnp.array(row_buffer[:, :-1])
        targets = jnp.array(row_buffer[:, 1:])
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def get_token_bytes_jax():
    import torch
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        tb = torch.load(f, map_location="cpu")
    return jnp.array(tb.numpy(), dtype=jnp.int32)


def evaluate_bpb_jax(model, params, tokenizer, batch_size, token_bytes, eval_fn):
    val_loader = make_dataloader_jax(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        nats, tb = eval_fn(params, x, y, token_bytes)
        total_nats += nats.item()
        total_bytes += tb.item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 64           # target head dimension for attention
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = 2**16
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
DEPTH = 8
DEVICE_BATCH_SIZE = 8
SEQ_LEN = 512

# Optimizer selection: True = Muon (2D weights) + AdamW (embeddings/1D)
#                      False = AdamW only (baseline)
USE_MUON = True
MUON_LR = 2e-2          # Muon converges well with higher LR
ADAMW_LR = 3e-3         # AdamW baseline LR

# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main(log_file=None):
    """Run training. If log_file is given, all output goes there (no stdout).
    This lets callers run training in a background thread without blocking Jupyter."""
    import sys

    if log_file:
        _f = open(log_file, 'w')
        def p(s):
            _f.write(s + '\n')
            _f.flush()
    else:
        def p(s):
            print(s, flush=True)

    try:
        t_start = time.time()

        tokenizer = Tokenizer.from_directory()
        vocab_size = tokenizer.get_vocab_size()
        p(f"Vocab size: {vocab_size:,}")

        base_dim = DEPTH * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        num_heads = model_dim // HEAD_DIM
        config = GPTConfig(
            sequence_len=SEQ_LEN, vocab_size=vocab_size,
            n_layer=DEPTH, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=WINDOW_PATTERN,
        )
        p(f"Model config: {asdict(config)}")

        model = GPT(config=config)
        key = jax.random.PRNGKey(42)
        dummy = jnp.ones((DEVICE_BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
        params = model.init(key, dummy, dummy)

        param_count = sum(x.size for x in jax.tree.leaves(params))
        p(f"Total parameters: {param_count:,}")

        if USE_MUON:
            # Muon for 2D kernel weights (linear layers), AdamW for embeddings/1D params.
            # Kernel params have path ending in 'kernel'; everything else goes to AdamW.
            def param_labels(params):
                def label_leaf(path, _):
                    return 'muon' if len(path) > 0 and 'kernel' in str(path[-1]) else 'adamw'
                return jax.tree_util.tree_map_with_path(label_leaf, params)

            optimizer = optax.multi_transform(
                {
                    'muon': optax.contrib.muon(learning_rate=MUON_LR, momentum=0.95, nesterov=True),
                    'adamw': optax.adamw(learning_rate=ADAMW_LR, weight_decay=WEIGHT_DECAY),
                },
                param_labels(params),
            )
            p(f"Optimizer: Muon (lr={MUON_LR}) + AdamW (lr={ADAMW_LR})")
        else:
            optimizer = optax.adamw(learning_rate=ADAMW_LR, weight_decay=WEIGHT_DECAY)
            p(f"Optimizer: AdamW (lr={ADAMW_LR})")
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(lambda p: model.apply(p, x, y))(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_opt_state, loss

        token_bytes = get_token_bytes_jax()

        @jax.jit
        def eval_step(params, x, y, token_bytes):
            logits = model.apply(params, x).astype(jnp.float32)
            loss_flat = optax.softmax_cross_entropy_with_integer_labels(logits, y).reshape(-1)
            y_flat = y.reshape(-1)
            nbytes = token_bytes[y_flat]
            mask = (nbytes > 0).astype(jnp.float32)
            return jnp.sum(loss_flat * mask), jnp.sum(nbytes)

        train_loader = make_dataloader_jax(tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, "train")

        p("Compiling JIT...")
        x, y, _ = next(train_loader)
        params, opt_state, loss = train_step(params, opt_state, x, y)
        loss.block_until_ready()
        p("JIT done. Starting training...")

        def get_lr_multiplier(progress):
            if progress < WARMUP_RATIO:
                return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
            elif progress < 1.0 - WARMDOWN_RATIO:
                return 1.0
            else:
                cooldown = (1.0 - progress) / WARMDOWN_RATIO
                return cooldown + (1 - cooldown) * FINAL_LR_FRAC

        t_start_training = time.time()
        total_training_time = 0
        step = 1
        smooth_loss = 0

        while True:
            t0 = time.time()
            x, y, epoch = next(train_loader)
            params, opt_state, loss = train_step(params, opt_state, x, y)
            loss.block_until_ready()
            dt = time.time() - t0

            if step > 10:
                total_training_time += dt

            loss_f = loss.item()
            if math.isnan(loss_f) or loss_f > 100:
                p("FAIL: loss exploded")
                return

            ema_beta = 0.95
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
            debiased = smooth_loss / (1 - ema_beta ** step)

            if step % 50 == 0:
                progress = min(total_training_time / TIME_BUDGET, 1.0)
                tok_per_sec = int(DEVICE_BATCH_SIZE * SEQ_LEN / dt)
                remaining = max(0, TIME_BUDGET - total_training_time)
                p(f"step {step:05d} ({100*progress:.1f}%) | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s")

            if step == 1:
                gc.collect()
                gc.disable()

            step += 1
            if step > 10 and total_training_time >= TIME_BUDGET:
                break

        total_tokens = step * DEVICE_BATCH_SIZE * SEQ_LEN

        p("Evaluating val_bpb...")
        val_bpb = evaluate_bpb_jax(model, params, tokenizer, DEVICE_BATCH_SIZE, token_bytes, eval_step)

        t_end = time.time()
        p("---")
        p(f"val_bpb:          {val_bpb:.6f}")
        p(f"training_seconds: {total_training_time:.1f}")
        p(f"total_seconds:    {t_end - t_start:.1f}")
        p(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
        p(f"num_steps:        {step}")
        p(f"num_params_M:     {param_count / 1e6:.1f}")
        p(f"depth:            {DEPTH}")
        p("DONE")

    except Exception as e:
        import traceback
        p(f"FAIL: {e}")
        p(traceback.format_exc())
    finally:
        if log_file:
            _f.close()


if __name__ == "__main__":
    main()
