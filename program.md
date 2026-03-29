# autoresearch-jax-tpu

This is the JAX/TPU port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted to run autonomously on a free Google Colab TPU v2. The agent modifies `train_tpu.py`, runs 5-minute experiments via a Colab notebook connected through `colab-mcp`, records results, and loops forever.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar28`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context and key differences from original.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. **Do not modify.**
   - `train_tpu.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify Colab is connected**: Use `mcp__colab-mcp__open_colab_browser_connection` to connect to the user's active Colab notebook with TPU runtime. Check that `/root/.cache/autoresearch/` exists with data shards and tokenizer. If not, run `prepare.py` first.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**.

## Colab execution model

**Critical**: Unlike the original autoresearch which runs `uv run train.py` as a subprocess, this JAX port must run training **inline in the Colab kernel** to avoid TPU conflicts. Use this pattern for every experiment:

```python
# In a Colab cell:
import threading, sys, os, importlib

os.chdir('/content/autoresearch-jax-tpu')
if '/content/autoresearch-jax-tpu' not in sys.path:
    sys.path.insert(0, '/content/autoresearch-jax-tpu')

# Pull latest code
import subprocess
subprocess.run(['git', '-C', '/content/autoresearch-jax-tpu', 'pull'], check=True)

import train_tpu
importlib.reload(train_tpu)

LOG = '/content/run.log'
threading.Thread(target=train_tpu.main, kwargs={'log_file': LOG}, daemon=True).start()
print(f"Training started. Log: {LOG}")
```

The cell returns immediately (non-blocking). Monitor progress by running:
```python
import subprocess
print(subprocess.run(['tail', '-20', '/content/run.log'], capture_output=True, text=True).stdout)
```

**DONE/FAIL detection**: Poll the log file. The training script writes `DONE` or `FAIL: ...` as the last line when finished.

**Timeout**: If no `DONE`/`FAIL` appears within 10 minutes of starting, kill by restarting the Colab runtime and treat as crash.

## Experimentation

**What you CAN do:**
- Modify `train_tpu.py` — everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It contains fixed evaluation, data loading, tokenizer, and training constants.
- Add new pip dependencies. You can only use what's already available: `jax`, `flax`, `optax` (>=0.2.8), `numpy`, `torch` (CPU only, for tokenizer loading).
- Modify the evaluation harness. `evaluate_bpb_jax` in `train_tpu.py` is the ground truth metric.

**The goal**: get the lowest `val_bpb`. Since the time budget is fixed at 300s, you don't need to worry about training time. Everything is fair game: architecture, optimizer, hyperparameters, batch size, model size. The only constraints are:
- Code runs without crashing
- Finishes within 300s training time (startup/eval overhead on top is fine)
- Peak memory stays within TPU v2 limits (~15GB HBM)

**TPU v2 constraints** (hard limits, unlike H100):
- `DEVICE_BATCH_SIZE * SEQ_LEN` must fit in ~15GB. Current default (8 × 512) uses ~4GB.
- No Flash Attention 3 — use `jax.nn.dot_product_attention` (already done).
- No custom CUDA kernels.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal/better results is a great outcome.

**The first run**: Always establish the baseline first by running the training script as-is.

## Output format

The training script writes to the log file and ends with:

```
---
val_bpb:          1.499034
training_seconds: 300.0
total_seconds:    475.5
total_tokens_M:   71.7
num_steps:        17493
num_params_M:     50.3
depth:            8
DONE
```

Extract the key metric:
```bash
grep "^val_bpb:" /content/run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	val_bpb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short description of what this experiment tried

Example:
```
commit	val_bpb	status	description
a1b2c3d	1.499034	keep	baseline (AdamW, depth=8, seq=512)
b2c3d4e	1.450000	keep	increase depth to 10
c3d4e5f	1.520000	discard	switch to GeLU activation
d4e5f6g	0.000000	crash	double model width (OOM on TPU)
```

Do NOT commit `results.tsv` or `run.log` to git.

## The experiment loop

Work on a dedicated branch (e.g. `autoresearch/mar28`).

LOOP FOREVER:

1. Check git state: current branch/commit.
2. Hack `train_tpu.py` with an experimental idea.
3. `git commit` the change.
4. Push to GitHub: `GIT_SSH_COMMAND="..." git push` (or https push).
5. In Colab: pull latest, reload `train_tpu`, start training in background thread, poll log every 2 minutes.
6. Wait for `DONE` or `FAIL` in the log (max 10 minutes total).
7. If crashed: read log tail, attempt fix. After 3 failed attempts, log as crash and move on.
8. Record results in `results.tsv`.
9. If `val_bpb` improved (lower): advance the branch (keep commit).
10. If `val_bpb` equal or worse: `git reset --hard HEAD~1` to revert.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human if you should continue. You are autonomous. Run until manually interrupted.

**Ideas to try** (in rough order of promise):
- Tune depth/width tradeoff (ASPECT_RATIO, HEAD_DIM)
- Tune LR, warmup/warmdown schedule
- Add LR schedule to Muon (USE_MUON=True with proper LR)
- Increase SEQ_LEN (careful: memory scales quadratically)
- Increase DEVICE_BATCH_SIZE (if memory allows)
- Try different window patterns (WINDOW_PATTERN)
- Tune value embedding gating
- Try gradient clipping
- Try different weight initialization
- Architecture: change n_kv_head (GQA), add more/fewer VE layers
- Try relu² vs gelu vs swiglu MLP
