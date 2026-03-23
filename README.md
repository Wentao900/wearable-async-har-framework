# wearable-async-har-framework

A research scaffold for **asynchronous multimodal fusion in wearable human activity recognition (HAR)**.

## What this repo is

This repository is an **early, honest framework** for exploring async wearable HAR:
- a runnable synthetic smoke path,
- starter dataset adapters for **PAMAP2** and **WISDM**,
- a small PyTorch baseline with timestamp-aware alignment,
- CPU and GPU example configs,
- lightweight scripts and notes for getting experiments off the ground.

It is **not** yet a finalized benchmark implementation, and it does **not** claim tuned or paper-ready results.

## What "GPU support" means here

This repo now includes a practical GPU starter path for **RTX 4090D + CUDA 11.8** environments:
- `requirements-gpu.txt` for a simple CUDA 11.8 Python install,
- one-command run scripts for PAMAP2 and WISDM,
- dual GPU config presets:
  - **safe** = conservative starting point,
  - **fast** = more aggressive throughput-oriented starting point.

These are **starter configs**, not tuned SOTA recipes. They are meant to get a 4090D machine running cleanly, not to pretend a benchmark has already been optimized.

## Repo layout

- `configs/` — YAML experiment configs for CPU and GPU runs
- `docs/` — setup notes, framework docs, experiment ideas, release notes
- `scripts/` — training and convenience scripts
- `src/` — data adapters, model code, training loop, config loader
- `tests/` — smoke tests

## Important honesty notes

### Synthetic path
The default runnable path uses a **tiny synthetic dataset**. That path exists to verify:
- config loading,
- dataset construction,
- mask/timestamp handling,
- model wiring,
- train/eval loop behavior.

It is a pipeline smoke test, **not** a meaningful HAR benchmark.

### PAMAP2 and WISDM
This repo includes starter loaders for PAMAP2 and WISDM, but:
- the datasets are **not bundled**,
- preprocessing assumptions are still **starter assumptions**,
- evaluation protocol details should be checked before using results in a paper.

### GPU presets
The GPU configs included here are deliberately modest and readable. They are meant for:
- getting onto CUDA quickly,
- reducing the chance of first-run friction,
- giving you a safe and a faster preset to iterate from.

They are **not** presented as optimal settings.

## Quick start (CPU smoke path)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/print_project_summary.py
python3 scripts/train.py --config configs/base.yaml
```

Expected result:
- a short CPU-friendly run,
- metrics printed to stdout,
- `outputs/default/results.json` written.

## Dataset layouts

### PAMAP2
Expected raw-data layout:

```bash
data/PAMAP2/Protocol/subject101.dat
data/PAMAP2/Protocol/subject102.dat
```

Optional quick inspection:

```bash
python3 scripts/check_pamap2.py --root data/PAMAP2 --split train
```

CPU starter run:

```bash
python3 scripts/train.py --config configs/pamap2.yaml
```

### WISDM
Expected raw-data layout:

```bash
data/WISDM/WISDM_ar_v1.1_raw.txt
```

CPU starter run:

```bash
python3 scripts/train.py --config configs/wisdm.yaml
```

If files are missing, `scripts/train.py` fails with a direct, honest message.

## GPU setup for RTX 4090D + CUDA 11.8

Use Python 3.10 or 3.11 if possible.

### Option A: simple pip install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

That keeps the install simple:
- base deps come from `requirements.txt`,
- PyTorch packages come from the CUDA 11.8 wheel index.

### Verify CUDA visibility

```bash
python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
PY
```

## One-command GPU runs

### PAMAP2

```bash
bash scripts/run_pamap2_gpu.sh safe
bash scripts/run_pamap2_gpu.sh fast
```

### WISDM

```bash
bash scripts/run_wisdm_gpu.sh safe
bash scripts/run_wisdm_gpu.sh fast
```

If you omit the argument, each script defaults to `safe`.

## GPU config presets

### PAMAP2
- `configs/pamap2-gpu-safe.yaml`
- `configs/pamap2-gpu-fast.yaml`

### WISDM
- `configs/wisdm-gpu-safe.yaml`
- `configs/wisdm-gpu-fast.yaml`

Backward-compatible default GPU configs also remain:
- `configs/pamap2-gpu.yaml`
- `configs/wisdm-gpu.yaml`

### How to think about them

**Safe** presets:
- moderate batch size,
- moderate hidden size,
- intended as the first thing to try.

**Fast** presets:
- larger batch size,
- slightly larger hidden size and/or windowing,
- meant for machines that have headroom.

If the fast preset runs out of memory or becomes unstable, step back to the safe preset first.

## Running directly without helper scripts

```bash
python3 scripts/train.py --config configs/pamap2-gpu-safe.yaml
python3 scripts/train.py --config configs/pamap2-gpu-fast.yaml
python3 scripts/train.py --config configs/wisdm-gpu-safe.yaml
python3 scripts/train.py --config configs/wisdm-gpu-fast.yaml
```

## Current model/baseline summary

The current baseline includes:
- a small dataset factory,
- starter adapters for synthetic / PAMAP2 / WISDM,
- nearest-neighbor timestamp alignment,
- per-modality encoders,
- a simple fusion block,
- a classifier head,
- YAML-driven experiment configs.

Supported alignment reference strategies:
- `densest`
- `first`
- explicit modality names such as `accelerometer`

This is intentionally a baseline implementation, not a claimed algorithmic contribution.

## Tests

```bash
python3 -m pytest tests/test_smoke.py
python3 -m pytest tests/test_alignment.py
```

## If GPU runs hit trouble

Try, in order:
1. use the `safe` preset,
2. reduce `training.batch_size`,
3. reduce `model.hidden_dim`,
4. reduce `data.window_size`,
5. lower `runtime.num_workers` if data loading is noisy.

## Suggested next steps

1. Verify PAMAP2 preprocessing assumptions in `src/data/pamap2.py`
2. Verify WISDM parsing assumptions against the exact raw release in use
3. Add subject-wise split controls
4. Add stronger alignment baselines beyond nearest-neighbor
5. Add more robust fusion and missing-modality baselines
6. Add real experiment tracking

## License

MIT
