# wearable-async-har-framework

A small, honest research scaffold for **asynchronous multimodal fusion in wearable human activity recognition (HAR)**.

## What this repo is

This repository is an **early framework**, not a polished benchmark release. It gives you:
- a runnable synthetic smoke path,
- starter dataset adapters for **PAMAP2** and **WISDM**,
- a lightweight PyTorch baseline with timestamp-aware alignment,
- CPU and GPU example configs,
- simple scripts for launching and logging experiments.

It is meant to help you get experiments moving without pretending the recipes are already tuned or paper-ready.

## What “GPU support” means here

This repo includes a practical GPU starter path for **CUDA 11.8 / RTX-class NVIDIA machines**:
- `requirements-gpu.txt` for a straightforward install,
- **safe** and **fast** GPU presets for PAMAP2 and WISDM,
- dataset-specific helper scripts,
- a generic background launcher for long runs,
- automatic runtime metadata logging into each run directory.

These are **starter configs**, not claims of optimal performance.

## Repo layout

- `configs/` — YAML experiment configs for CPU and GPU runs
- `docs/` — setup notes, framework docs, experiment ideas, release notes
- `scripts/` — training and convenience scripts
- `src/` — data adapters, model code, training loop, config + runtime helpers
- `tests/` — smoke tests

## Important honesty notes

### Synthetic path
The default runnable path uses a **tiny synthetic dataset**. It exists to verify:
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
- evaluation protocol details should be checked before using results in serious reporting.

For **PAMAP2 specifically**, subject-wise splitting is now configurable from YAML. That is useful for reproducible experiments, but it is still just a **configurable starter split mechanism**. The example subject lists in this repo are not presented as a paper-standard protocol.

### GPU presets
The included GPU configs are deliberately readable and conservative enough to start from. They are meant for:
- getting onto CUDA quickly,
- reducing first-run friction,
- giving you a safe preset and a faster preset to iterate from.

They are **not** presented as tuned benchmark recipes.

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
- `outputs/default/results.json` written,
- `outputs/default/runtime_info.json` written.

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

#### PAMAP2 subject split control

You can control subject-wise splits directly in YAML:

```yaml
data:
  dataset: pamap2
  root: data/PAMAP2
  train_subjects: [101, 102, 103, 104, 105, 106]
  val_subjects: [107]
  test_subjects: [108, 109]
```

Behavior is intentionally simple:
- if any of `train_subjects`, `val_subjects`, or `test_subjects` are provided, the loader uses those explicit lists,
- if they are all omitted, the loader falls back to the older automatic file-order split,
- missing splits are treated as empty when explicit splitting is enabled,
- subject IDs may be written as integers like `101` or strings like `subject101`.

This is convenient for controlled experiments, but it should be described honestly in any report: it is a **configurable starter subject split**, not a claimed canonical PAMAP2 protocol.

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

## GPU setup

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

## Foreground GPU runs

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

If you omit the mode argument, each script defaults to `safe`.

## Background / tmux-friendly GPU runs

For longer runs, use the generic launcher or the dataset wrappers in `background` mode.

### Generic launcher

```bash
bash scripts/run_gpu_background.sh --config configs/pamap2-gpu-safe.yaml
bash scripts/run_gpu_background.sh --config configs/wisdm-gpu-fast.yaml --run-name wisdm-fast
```

By default this will:
- create a timestamped run directory under the config’s `logging.output_dir`,
- start training with `nohup ... &`,
- write stdout/stderr to `train.log`,
- save the PID to `run.pid`.

### Dataset-specific background runs

```bash
bash scripts/run_pamap2_gpu.sh safe background
bash scripts/run_pamap2_gpu.sh fast background

bash scripts/run_wisdm_gpu.sh safe background
bash scripts/run_wisdm_gpu.sh fast background
```

This is intentionally shell-friendly: it works in a plain terminal, inside `tmux`, or over SSH.

## What gets logged for each run

Every training run now writes `runtime_info.json` in the output directory.

Typical fields include:
- run status (`running`, `completed`, or `failed`),
- UTC timestamp,
- config path,
- output directory,
- current working directory,
- hostname,
- Python/platform info,
- process PID and argv,
- `CUDA_VISIBLE_DEVICES`,
- torch version,
- CUDA / cuDNN visibility,
- visible GPU names and counts,
- git commit / short commit / branch if available.

Results are still written to `results.json`.

## Example long-run workflow

### Safe PAMAP2 run

```bash
bash scripts/run_pamap2_gpu.sh safe background
```

Then monitor it with:

```bash
tail -f outputs/pamap2-gpu-safe/<timestamp>-pamap2-safe/train.log
cat outputs/pamap2-gpu-safe/<timestamp>-pamap2-safe/runtime_info.json
```

### Fast WISDM run

```bash
bash scripts/run_wisdm_gpu.sh fast background
```

Or launch directly with a custom run name:

```bash
bash scripts/run_gpu_background.sh \
  --config configs/wisdm-gpu-fast.yaml \
  --run-name wisdm-fast-1gpu
```

## Running directly without helper scripts

You can still run training directly:

```bash
python3 scripts/train.py --config configs/pamap2-gpu-safe.yaml
python3 scripts/train.py --config configs/pamap2-gpu-fast.yaml
python3 scripts/train.py --config configs/wisdm-gpu-safe.yaml
python3 scripts/train.py --config configs/wisdm-gpu-fast.yaml
```

If you want a custom output directory for a run:

```bash
python3 scripts/train.py \
  --config configs/pamap2-gpu-safe.yaml \
  --output-dir outputs/manual/pamap2-safe-test
```

## GPU config presets

### PAMAP2
- `configs/pamap2.yaml`
- `configs/pamap2-gpu-safe.yaml`
- `configs/pamap2-gpu-fast.yaml`
- `configs/pamap2-gpu.yaml`

The PAMAP2 example configs now include explicit starter subject lists so the split is visible and editable in one place.

### WISDM
- `configs/wisdm-gpu-safe.yaml`
- `configs/wisdm-gpu-fast.yaml`

Backward-compatible default GPU configs also remain:
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

## Current model / baseline summary

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
3. Compare multiple explicit PAMAP2 subject splits instead of relying on a single starter split
4. Add stronger alignment baselines beyond nearest-neighbor
5. Add more robust fusion and missing-modality baselines
6. Add real experiment tracking if the project grows beyond shell logs + JSON metadata

## License

MIT
