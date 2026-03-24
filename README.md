# wearable-async-har-framework

A small, honest research scaffold for **asynchronous multimodal fusion in wearable human activity recognition (HAR)**.

## What this repo is

This repository is an **early framework**, not a polished benchmark release. It gives you:
- a runnable synthetic smoke path,
- starter dataset adapters for **PAMAP2** and **WISDM**,
- a lightweight PyTorch baseline with timestamp-aware alignment,
- CPU and GPU example configs,
- simple scripts for launching and logging experiments,
- minimal best-checkpoint and early-stopping support for validation-driven runs.

It is meant to help you get experiments moving without pretending the recipes are already tuned or paper-ready.

## What “GPU support” means here

This repo includes a practical GPU starter path for **CUDA 11.8 / RTX-class NVIDIA machines**:
- `requirements-gpu.txt` for a straightforward install,
- **safe**, **fast**, and explicit **split-comparison** GPU presets for PAMAP2,
- safe and fast GPU presets for WISDM,
- dataset-specific helper scripts,
- a generic background launcher for long runs,
- automatic runtime metadata logging into each run directory,
- optional best-checkpoint saving and simple early stopping.

These are **starter configs**, not claims of optimal performance.

## Repo layout

- `configs/` — YAML experiment configs for CPU and GPU runs
- `docs/` — setup notes, framework docs, experiment ideas, release notes, result templates
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

For **PAMAP2 specifically**, subject-wise splitting is configurable from YAML. That is useful for reproducible experiments, but it is still just a **configurable starter split mechanism**. The example subject lists in this repo are not presented as a paper-standard or canonical benchmark protocol.

### GPU presets
The included GPU configs are deliberately readable and conservative enough to start from. They are meant for:
- getting onto CUDA quickly,
- reducing first-run friction,
- giving you a safe preset, a faster preset, and several explicit split variants to compare.

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

#### Included PAMAP2 GPU split variants

Five explicit comparison configs are included so you can run subject-split comparisons without editing YAML first:

- `configs/pamap2-gpu-split-a.yaml`
  - train: `101, 102, 103, 104, 105, 106`
  - val: `107`
  - test: `108, 109`
- `configs/pamap2-gpu-split-b.yaml`
  - train: `101, 102, 104, 105, 107, 108`
  - val: `106`
  - test: `103, 109`
- `configs/pamap2-gpu-split-c.yaml`
  - train: `101, 102, 103, 104, 105`
  - val: `106, 107`
  - test: `108, 109`
- `configs/pamap2-gpu-split-d.yaml`
  - train: `101, 102, 104, 107, 108`
  - val: `105, 106`
  - test: `103, 109`
- `configs/pamap2-gpu-split-e.yaml`
  - train: `102, 103, 105, 106, 109`
  - val: `107, 108`
  - test: `101, 104`

The three newer configs (`split-c`, `split-d`, `split-e`) each use **two validation subjects** so validation is a bit less tied to one participant. They are still **practical starter comparisons**, not canonical protocol claims.

A simple comparison sheet is also included at:
- `docs/pamap2-results-template.md`

That template is intentionally lightweight and explicitly calls out **split sensitivity** as something worth recording.

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
python3 scripts/train.py --config configs/pamap2-gpu-split-a.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-b.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-c.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-d.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-e.yaml
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
bash scripts/run_gpu_background.sh --config configs/pamap2-gpu-split-a.yaml --run-name pamap2-split-a
bash scripts/run_gpu_background.sh --config configs/pamap2-gpu-split-b.yaml --run-name pamap2-split-b
bash scripts/run_gpu_background.sh --config configs/pamap2-gpu-split-c.yaml --run-name pamap2-split-c
bash scripts/run_gpu_background.sh --config configs/pamap2-gpu-split-d.yaml --run-name pamap2-split-d
bash scripts/run_gpu_background.sh --config configs/pamap2-gpu-split-e.yaml --run-name pamap2-split-e
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

## Best checkpoint and early stopping

Training now supports a small, explicit validation-driven control block in YAML:

```yaml
training:
  epochs: 20
  checkpoint:
    save_best: true
    monitor: val_accuracy
    filename: best_checkpoint.pt
  early_stopping:
    enabled: true
    monitor: val_accuracy
    mode: max
    patience: 5
    min_delta: 0.0
```

### Supported behavior

- `checkpoint.save_best: true` saves the best validation checkpoint into the run output directory.
- `checkpoint.monitor` and `early_stopping.monitor` currently expect validation metrics such as:
  - `val_accuracy`
  - `val_loss`
- `early_stopping.mode` can be:
  - `max` for metrics like accuracy
  - `min` for metrics like loss
- `patience` is the number of consecutive non-improving validation epochs tolerated before stopping.
- `min_delta` is the minimum change required to count as a real improvement.

This is intentionally minimal. It is not a full training-framework callback system.

### Where results show up

`results.json` now includes additional run summary fields when applicable:
- `epochs_requested`
- `epochs_completed`
- `stopped_early`
- `best.epoch`
- `best.monitor`
- `best.value`
- `best.checkpoint_path`

If a best checkpoint was saved, the trainer evaluates the test split using that restored best checkpoint rather than blindly using the last epoch.

## What gets logged for each run

Every training run writes `runtime_info.json` in the output directory.

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

### Split comparison workflow

```bash
python3 scripts/train.py --config configs/pamap2-gpu-split-c.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-d.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-e.yaml
```

Then compare:
- `outputs/pamap2-gpu-split-c/results.json`
- `outputs/pamap2-gpu-split-d/results.json`
- `outputs/pamap2-gpu-split-e/results.json`
- `docs/pamap2-results-template.md`

Look especially at:
- `best.epoch`
- `best.value`
- `stopped_early`
- final test metrics
- whether different subject partitions materially change the conclusion

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
python3 scripts/train.py --config configs/pamap2-gpu-split-a.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-b.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-c.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-d.yaml
python3 scripts/train.py --config configs/pamap2-gpu-split-e.yaml
python3 scripts/train.py --config configs/wisdm-gpu-safe.yaml
python3 scripts/train.py --config configs/wisdm-gpu-fast.yaml
```

If you want a custom output directory for a run:

```bash
python3 scripts/train.py \
  --config configs/pamap2-gpu-split-c.yaml \
  --output-dir outputs/manual/pamap2-split-c-test
```

## GPU config presets

### PAMAP2
- `configs/pamap2.yaml`
- `configs/pamap2-gpu.yaml`
- `configs/pamap2-gpu-safe.yaml`
- `configs/pamap2-gpu-fast.yaml`
- `configs/pamap2-gpu-split-a.yaml`
- `configs/pamap2-gpu-split-b.yaml`
- `configs/pamap2-gpu-split-c.yaml`
- `configs/pamap2-gpu-split-d.yaml`
- `configs/pamap2-gpu-split-e.yaml`

The PAMAP2 example configs include explicit starter subject lists so the split is visible and editable in one place.

### WISDM
- `configs/wisdm-gpu-safe.yaml`
- `configs/wisdm-gpu-fast.yaml`
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

**Split comparison** presets:
- keep the model recipe roughly fixed,
- change subject partitions explicitly,
- useful when you want a quick sense of split sensitivity before doing more serious protocol work.

If a fast preset runs out of memory or becomes unstable, step back to the safe preset first.

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
5. lower `runtime.num_workers` if data loading is noisy,
6. disable early stopping first only if you are specifically debugging full-length runs.

## Suggested next steps

1. Verify PAMAP2 preprocessing assumptions in `src/data/pamap2.py`
2. Compare multiple explicit PAMAP2 subject splits instead of relying on a single starter split
3. Use `docs/pamap2-results-template.md` to keep split comparisons in one place
4. Verify WISDM parsing assumptions against the exact raw release in use
5. Add stronger alignment baselines beyond nearest-neighbor
6. Add more robust fusion and missing-modality baselines
7. Add real experiment tracking if the project grows beyond shell logs + JSON metadata

## License

MIT
