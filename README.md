# wearable-async-har-framework

A review-and-framework repo for **asynchronous multimodal fusion in wearable human activity recognition (HAR)**.

## Status

This is an **early public scaffold** for a research repo.
It already includes:
- a CPU-friendly synthetic training path,
- a minimal PyTorch baseline,
- conservative starter adapters for **PAMAP2** and **WISDM**,
- a configurable async alignment baseline,
- review/framework documents for evolving the repo into a stronger benchmark or paper codebase later.

It does **not** yet claim a finalized benchmark pipeline.

## Problem

Wearable HAR often combines multiple sensors (e.g. accelerometer, gyroscope, ECG, EMG, auxiliary context), but real deployments are messy:
- streams arrive at different rates,
- timestamps drift,
- packets drop,
- modalities go missing,
- edge hardware is constrained.

This repo is organized around that gap: **asynchronous multimodal fusion**, not just clean synchronized benchmarking.

## Repo contents

- **docs/**: research framing, review notes, method design, experiment plan, and release notes
- **src/**: runnable synthetic scaffold plus conservative starter dataset loaders
- **configs/**: example experiment configurations
- **scripts/**: convenience entrypoints
- **tests/**: smoke tests
- **CITATION.cff**: citation metadata

## Important honesty note

The default runnable path uses a **tiny synthetic dataset**. It exists only to verify that the project wiring works end-to-end on CPU:
- config loading,
- dataset/dataloader creation,
- explicit masks/timestamps,
- a minimal multimodal model,
- one short train/eval cycle.

It is **not** a real HAR benchmark, and any metrics produced by this scaffold should be treated only as a pipeline smoke test.

PAMAP2 and WISDM are wired into the training path too, but with the same honesty:
- the repo does **not** ship those datasets,
- `src/data/pamap2.py` and `src/data/wisdm.py` are **starter adapters**, not finalized preprocessing recipes,
- file parsing assumptions, column mappings, and evaluation protocols should be verified before using results in a paper.

## Minimal baseline included

The repo includes:
- a synthetic dataset adapter under `src/data/synthetic.py`
- a PAMAP2 dataset path under `src/data/pamap2.py`
- a WISDM dataset path under `src/data/wisdm.py`
- a small dataset factory under `src/data/factory.py`
- a timestamp-aware nearest-neighbor alignment baseline
- per-modality PyTorch encoders
- a small fusion block (`mean` or `gated`)
- a classifier head
- a train entry script that reads YAML configs

## Alignment v1: what it actually does

The current async alignment module is intentionally modest.
It is a **baseline**, not a claimed research contribution.

Given per-modality `values`, `timestamps`, and `masks`, it:
- chooses a reference timeline,
- aligns every modality onto that timeline with **nearest-neighbor timestamp matching**,
- ignores missing source samples,
- zero-fills reference positions that cannot be aligned,
- returns alignment metadata such as the selected reference modality, aligned masks, source indices, and valid ratios.

Supported reference timeline strategies:
- `densest` → choose the modality with the highest observed density in the batch
- `first` → choose the first configured modality
- a modality name such as `accelerometer` or `gyroscope`

This keeps irregular timing and missingness explicit while staying lightweight enough for CPU smoke runs.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/print_project_summary.py
```

## Run the synthetic smoke path

```bash
python3 scripts/train.py --config configs/base.yaml
```

Expected output:
- CPU-friendly training run
- JSON metrics printed to stdout
- saved file at `outputs/default/results.json`

## Run the PAMAP2 path

Expected raw-data layout:

```bash
data/PAMAP2/Protocol/subject101.dat
data/PAMAP2/Protocol/subject102.dat
```

Optional quick inspection before training:

```bash
python3 scripts/check_pamap2.py --root data/PAMAP2 --split train
```

Training command:

```bash
python3 scripts/train.py --config configs/pamap2.yaml
```

If the raw files are missing, `train.py` fails clearly with an honest message.

PAMAP2 caveats for this repo:
- only a small subset of modalities is currently used (`accelerometer`, `gyroscope`)
- the adapter builds fixed windows directly from raw subject files
- this is enough for a clean training path, but not enough to claim finalized benchmark preprocessing

## Run the WISDM path

Expected raw-data layout:

```bash
data/WISDM/WISDM_ar_v1.1_raw.txt
```

Training command:

```bash
python3 scripts/train.py --config configs/wisdm.yaml
```

If the raw file is missing, `train.py` fails clearly with an honest message.

WISDM caveats for this repo:
- this starter adapter assumes a common raw text format with CSV-like rows
- only accelerometer is used in the initial scaffold
- parsing assumptions should be verified against the exact WISDM release you use before reporting results

## Configuration

Three example configs are included:
- `configs/base.yaml` → synthetic CPU smoke run
- `configs/pamap2.yaml` → PAMAP2 starter run
- `configs/wisdm.yaml` → WISDM starter run

Both real-dataset starter configs keep CPU compatibility by default with:
- `runtime.device: cpu`
- `runtime.num_workers: 0`

The training path currently supports:
- `data.dataset: synthetic`
- `data.dataset: pamap2`
- `data.dataset: wisdm`

The alignment module is configurable from YAML instead of being hard-coded, which makes simple ablations possible without touching Python source.

Example:

```yaml
model:
  alignment_strategy: nearest-neighbor
  reference_timeline_strategy: densest
```

## Tests

Run the smoke tests with:

```bash
python3 -m pytest tests/test_smoke.py
python3 -m pytest tests/test_alignment.py
```

These tests cover:
- docs presence
- synthetic training smoke run
- a tiny fake-PAMAP2 dataloader smoke run
- honest failure when PAMAP2 data is missing
- honest failure when WISDM data is missing
- alignment reference selection and mask-aware nearest-neighbor behavior

## Suggested next steps

1. Verify and refine the PAMAP2 column mapping in `src/data/pamap2.py`
2. Verify the WISDM parsing assumptions against the exact raw release being used
3. Add subject-wise split controls instead of the current simple file split
4. Expose more alignment options such as fixed-grid or union-grid baselines
5. Expand the baseline family (e.g. GRU / transformer / missing-modality robustness)
6. Add proper experiment tracking and dataset-specific evaluation

## Candidate datasets

- PAMAP2
- WISDM
- UCI HAR
- Opportunity
- mHealth

## Candidate baselines

- Late fusion MLP/CNN
- LSTM / GRU sensor encoders
- TCN-based encoder
- Transformer/Conformer-based fusion
- Missing-modality robust fusion

## License

MIT
