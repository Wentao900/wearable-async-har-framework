# wearable-async-har-framework

A review-and-framework repo for **asynchronous multimodal fusion in wearable human activity recognition (HAR)**.

## Status

This is an **early public scaffold** for a research repo.
It already includes:
- a CPU-friendly synthetic training path,
- a minimal PyTorch baseline,
- a conservative PAMAP2 starter adapter,
- review/framework documents for turning the repo into a paper or full benchmark later.

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

- **docs/**: research framing, review notes, method design, and experiment plan
- **src/**: starter code for a runnable synthetic scaffold plus a conservative PAMAP2 loader path
- **configs/**: example experiment configurations
- **scripts/**: convenience entrypoints
- **tests/**: smoke tests

## Important honesty note

The runnable default path uses a **tiny synthetic dataset**. It exists only to verify that the project wiring works end-to-end on CPU:
- config loading,
- dataset/dataloader creation,
- explicit masks/timestamps,
- a minimal multimodal model,
- one short train/eval cycle.

It is **not** a real HAR benchmark, and any metrics produced by this scaffold should be treated only as a pipeline smoke test.

PAMAP2 is now wired into the training path too, but with the same honesty:
- the repo does **not** ship PAMAP2 data,
- `src/data/pamap2.py` is a **starter adapter**, not a finalized preprocessing recipe,
- column choices and preprocessing should still be verified against the official dataset documentation before using results in a paper.

## Minimal baseline included

The repo now includes:
- a synthetic dataset adapter under `src/data/synthetic.py`
- a PAMAP2 dataset path under `src/data/pamap2.py`
- a small dataset factory under `src/data/factory.py`
- a simple mask-aware alignment stub
- per-modality PyTorch encoders
- a small fusion block (`mean` or `gated`)
- a classifier head
- a train entry script that reads YAML configs

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

If the raw files are missing, `train.py` now fails clearly with an honest message instead of pretending the dataset is ready.

PAMAP2 caveats for this repo:
- only a small subset of modalities is currently used (`accelerometer`, `gyroscope`)
- the adapter builds fixed windows directly from raw subject files
- this is enough for a clean training path, but not enough to claim finalized benchmark preprocessing

## Configuration

Two example configs are included:
- `configs/base.yaml` → synthetic CPU smoke run
- `configs/pamap2.yaml` → PAMAP2 starter run

Both keep CPU compatibility by default with:
- `runtime.device: cpu`
- `runtime.num_workers: 0`

The training path currently supports:
- `data.dataset: synthetic`
- `data.dataset: pamap2`

## Tests

Run the smoke tests with:

```bash
python3 -m pytest tests/test_smoke.py
```

These tests cover:
- docs presence
- synthetic training smoke run
- a tiny fake-PAMAP2 dataloader smoke run
- honest failure when PAMAP2 data is missing

## Suggested next steps

1. Verify and refine the PAMAP2 column mapping in `src/data/pamap2.py`
2. Add subject-wise split controls instead of the current simple file split
3. Replace the alignment stub with a real asynchronous alignment method
4. Expand the baseline family (e.g. GRU / transformer / missing-modality robustness)
5. Add proper experiment tracking and dataset-specific evaluation

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

MIT (or your preferred license — not set yet)
