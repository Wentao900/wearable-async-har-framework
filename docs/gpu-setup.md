# GPU Setup Guide

This project includes a **starter GPU path** for **NVIDIA RTX 4090D + CUDA 11.8** machines.

The important word is **starter**:
- these configs are meant to be practical and easy to run,
- they are not sold as tuned benchmark settings,
- they are a clean place to begin iteration.

## Recommended environment

- Python: 3.10 or 3.11
- NVIDIA driver: compatible with CUDA 11.8 wheels
- PyTorch: CUDA 11.8 (`cu118`) build

## 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Install dependencies

Simplest route:

```bash
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

That installs:
- the normal project dependencies from `requirements.txt`,
- CUDA 11.8 PyTorch wheels from the PyTorch index.

If you prefer the older two-step method, this also works:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 3. Verify that CUDA is visible

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

Expected result:
- `cuda available: True`
- GPU name includes your RTX 4090D

## 4. Available GPU presets

### PAMAP2
- `configs/pamap2-gpu-safe.yaml`
- `configs/pamap2-gpu-fast.yaml`

### WISDM
- `configs/wisdm-gpu-safe.yaml`
- `configs/wisdm-gpu-fast.yaml`

The old default-style GPU configs remain for compatibility:
- `configs/pamap2-gpu.yaml`
- `configs/wisdm-gpu.yaml`

## 5. One-command GPU runs

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

If no mode is given, the scripts default to `safe`.

## 6. Direct commands

```bash
python3 scripts/train.py --config configs/pamap2-gpu-safe.yaml
python3 scripts/train.py --config configs/pamap2-gpu-fast.yaml
python3 scripts/train.py --config configs/wisdm-gpu-safe.yaml
python3 scripts/train.py --config configs/wisdm-gpu-fast.yaml
```

## 7. Choosing between safe and fast

Use **safe** when:
- this is the first run on a new machine,
- you want lower OOM risk,
- you are debugging dataset paths or loader behavior.

Use **fast** when:
- the safe config already works,
- you have some VRAM headroom,
- you want higher throughput and are okay with more aggressive settings.

Again: these are **reasonable starting points**, not tuned final answers.

## 8. If you hit problems

### CUDA not available
Usually means one of:
- CPU-only PyTorch got installed,
- the wrong wheel index was used,
- the driver/runtime stack is mismatched,
- the Python environment is not the one you think it is.

### Out of memory
Try:
- switching from `fast` to `safe`,
- reducing `training.batch_size`,
- reducing `model.hidden_dim`,
- reducing `data.window_size`.

### Slow data loading
Try:
- lowering or raising `runtime.num_workers`,
- making sure the dataset is on SSD storage,
- starting with the `safe` preset before pushing workers higher.

## 9. Honest caveat

GPU support here makes the scaffold more usable on a modern NVIDIA machine. It does **not** mean:
- the dataset preprocessing is finalized,
- the evaluation protocol is benchmark-complete,
- the included hyperparameters are tuned for best accuracy.

Treat the configs as sensible launch points, then adapt them to your actual workload.
