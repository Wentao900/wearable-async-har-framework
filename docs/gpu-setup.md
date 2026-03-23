# GPU Setup Guide

This project can be run on a machine with **NVIDIA RTX 4090D** and **CUDA 11.8**.

## Recommended environment

- Python: 3.10 or 3.11
- CUDA: 11.8
- PyTorch: build targeting cu118

## 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Install PyTorch for CUDA 11.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## 3. Verify GPU visibility

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

Expected: CUDA available = True, and the GPU name should mention your RTX 4090D.

## 4. Run training

### PAMAP2
```bash
python3 scripts/train.py --config configs/pamap2-gpu.yaml
```

### WISDM
```bash
python3 scripts/train.py --config configs/wisdm-gpu.yaml
```

## 5. Notes on current configs

These GPU configs are still **starter configs**, not tuned SOTA settings.
They mainly do three things:
- switch runtime to `cuda`
- increase `batch_size`
- increase worker count and training budget

If you hit out-of-memory errors, reduce:
- `training.batch_size`
- `model.hidden_dim`
- or `data.window_size`

## 6. Common issues

### CUDA not available
Usually means one of:
- wrong PyTorch build installed (CPU-only wheel)
- NVIDIA driver mismatch
- CUDA runtime not visible in the environment

### Out of memory
Try:
- lowering batch size (e.g. 128 -> 64 -> 32)
- lowering hidden_dim (128 -> 64)
- reducing worker count if data loading is heavy

### Slow data loading
Try adjusting:
- `runtime.num_workers`
- dataset storage location (SSD preferred)

## 7. Honest caveat

The repository now supports GPU execution, but the dataset adapters and benchmark protocols are still **starter paths**. GPU makes training feasible; it does not magically turn the preprocessing and evaluation protocol into a finalized paper setup.
