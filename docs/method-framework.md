# Proposed Method Framework

## Design goal

Build a modular framework for **asynchronous multimodal fusion in wearable HAR** that can tolerate:
- different sensor frequencies,
- partial missingness,
- timestamp irregularity,
- sensor dropout,
- deployment on constrained hardware.

## System modules

### 1. Modality-specific input adapters
Each sensor stream is converted into a standardized internal representation:
- values
- timestamps
- mask / availability indicators
- optional metadata (sampling rate, device id, placement)

### 2. Async alignment module
This module is the heart of the system.

Candidate strategies:
- interpolation-based alignment
- learned resampling
- timestamp embeddings
- cross-modal temporal attention
- windowed event aggregation

### 3. Modality encoders
Each modality gets its own lightweight encoder.

Examples:
- 1D CNN encoder
- TCN encoder
- GRU/LSTM encoder
- tiny Transformer encoder

### 4. Fusion block
Candidate fusion strategies:
- late fusion
- gated fusion
- cross-attention fusion
- confidence-aware weighted fusion
- missing-modality robust fusion using masks

### 5. Classification head
Maps fused representation to HAR classes.

### 6. Robustness layer
Optional components:
- modality dropout during training
- timestamp jitter augmentation
- missing-window simulation
- confidence calibration

## Suggested initial architecture

A pragmatic v1 architecture:
1. modality-specific 1D CNN or TCN encoders
2. timestamp-aware alignment embeddings
3. gated cross-modal fusion block
4. classification MLP head

This is simpler than going full giant Transformer, and more believable for wearable deployment.

## Key hypothesis

Explicitly modeling asynchrony as structure rather than preprocessing residue should improve robustness under sensor mismatch and dropout.
