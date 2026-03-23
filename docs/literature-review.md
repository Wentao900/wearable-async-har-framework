# Literature Review Notes

## Scope

This review focuses on research at the intersection of:
- wearable sensing,
- multimodal time series,
- asynchronous or imperfectly aligned sensor streams,
- human activity recognition (HAR),
- robust fusion under deployment constraints.

## Core review questions

1. How do existing wearable HAR systems represent multiple sensor modalities?
2. How do they handle different sampling rates or missing timestamps?
3. Is asynchrony treated as noise, a preprocessing issue, or a first-class modeling problem?
4. Which fusion strategies are common: early, late, hybrid, cross-attention, gating?
5. What robustness mechanisms exist for missing modalities and sensor dropout?
6. Which datasets realistically capture asynchronous behavior?

## Working taxonomy

### By sensor modality
- inertial only: accelerometer, gyroscope, magnetometer
- physiological: ECG, EMG, EEG, PPG
- contextual: audio, GPS, proximity, environmental sensors
- mixed multimodal wearable setups

### By synchronization assumption
- fully synchronized windows
- resampled / interpolated pseudo-synchrony
- event-driven or timestamp-aware fusion
- missing-modality / partially observed fusion

### By model family
- handcrafted features + classical ML
- CNN / TCN encoders
- RNN / GRU / LSTM models
- Transformer / attention-based fusion
- graph or cross-modal interaction models
- lightweight edge-oriented models

## Known gaps

- Most work assumes tidy alignment after preprocessing.
- Realistic sensor drift and packet loss are often under-modeled.
- Benchmark design for asynchronous wearable HAR is weak.
- Edge deployment constraints are often discussed but not co-designed with model structure.

## What to collect next

- papers explicitly addressing asynchronous multimodal fusion
- datasets with timestamps or naturally different sampling rates
- methods for missing-modality robustness
- lightweight methods suitable for on-device inference
