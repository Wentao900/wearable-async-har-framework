# Experiment Plan

## Goal

Evaluate whether explicit asynchronous fusion improves wearable HAR robustness over synchronization-heavy baselines.

## Candidate tasks

1. activity classification under fully observed multimodal input
2. activity classification under sampling-rate mismatch
3. activity classification under sensor dropout
4. activity classification under timestamp jitter / drift

## Candidate datasets

- PAMAP2
- Opportunity
- mHealth
- WISDM
- UCI HAR (less multimodal, but useful baseline)

## Baselines

### Baseline A: naive synchronization
Resample all streams to a shared grid and perform early fusion.

### Baseline B: late fusion
Independent encoders per modality, fusion after feature extraction.

### Baseline C: recurrent fusion
GRU/LSTM per modality plus concatenation.

### Proposed model
Timestamp-aware asynchronous multimodal fusion with missingness masks.

## Metrics

- accuracy
- macro F1
- weighted F1
- robustness under missing modality scenarios
- latency / throughput for deployment-oriented experiments

## Ablations

- remove timestamp embedding
- replace learned fusion with simple concatenation
- disable modality dropout
- disable missingness mask
- compare interpolation vs learned alignment

## Stress tests

- 10%, 20%, 30% packet loss
- synthetic timestamp drift
- one modality absent at inference
- reduced window size for edge deployment

## Reporting

The eventual paper should include:
- benchmark table
- robustness table
- ablation table
- architecture diagram
- failure case analysis
