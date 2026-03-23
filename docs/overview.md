# Project Overview

## Aim

This project targets **asynchronous multimodal fusion for wearable human activity recognition (HAR)** from a review-and-framework perspective.

Instead of pretending to have already completed large-scale deep-learning experiments, the repository is honest about its role:
- it organizes the research question,
- narrows the design space,
- proposes an implementable framework,
- and prepares the codebase for later execution on a machine with sufficient compute.

## Why this topic matters

Wearable activity recognition systems increasingly combine multiple sensing modalities. In practice, however, these modalities are rarely perfectly synchronized.

Common failure modes include:
- different sampling rates,
- missing packets,
- temporary sensor dropout,
- clock drift,
- noisy windows,
- edge-device power constraints.

Many papers benchmark multimodal fusion in relatively clean settings, while real deployment is messier. This project focuses on that mess.

## Repo philosophy

- **review first**: understand the field before over-engineering
- **framework second**: design a modular system that can later be implemented and tested
- **compute-aware honesty**: separate conceptual rigor from unavailable GPU resources
