# Release Notes — v0.1.0

## What this release is

This is the **first public scaffold** of `wearable-async-har-framework`.

It is intended to publish the project's direction early and honestly:
- the research framing is present,
- the codebase is runnable,
- the training path works on synthetic data,
- starter adapters exist for PAMAP2 and WISDM,
- but the repository does **not** yet claim a finalized benchmark pipeline.

## Included in v0.1.0

### Research framing
- overview of the problem space
- literature review notes
- method framework draft
- experiment plan

### Runnable scaffold
- CPU-friendly synthetic dataset path
- minimal PyTorch baseline for multimodal fusion
- config-driven training script
- smoke tests

### Real dataset starter adapters
- PAMAP2 starter adapter
- WISDM starter adapter

## Important caveats

- Synthetic metrics are only smoke-test outputs.
- PAMAP2 and WISDM loaders are **starter paths**, not paper-ready preprocessing pipelines.
- Column mappings, parsing assumptions, split policy, and evaluation protocol should be verified before claiming benchmark results.

## Recommended next steps

1. tighten dataset preprocessing protocols
2. implement a real asynchronous alignment module
3. expand the baseline family
4. add experiment tracking and benchmark tables
