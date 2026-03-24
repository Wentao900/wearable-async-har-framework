# PAMAP2 experiment comparison notes

A simple place to compare explicit subject-wise PAMAP2 runs.

## Why keep this table?

PAMAP2 results can move noticeably when the subject split changes. That split sensitivity is one of the main observations worth tracking here, especially because the configs in this repo are **starter subject-wise evaluation configs**, not a canonical benchmark protocol.

## Suggested metadata to record

- config file used
- train / val / test subject lists
- output directory or run ID
- hardware notes if relevant
- any preprocessing or hyperparameter deviations

## Current result table

| Split | Train subjects | Val subjects | Test subjects | Best val acc | Test acc | Test loss | Best epoch | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| split-a | 101,102,103,104,105,106 | 107 | 108,109 | 0.7083 | 0.3209 | 11.83 | 13 | Severe collapse on test despite decent val accuracy |
| split-b | 101,102,104,105,107,108 | 106 | 103,109 | 0.7126 | 0.6721 | 1.01 | 18 | Strongest and most stable result so far |
| split-c | 101,102,103,104,105 | 106,107 | 108,109 | 0.7047 | 0.3075 | 13.07 | 9 | Still collapses even with two validation subjects |
| split-d | 101,102,104,107,108 | 105,106 | 103,109 | 0.6421 | 0.6373 | 1.16 | 10 | Healthy baseline behavior, slightly below split-b |
| split-e | 102,103,105,106,109 | 107,108 | 101,104 | 0.4960 | 0.6085 | 1.55 | 17 | Lower val accuracy but still normal test behavior |

## Stage summary

### Main observation

The baseline is **highly sensitive to subject-wise split choice**.

Across the current five runs, test performance separates into two broad regimes:
- a **collapsed regime** around **0.31** test accuracy,
- a **normal regime** around **0.61-0.67** test accuracy.

### Strongest pattern so far

The most suspicious held-out test combination is:
- `test_subjects = [108,109]`

This combination appears in:
- `split-a`
- `split-c`

Both runs collapse badly:
- split-a test acc = **0.3209**
- split-c test acc = **0.3075**

By contrast, splits without that exact held-out pair behave much better:
- split-b (`[103,109]`) -> **0.6721**
- split-d (`[103,109]`) -> **0.6373**
- split-e (`[101,104]`) -> **0.6085**

This suggests that the current baseline has a pronounced weakness on certain held-out subject combinations, with `[108,109]` being the strongest current suspect.

### Validation accuracy is not a reliable predictor of test behavior

A second important finding is that **validation accuracy does not reliably predict held-out test performance**.

Examples:
- split-a and split-c both reach around **0.70** best validation accuracy, yet both collapse to around **0.31** test accuracy.
- split-e reaches only **0.4960** best validation accuracy, yet still achieves **0.6085** test accuracy.

So even improving the validation split from 1 subject to 2 subjects does **not automatically fix** the representativeness problem.

## Practical interpretation

At this stage, the fairest summary is:
- under some held-out subject choices, this baseline can achieve roughly **0.61-0.67** test accuracy;
- under the harder `[108,109]` test combination, it collapses to roughly **0.31**;
- therefore, any single PAMAP2 split should be treated cautiously.

## Reporting-ready takeaway

A concise wording for notes or a draft paper section:

> Across five explicit subject-wise PAMAP2 splits, the baseline exhibits strong split sensitivity. In particular, runs with `test_subjects = [108,109]` collapse to roughly 30% test accuracy, while other held-out subject combinations remain in the 61%-67% range. Moreover, validation accuracy does not reliably predict test performance, suggesting that the current validation protocol is not sufficiently representative of subject-level generalization difficulty.

## Next steps

- run more subject-wise splits if a broader stability estimate is needed
- inspect subject-level difficulty, especially for subject 108 and the `[108,109]` pair
- consider stronger validation designs or multi-split reporting instead of relying on a single split
- treat these numbers as **starter baseline evidence**, not as a final benchmark claim
