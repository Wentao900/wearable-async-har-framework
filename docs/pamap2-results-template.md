# PAMAP2 experiment comparison template

A simple place to compare explicit subject-wise PAMAP2 runs.

## Why keep this table?

PAMAP2 results can move noticeably when the subject split changes. That split sensitivity is one of the main observations worth tracking here, especially because the configs in this repo are **starter subject-wise evaluation configs**, not a canonical benchmark protocol.

## Suggested metadata to record

- config file used
- train / val / test subject lists
- output directory or run ID
- hardware notes if relevant
- any preprocessing or hyperparameter deviations

## Result table

| Split | Train subjects | Val subjects | Test subjects | Best val acc | Test acc | Test loss | Best epoch | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| split-a | 101,102,103,104,105,106 | 107 | 108,109 |  |  |  |  |  |
| split-b | 101,102,104,105,107,108 | 106 | 103,109 |  |  |  |  |  |
| split-c | 101,102,103,104,105 | 106,107 | 108,109 |  |  |  |  |  |
| split-d | 101,102,104,107,108 | 105,106 | 103,109 |  |  |  |  |  |
| split-e | 102,103,105,106,109 | 107,108 | 101,104 |  |  |  |  |  |

## Notes prompt

When filling the `Notes` column, it is worth stating whether:
- validation and test metrics move together or diverge,
- some held-out subjects appear much harder than others,
- early stopping picked very different epochs across splits,
- the run should be treated as a quick starter result rather than a reporting-grade benchmark number.
