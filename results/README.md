# Paper Benchmark Final Results

This folder contains the final curated result artifacts used for the QUB-PHEO SDM--IPM benchmark paper.

## Included artifacts

- `01_final_SDM_V3.8.1_sorted_GRU`: final current-subtask detection result.
- `02_final_soft_IPM_V3.8.10d_MLP_K4`: final soft-state IPM result.
- `04_hard_chained_IPM_V3.8.4_GRU_K4`: hard-chained IPM comparator.

Each result folder includes metrics, classification reports, confusion matrices, logs, predictions, and training curves where available.

## History-depth notation

The implementation folder names use the original code history length, where `K=4` means four observed SDM states are used as input: the current state plus three previous states.

In the paper, the history depth is reported with the current state treated as depth 0. Therefore:

| Code notation | Paper notation | Meaning |
|---|---|---|
| K=1 | K=0 | current SDM state only |
| K=3 | K=2 | current SDM state + two previous states |
| K=4 | K=3 | current SDM state + three previous states |

Thus, folders labelled `K4` in this release correspond to `K=3` in the paper notation.

Intermediate development runs, intermediate experiments, raw dataset files, and model weights are not included in this curated release folder.
