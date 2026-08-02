# QUB-PHEO SDM--IPM Benchmark Release

This repository provides the curated public release artifacts for the QUB-PHEO SDM--IPM benchmark paper.

The release contains final result artifacts only. It does not include the raw QUB-PHEO dataset, private development files, intermediate experiments, or model weights.

## Contents

- `results/01_final_SDM_V3.8.1_sorted_GRU`: final current-subtask detection result.
- `results/02_final_soft_IPM_V3.8.10d_MLP_K4`: final soft-state IPM result.
- `results/04_hard_chained_IPM_V3.8.4_GRU_K4`: hard-chained IPM comparator.

Each result folder includes metrics, classification reports, confusion matrices, logs, predictions, and training curves where available.

## History-depth notation

The implementation folder names use the original code history length, where `K=4` means four observed SDM states are used as input: the current state plus three previous states.

In the paper, the history depth is reported with the current state treated as depth 0.

| Code notation | Paper notation | Meaning |
|---|---|---|
| K=1 | K=0 | current SDM state only |
| K=3 | K=2 | current SDM state + two previous states |
| K=4 | K=3 | current SDM state + three previous states |

Thus, folders labelled `K4` in this release correspond to `K=3` in the paper notation.

## Dataset note

The raw QUB-PHEO dataset is not redistributed in this repository. Users should obtain dataset access through the official dataset release and follow its terms of use.
