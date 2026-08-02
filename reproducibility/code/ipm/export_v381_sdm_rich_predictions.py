import os
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from L_action.datasets.L_fusion_dataset_v38_task_aware_sorted import (
    QUBPHEOFusionTaskAwareMergedSortedDataset,
)
from L_action.L_train_action_v381_sdm_fusion_taskaware_gru_e60_sorted import (
    GRUClassifier,
    ResampleWrapper,
)

CSV_PATH = "L_action/datasets/L_subtasks_byTask_time_FULL_exactH5.csv"
H5_ROOT = "dataset/landmarks"

VERSION = "V3.8.1_SDM_fusion_taskaware_merged34_GRU_E60_L64_sorted"
RESULT_DIR = "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/V3.8.1_fusion_taskaware_merged34_GRU_E60_L64_sorted"
WEIGHT_PATH = os.path.join(RESULT_DIR, "weights", "best_model.pt")

OUT_DIR = os.path.join(RESULT_DIR, "rich_predictions")
os.makedirs(OUT_DIR, exist_ok=True)

L = 64
HIDDEN = 128
BATCH_SIZE = 64


def entropy_np(p):
    eps = 1e-12
    return float(-(p * np.log(p + eps)).sum())


def export_split(split):
    base_ds = QUBPHEOFusionTaskAwareMergedSortedDataset(CSV_PATH, H5_ROOT, split=split)
    ds = ResampleWrapper(base_ds, L=L)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    x0, y0 = ds[0]
    input_dim = x0.shape[1]
    num_classes = len(base_ds.subtasks)

    idx2subtask = {i: s for i, s in enumerate(base_ds.subtasks)}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN,
        num_classes=num_classes,
    ).to(device)

    state = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_true = []
    all_pred = []
    all_conf = []
    all_entropy = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            conf = probs.max(axis=1)

            all_true.extend(y.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_conf.extend(conf.tolist())
            all_entropy.extend([entropy_np(p) for p in probs])
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    meta = base_ds.df.reset_index(drop=True).copy()

    out_df = pd.DataFrame({
        "split": split,
        "file": meta["filename"],
        "participant": meta["participant"],
        "task": meta["task"],
        "start_time": meta["start_time"],
        "end_time": meta["end_time"],
        "true_label_id": all_true,
        "pred_label_id": all_pred,
        "true_label_name": [idx2subtask[int(i)] for i in all_true],
        "pred_label_name": [idx2subtask[int(i)] for i in all_pred],
        "correct": [int(t == p) for t, p in zip(all_true, all_pred)],
        "confidence": all_conf,
        "entropy": all_entropy,
    })

    for i in range(num_classes):
        out_df[f"prob_{i:02d}_{idx2subtask[i]}"] = all_probs[:, i]

    out_path = os.path.join(OUT_DIR, f"v381_sdm_rich_predictions_{split}.csv")
    out_df.to_csv(out_path, index=False)

    summary = {
        "version": VERSION,
        "split": split,
        "rows": int(len(out_df)),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "checkpoint": WEIGHT_PATH,
        "output_csv": out_path,
        "columns_include": [
            "file",
            "participant",
            "task",
            "start_time",
            "end_time",
            "true_label_id",
            "pred_label_id",
            "true_label_name",
            "pred_label_name",
            "confidence",
            "entropy",
            "prob_00...prob_33",
        ],
    }

    summary_path = os.path.join(OUT_DIR, f"v381_sdm_rich_predictions_{split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] {split}: {out_df.shape} -> {out_path}")

    return out_df


def main():
    dfs = []
    for split in ["train", "val", "test"]:
        dfs.append(export_split(split))

    all_df = pd.concat(dfs, ignore_index=True)
    all_path = os.path.join(OUT_DIR, "v381_sdm_rich_predictions_all_splits.csv")
    all_df.to_csv(all_path, index=False)

    print("[OK] all_splits:", all_df.shape, "->", all_path)


if __name__ == "__main__":
    main()
