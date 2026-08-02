import os
import csv
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from L_action.datasets.L_fusion_dataset_v38_task_aware_sorted import QUBPHEOFusionTaskAwareMergedSortedDataset
from L_action.models.L_action_models import GRUClassifier


VERSION = "V3.8.1_SDM_fusion_taskaware_merged34_GRU_E60_L64_sorted"
RESULT_DIR = "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/V3.8.1_fusion_taskaware_merged34_GRU_E60_L64_sorted"
WEIGHTS_DIR = os.path.join(RESULT_DIR, "weights")

CSV_PATH = "L_action/datasets/L_subtasks_byTask_time_FULL_exactH5.csv"
H5_ROOT = "dataset/landmarks"

SEED = 42
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
L = 64
HIDDEN = 128


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resample_seq(x: np.ndarray, out_len: int = 64) -> np.ndarray:
    t, d = x.shape
    if t == out_len:
        return x.astype(np.float32)

    old_idx = np.linspace(0, 1, t)
    new_idx = np.linspace(0, 1, out_len)

    out = np.zeros((out_len, d), dtype=np.float32)
    for j in range(d):
        out[:, j] = np.interp(new_idx, old_idx, x[:, j])

    return out


class ResampleWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds, L=64):
        self.ds = base_ds
        self.L = L
        self.df = base_ds.df
        self.subtask2idx = base_ds.subtask2idx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        x = resample_seq(x.numpy(), self.L)
        return torch.tensor(x).float(), y.long()


def class_weights(ds, device):
    counts = np.zeros(len(ds.subtask2idx), dtype=np.int64)
    for i in range(len(ds)):
        _, y = ds[i]
        counts[int(y)] += 1

    counts = np.maximum(counts, 1)
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32).to(device)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()

    all_y, all_pred = [], []
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        pred = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_y, all_pred),
        "macro_f1": f1_score(all_y, all_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(all_y, all_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(all_y, all_pred, average="macro", zero_division=0),
        "y_true": all_y,
        "y_pred": all_pred,
    }


if __name__ == "__main__":
    set_seed(SEED)

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_tr_base = QUBPHEOFusionTaskAwareMergedSortedDataset(CSV_PATH, H5_ROOT, split="train")
    ds_va_base = QUBPHEOFusionTaskAwareMergedSortedDataset(CSV_PATH, H5_ROOT, split="val")
    ds_te_base = QUBPHEOFusionTaskAwareMergedSortedDataset(CSV_PATH, H5_ROOT, split="test")

    ds_tr = ResampleWrapper(ds_tr_base, L=L)
    ds_va = ResampleWrapper(ds_va_base, L=L)
    ds_te = ResampleWrapper(ds_te_base, L=L)

    x0, y0 = ds_tr[0]
    input_dim = x0.shape[1]
    num_classes = len(ds_tr.subtask2idx)

    print("Version:", VERSION)
    print("Sanity:", x0.shape, "label:", y0.item())
    print("Input dim:", input_dim)
    print("Classes:", num_classes)
    print("Device:", device)

    assert input_dim == 207, f"Expected task-aware fusion input_dim=207, got {input_dim}"
    assert num_classes == 34, f"Expected merged-34 SDM labels, got {num_classes}"

    train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=(device == "cuda"))
    val_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))
    test_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

    model = GRUClassifier(input_dim, HIDDEN, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(ds_tr, device))
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(WEIGHTS_DIR, "best_model.pt")
    last_model_path = os.path.join(WEIGHTS_DIR, "last_model.pt")

    training_log_path = os.path.join(RESULT_DIR, "training_log.csv")

    with open(training_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_accuracy", "train_macro_f1", "train_macro_precision", "train_macro_recall",
            "val_loss", "val_accuracy", "val_macro_f1", "val_macro_precision", "val_macro_recall",
        ])

        for epoch in range(1, EPOCHS + 1):
            tr = run_epoch(model, train_loader, criterion, optimizer, device)
            va = run_epoch(model, val_loader, criterion, None, device)

            writer.writerow([
                epoch,
                tr["loss"], tr["accuracy"], tr["macro_f1"], tr["macro_precision"], tr["macro_recall"],
                va["loss"], va["accuracy"], va["macro_f1"], va["macro_precision"], va["macro_recall"],
            ])

            if va["macro_f1"] > best_val_f1:
                best_val_f1 = va["macro_f1"]
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

            print(
                f"Epoch {epoch}: "
                f"train acc={tr['accuracy']:.4f} f1={tr['macro_f1']:.4f} | "
                f"val acc={va['accuracy']:.4f} f1={va['macro_f1']:.4f} "
                f"precision={va['macro_precision']:.4f} recall={va['macro_recall']:.4f}"
            )

    torch.save(model.state_dict(), last_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    te = run_epoch(model, test_loader, criterion, None, device)

    metrics = {
        "version": VERSION,
        "stage": "SDM current-subtask detection",
        "input": "fusion visual features + task-aware feature",
        "label_space": "merged-34 current-subtask labels",
        "backbone": "GRU",
        "input_dim": input_dim,
        "sequence_length": L,
        "hidden_dim": HIDDEN,
        "num_classes": num_classes,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "loss": "Weighted Cross Entropy",
        "best_epoch_by_val_macro_f1": best_epoch,
        "test_accuracy": te["accuracy"],
        "test_macro_f1": te["macro_f1"],
        "test_macro_precision": te["macro_precision"],
        "test_macro_recall": te["macro_recall"],
    }

    with open(os.path.join(RESULT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(RESULT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    model_info = f"""Model: GRUClassifier
Version: {VERSION}
Task: SDM current-subtask detection
Input shape per sample: (L={L}, D={input_dim})
Input features: fusion visual vector = hands + gaze + objects + task-aware feature
Label space: merged-34 current-subtask labels
Backbone: nn.GRU(input_dim={input_dim}, hidden_size={HIDDEN}, batch_first=True)
Classifier: Linear(hidden={HIDDEN} -> classes={num_classes})
Loss: weighted cross entropy
Optimizer: AdamW, lr={LR}
Batch size: {BATCH_SIZE}
Epochs: {EPOCHS}
Selection: best model by validation Macro-F1
Evaluation: participant-held-out test split
Primary metric: Macro-F1
Secondary metrics: Accuracy, Macro-Precision, Macro-Recall
"""

    with open(os.path.join(RESULT_DIR, "model_info.txt"), "w", encoding="utf-8") as f:
        f.write(model_info)

    with open(os.path.join(RESULT_DIR, "results_log.txt"), "w", encoding="utf-8") as f:
        f.write(model_info)
        f.write("\nFinal TEST using best validation model:\n")
        f.write(f"Test Accuracy: {te['accuracy']:.6f}\n")
        f.write(f"Test Macro-F1: {te['macro_f1']:.6f}\n")
        f.write(f"Test Macro-Precision: {te['macro_precision']:.6f}\n")
        f.write(f"Test Macro-Recall: {te['macro_recall']:.6f}\n")
        f.write(f"Best epoch by validation Macro-F1: {best_epoch}\n")

    labels = list(range(num_classes))

    cm = confusion_matrix(te["y_true"], te["y_pred"], labels=labels)
    pd.DataFrame(cm).to_csv(os.path.join(RESULT_DIR, "confusion_matrix.csv"), index=False)

    report = classification_report(te["y_true"], te["y_pred"], labels=labels, zero_division=0)
    with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame({
        "y_true": te["y_true"],
        "y_pred": te["y_pred"],
    }).to_csv(os.path.join(RESULT_DIR, "test_predictions.csv"), index=False)

    print("\nSaved results to:", RESULT_DIR)
    print("Best model:", best_model_path)
    print("TEST acc:", round(te["accuracy"], 6))
    print("TEST macro_f1:", round(te["macro_f1"], 6))
    print("TEST precision:", round(te["macro_precision"], 6))
    print("TEST recall:", round(te["macro_recall"], 6))
