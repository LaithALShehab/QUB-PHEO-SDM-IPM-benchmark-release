import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from L_action.transition.transition_fusion_dataset import QUBPHEOTransitionFusionDataset


RESULT_DIR = "L_action/docs/results/V2.4_transition_fusion_taskAware"
os.makedirs(RESULT_DIR, exist_ok=True)


def resample_seq(x: np.ndarray, L: int = 64) -> np.ndarray:
    T, D = x.shape
    if T == L:
        return x

    old_idx = np.linspace(0, 1, T)
    new_idx = np.linspace(0, 1, L)

    out = np.zeros((L, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_idx, old_idx, x[:, d])

    return out


class TransitionResampleTaskWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds, L=64, csv_path=None):
        self.ds = base_ds
        self.L = L

        df = pd.read_csv(csv_path, sep="\t")
        self.tasks = sorted(df["task"].unique())
        self.task2idx = {t: i for i, t in enumerate(self.tasks)}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds.df.iloc[idx]
        task_id = self.task2idx[row["task"]]

        x, y = self.ds[idx]
        x = resample_seq(x.numpy(), self.L)

        return torch.tensor(x).float(), torch.tensor(task_id).long(), y.long()


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden=128,
        num_classes=36,
        num_tasks=9,
        task_emb_dim=16,
    ):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.task_emb = nn.Embedding(num_tasks, task_emb_dim)
        self.fc = nn.Linear(hidden + task_emb_dim, num_classes)

    def forward(self, x, task_id):
        _, h = self.gru(x)
        h = h.squeeze(0)
        t = self.task_emb(task_id)
        z = torch.cat([h, t], dim=1)
        return self.fc(z)


def get_class_weights(ds, num_classes=36):
    counts = np.zeros(num_classes, dtype=np.int64)

    for i in range(len(ds)):
        _, _, y = ds[i]
        counts[int(y)] += 1

    counts = np.maximum(counts, 1)
    w = counts.sum() / counts
    w = w / w.mean()

    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, criterion, optim=None, device="cpu"):
    train = optim is not None
    model.train() if train else model.eval()

    ys, preds = [], []
    total_loss = 0.0

    for x, task_id, y in loader:
        x = x.to(device)
        task_id = task_id.to(device)
        y = y.to(device)

        if train:
            optim.zero_grad()

        logits = model(x, task_id)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        total_loss += loss.item() * len(y)
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, average="macro", zero_division=0)

    return total_loss / len(loader.dataset), acc, f1


if __name__ == "__main__":

    CSV_PATH = "L_action/datasets/L_subtasks_byTask_time_FULL_exactH5.csv"
    H5_ROOT = "dataset/landmarks"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_tr_base = QUBPHEOTransitionFusionDataset(CSV_PATH, H5_ROOT, split="train")
    ds_te_base = QUBPHEOTransitionFusionDataset(CSV_PATH, H5_ROOT, split="test")

    ds_tr = TransitionResampleTaskWrapper(ds_tr_base, L=64, csv_path=CSV_PATH)
    ds_te = TransitionResampleTaskWrapper(ds_te_base, L=64, csv_path=CSV_PATH)

    x0, t0, y0 = ds_tr[0]
    input_dim = x0.shape[1]

    print("Sanity:", x0.shape, "task_id:", t0.item(), "label:", y0.item())
    print("Input dim:", input_dim)
    print("Device:", device)

    num_tasks = len(ds_tr.task2idx)

    model = GRUClassifier(
        input_dim=input_dim,
        hidden=128,
        num_classes=36,
        num_tasks=num_tasks,
        task_emb_dim=16,
    ).to(device)

    weights = get_class_weights(ds_tr, 36).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(ds_tr, batch_size=64, shuffle=True)
    test_loader = DataLoader(ds_te, batch_size=64, shuffle=False)

    EPOCHS = 5

    best_test_f1 = -1.0
    best_test_acc = -1.0

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optim, device
        )

        te_loss, te_acc, te_f1 = run_epoch(
            model, test_loader, criterion, None, device
        )

        if te_f1 > best_test_f1:
            best_test_f1 = te_f1
            best_test_acc = te_acc

        print(
            f"Epoch {ep}: "
            f"train acc={tr_acc:.4f} f1={tr_f1:.4f} | "
            f"test acc={te_acc:.4f} f1={te_f1:.4f}"
        )

    protocol = f"""
Protocol: V2.4 Direct Transition Prediction (Segment(t) → Subtask(t+1))
Input: Fusion visual features (hand + gaze + objects, {input_dim}D) + Task Embedding
Temporal: Resample L=64
Backbone: GRU(128)
Loss: Weighted CE
Epochs: {EPOCHS}
Batch size: 64
Learning rate: 1e-3
Device: {device}
"""

    log_path = os.path.join(RESULT_DIR, "results_log.txt")
    metrics_path = os.path.join(RESULT_DIR, "metrics.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(protocol)
        f.write(f"\nFinal Test Accuracy: {te_acc:.6f}")
        f.write(f"\nFinal Test Macro-F1: {te_f1:.6f}")
        f.write(f"\nBest Test Accuracy: {best_test_acc:.6f}")
        f.write(f"\nBest Test Macro-F1: {best_test_f1:.6f}")
        f.write("\n" + "-" * 60 + "\n")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"final_test_accuracy: {te_acc:.6f}\n")
        f.write(f"final_test_macro_f1: {te_f1:.6f}\n")
        f.write(f"best_test_accuracy: {best_test_acc:.6f}\n")
        f.write(f"best_test_macro_f1: {best_test_f1:.6f}\n")
        f.write(f"input_dim: {input_dim}\n")
        f.write(f"epochs: {EPOCHS}\n")

    print("\nSaved to:", log_path)
    print("Metrics saved to:", metrics_path)