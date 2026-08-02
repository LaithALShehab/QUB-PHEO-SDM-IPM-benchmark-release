import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



VERSION = "V3.8.10d_IPM_soft_history_MLP_K4_E100_dropout03_seed42"

CSV_PATH = "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/V3.8.5_IPM_soft_history_chaining/v381_soft_history_chaining_K4.csv"
RESULT_DIR = "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/V3.8.10d_IPM_soft_history_MLP_K4_E100_dropout03_seed42"
WEIGHTS_DIR = os.path.join(RESULT_DIR, "weights")

K = 4
HIST_PREFIXES = ["hist-3", "hist-2", "hist-1", "hist0"]

TASK_EMB_DIM = 16
HIDDEN_LAYERS = [256, 128]
DROPOUT = 0.3
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_CLASSES = 34


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SoftHistoryMLPDataset(Dataset):
    def __init__(self, csv_path, split):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        task_values = sorted(pd.read_csv(csv_path)["task"].unique())
        self.task2idx = {t: i for i, t in enumerate(task_values)}
        self.num_tasks = len(self.task2idx)

        # Use hist0 to define the 34 probability suffixes, then reuse same suffixes for all history steps.
        hist0_prob_cols = sorted([c for c in self.df.columns if c.startswith("hist0_prob_")])
        if len(hist0_prob_cols) != 34:
            raise ValueError(f"Expected 34 hist0 probability columns, found {len(hist0_prob_cols)}")

        self.prob_suffixes = [c.replace("hist0_", "") for c in hist0_prob_cols]

        # Each step = 34 probabilities + confidence + entropy = 36
        self.soft_step_dim = len(self.prob_suffixes) + 2
        self.input_dim = K * self.soft_step_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        features = []
        for hp in HIST_PREFIXES:
            probs = [float(row[f"{hp}_{suffix}"]) for suffix in self.prob_suffixes]
            conf = float(row[f"{hp}_confidence"])
            ent = float(row[f"{hp}_entropy"])
            features.extend(probs + [conf, ent])

        x = torch.tensor(features, dtype=torch.float32)
        task_id = torch.tensor(self.task2idx[row["task"]], dtype=torch.long)
        y = torch.tensor(int(row["next_true_label_id"]), dtype=torch.long)

        return x, task_id, y


class SoftHistoryMLP(nn.Module):
    def __init__(self, input_dim, num_tasks, task_emb_dim=16, hidden_layers=(256, 128), dropout=0.2, num_classes=34):
        super().__init__()

        self.task_emb = nn.Embedding(num_tasks, task_emb_dim)

        layers = []
        in_dim = input_dim + task_emb_dim

        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, task_id):
        t = self.task_emb(task_id)
        z = torch.cat([x, t], dim=1)
        return self.net(z)


def class_weights(dataset, num_classes=34):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, _, y in dataset:
        counts[int(y)] += 1
    counts = np.maximum(counts, 1)
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    ys, preds = [], []

    with torch.set_grad_enabled(train):
        for x, task_id, y in loader:
            x = x.to(device)
            task_id = task_id.to(device)
            y = y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(x, task_id)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            ys.extend(y.detach().cpu().numpy().tolist())
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(ys, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(ys, preds, average="macro", zero_division=0),
        "y_true": ys,
        "y_pred": preds,
    }


def main():
    set_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SoftHistoryMLPDataset(CSV_PATH, "train")
    val_ds = SoftHistoryMLPDataset(CSV_PATH, "val")
    test_ds = SoftHistoryMLPDataset(CSV_PATH, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SoftHistoryMLP(
        input_dim=train_ds.input_dim,
        num_tasks=train_ds.num_tasks,
        task_emb_dim=TASK_EMB_DIM,
        hidden_layers=HIDDEN_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    ).to(device)

    weights = class_weights(train_ds, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("Version:", VERSION)
    print("CSV:", CSV_PATH)
    print("Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds))
    print("Input dim:", train_ds.input_dim)
    print("Soft step dim:", train_ds.soft_step_dim)
    print("Device:", device)

    best_val_f1 = -1.0
    best_epoch = -1
    best_path = os.path.join(WEIGHTS_DIR, "best_model.pt")
    last_path = os.path.join(WEIGHTS_DIR, "last_model.pt")

    rows = []

    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device)
        va = run_epoch(model, val_loader, criterion, None, device)

        rows.append({
            "epoch": ep,
            "train_loss": tr["loss"],
            "train_accuracy": tr["accuracy"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": va["loss"],
            "val_accuracy": va["accuracy"],
            "val_macro_f1": va["macro_f1"],
            "val_macro_precision": va["macro_precision"],
            "val_macro_recall": va["macro_recall"],
        })

        if va["macro_f1"] > best_val_f1:
            best_val_f1 = va["macro_f1"]
            best_epoch = ep
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {ep}: "
            f"train acc={tr['accuracy']:.4f} f1={tr['macro_f1']:.4f} | "
            f"val acc={va['accuracy']:.4f} f1={va['macro_f1']:.4f} "
            f"precision={va['macro_precision']:.4f} recall={va['macro_recall']:.4f}"
        )

    torch.save(model.state_dict(), last_path)
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "training_log.csv"), index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    te = run_epoch(model, test_loader, criterion, None, device)

    metrics = {
        "version": VERSION,
        "stage": "IPM soft-history chaining from V3.8.1 SDM predictions",
        "input": "K=4 SDM probability vectors + confidence + entropy + task embedding",
        "target": "next true subtask label",
        "label_space": "merged-34 next-subtask labels",
        "backbone": "MLP",
        "K": K,
        "soft_step_dim": train_ds.soft_step_dim,
        "flattened_input_dim": train_ds.input_dim,
        "task_embedding_dim": TASK_EMB_DIM,
        "hidden_layers": HIDDEN_LAYERS,
        "dropout": DROPOUT,
        "num_classes": NUM_CLASSES,
        "num_tasks": train_ds.num_tasks,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": "Weighted Cross Entropy",
        "best_epoch_by_val_macro_f1": best_epoch,
        "test_accuracy": te["accuracy"],
        "test_macro_f1": te["macro_f1"],
        "test_macro_precision": te["macro_precision"],
        "test_macro_recall": te["macro_recall"],
        "csv_path": CSV_PATH,
    }

    with open(os.path.join(RESULT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(RESULT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    model_info = f"""Model: SoftHistoryMLP
Version: {VERSION}
Task: IPM next-subtask prediction
Input: K={K} soft SDM history, each step = 34 probabilities + confidence + entropy
Flattened input dim: {train_ds.input_dim}
Task context: embedding, dim={TASK_EMB_DIM}
Backbone: MLP hidden layers {HIDDEN_LAYERS}, dropout={DROPOUT}
Target: next true merged-34 subtask label
Loss: weighted cross entropy
Optimizer: AdamW, lr={LR}, weight_decay={WEIGHT_DECAY}
Selection: best model by validation Macro-F1
Evaluation: participant-held-out test split
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

    labels = list(range(NUM_CLASSES))
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
    print("Best model:", best_path)
    print("TEST acc:", round(te["accuracy"], 6))
    print("TEST macro_f1:", round(te["macro_f1"], 6))
    print("TEST precision:", round(te["macro_precision"], 6))
    print("TEST recall:", round(te["macro_recall"], 6))


if __name__ == "__main__":
    main()
