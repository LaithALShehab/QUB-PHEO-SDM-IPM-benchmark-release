import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

VERSION = "V3.8.3_IPM_hard_history_GRU_K4_E60"
K = 4

CSV_PATH = (
    "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/"
    "V3.8.2_IPM_hard_history_chaining/v381_hard_history_chaining_K4.csv"
)

RESULT_DIR = (
    "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/"
    "V3.8.3_IPM_hard_history_GRU_K4_E60"
)
WEIGHTS_DIR = os.path.join(RESULT_DIR, "weights")

EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 34
STATE_EMB_DIM = 64
TASK_EMB_DIM = 16
HIDDEN_DIM = 128


class HardHistoryIPMDataset(Dataset):
    def __init__(self, csv_path, split):
        self.df_all = pd.read_csv(csv_path)
        self.df = self.df_all[self.df_all["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split}")

        self.tasks = sorted(self.df_all["task"].unique())
        self.task2idx = {t: i for i, t in enumerate(self.tasks)}

        # For K=4: hist-3, hist-2, hist-1, hist0
        self.hist_cols = [f"hist-{i}_pred_label_id" for i in range(K - 1, 0, -1)]
        self.hist_cols.append("hist0_pred_label_id")

        for c in self.hist_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing history column: {c}")

        if "next_true_label_id" not in self.df.columns:
            raise ValueError("Missing target column: next_true_label_id")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        hist = torch.tensor([int(row[c]) for c in self.hist_cols], dtype=torch.long)
        task_id = torch.tensor(self.task2idx[row["task"]], dtype=torch.long)
        y = torch.tensor(int(row["next_true_label_id"]), dtype=torch.long)

        return hist, task_id, y


class HardHistoryGRUIPM(nn.Module):
    def __init__(self, num_classes=34, num_tasks=9):
        super().__init__()

        self.state_emb = nn.Embedding(num_classes, STATE_EMB_DIM)
        self.task_emb = nn.Embedding(num_tasks, TASK_EMB_DIM)

        self.gru = nn.GRU(
            input_size=STATE_EMB_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(HIDDEN_DIM + TASK_EMB_DIM, num_classes)

    def forward(self, hist, task_id):
        x = self.state_emb(hist)   # B, K, state_emb_dim
        _, h = self.gru(x)         # h: 1, B, hidden
        h = h.squeeze(0)

        t = self.task_emb(task_id)
        z = torch.cat([h, t], dim=1)
        z = self.dropout(z)

        return self.fc(z)


def get_class_weights(ds, num_classes=34):
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, _, y in ds:
        counts[int(y)] += 1

    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    ys, preds = [], []

    with torch.set_grad_enabled(is_train):
        for hist, task_id, y in loader:
            hist = hist.to(device)
            task_id = task_id.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(hist, task_id)
            loss = criterion(logits, y)

            if is_train:
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
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = HardHistoryIPMDataset(CSV_PATH, "train")
    val_ds = HardHistoryIPMDataset(CSV_PATH, "val")
    test_ds = HardHistoryIPMDataset(CSV_PATH, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_tasks = len(train_ds.tasks)

    model = HardHistoryGRUIPM(
        num_classes=NUM_CLASSES,
        num_tasks=num_tasks,
    ).to(device)

    weights = get_class_weights(train_ds, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(WEIGHTS_DIR, "best_model.pt")
    last_model_path = os.path.join(WEIGHTS_DIR, "last_model.pt")

    log_rows = []

    print("Version:", VERSION)
    print("Device:", device)
    print("CSV:", CSV_PATH)
    print("Train/Val/Test:", len(train_ds), len(val_ds), len(test_ds))
    print("Num tasks:", num_tasks)
    print("K:", K)
    print("History columns:", train_ds.hist_cols)

    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device)
        va = run_epoch(model, val_loader, criterion, None, device)

        log_rows.append({
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
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {ep}: "
            f"train acc={tr['accuracy']:.4f} f1={tr['macro_f1']:.4f} | "
            f"val acc={va['accuracy']:.4f} f1={va['macro_f1']:.4f} "
            f"precision={va['macro_precision']:.4f} recall={va['macro_recall']:.4f}"
        )

    torch.save(model.state_dict(), last_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    te = run_epoch(model, test_loader, criterion, None, device)

    pd.DataFrame(log_rows).to_csv(
        os.path.join(RESULT_DIR, "training_log.csv"),
        index=False,
    )

    metrics = {
        "version": VERSION,
        "stage": "IPM hard-history chaining from V3.8.1 SDM predictions",
        "input": "K=4 hard SDM predicted current-subtask history + task embedding",
        "target": "next true subtask label",
        "label_space": "merged-34 next-subtask labels",
        "backbone": "GRU",
        "K": K,
        "history_columns": train_ds.hist_cols,
        "state_embedding_dim": STATE_EMB_DIM,
        "task_embedding_dim": TASK_EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "num_tasks": num_tasks,
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
    }

    with open(os.path.join(RESULT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(RESULT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    model_info = f"""Model: HardHistoryGRUIPM
Version: {VERSION}
Task: IPM next-subtask prediction from hard SDM history
Input: K={K} hard predicted SDM states + task embedding
History definition: [t-3, t-2, t-1, t] predicted current-subtask states -> true subtask(t+1)
History columns: {train_ds.hist_cols}
State embedding: nn.Embedding(num_classes={NUM_CLASSES}, dim={STATE_EMB_DIM})
Task embedding: nn.Embedding(num_tasks={num_tasks}, dim={TASK_EMB_DIM})
Backbone: nn.GRU(input_size={STATE_EMB_DIM}, hidden_size={HIDDEN_DIM}, batch_first=True)
Classifier: Linear(hidden={HIDDEN_DIM} + task_emb={TASK_EMB_DIM} -> classes={NUM_CLASSES})
Loss: weighted cross entropy
Optimizer: AdamW, lr={LR}, weight_decay={WEIGHT_DECAY}
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

    labels = list(range(NUM_CLASSES))

    cm = confusion_matrix(te["y_true"], te["y_pred"], labels=labels)
    pd.DataFrame(cm).to_csv(
        os.path.join(RESULT_DIR, "confusion_matrix.csv"),
        index=False,
    )

    report = classification_report(
        te["y_true"],
        te["y_pred"],
        labels=labels,
        zero_division=0,
    )
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


if __name__ == "__main__":
    main()