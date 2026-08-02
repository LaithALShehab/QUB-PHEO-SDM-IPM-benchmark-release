import os
import json
import pandas as pd

VERSION = "V3.8.1_hard_history_chaining_from_sorted_SDM"
K_VALUES = [3, 4]

SDM_DIR = "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/V3.8.1_fusion_taskaware_merged34_GRU_E60_L64_sorted"
IN_PATH = os.path.join(SDM_DIR, "rich_predictions", "v381_sdm_rich_predictions_all_splits.csv")

OUT_DIR = os.path.join(
    "L_action/docs/results/V3_HDT/paper_benchmark_pipeline",
    "V3.8.2_IPM_hard_history_chaining"
)
os.makedirs(OUT_DIR, exist_ok=True)


def build_hard_history(df: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []

    # Critical: sort chronologically inside each participant-task sequence.
    df = df.sort_values(["participant", "task", "start_time", "end_time", "file"]).reset_index(drop=True)

    for (participant, task), g in df.groupby(["participant", "task"], sort=False):
        g = g.sort_values(["start_time", "end_time", "file"]).reset_index(drop=True)

        # Need k history items ending at current t, and a next item t+1 as target.
        for i in range(k - 1, len(g) - 1):
            current = g.iloc[i]
            nxt = g.iloc[i + 1]
            hist = g.iloc[i - k + 1:i + 1]

            row = {
                "split": current["split"],
                "participant": participant,
                "task": task,

                "current_file": current["file"],
                "current_start_time": float(current["start_time"]),
                "current_end_time": float(current["end_time"]),

                "next_file": nxt["file"],
                "next_start_time": float(nxt["start_time"]),
                "next_end_time": float(nxt["end_time"]),

                # IPM target = next true subtask
                "next_true_label_id": int(nxt["true_label_id"]),
                "next_true_label_name": str(nxt["true_label_name"]),

                # Current SDM status, useful for audit
                "current_true_label_id": int(current["true_label_id"]),
                "current_true_label_name": str(current["true_label_name"]),
                "current_pred_label_id": int(current["pred_label_id"]),
                "current_pred_label_name": str(current["pred_label_name"]),
                "current_sdm_correct": int(current["correct"]),
                "current_confidence": float(current["confidence"]),
                "current_entropy": float(current["entropy"]),

                "K": k,
                "history_definition": f"K={k}: previous {k-1} predicted SDM states + current predicted SDM state -> next true subtask",
            }

            # Hard predicted-state history
            for j, (_, h) in enumerate(hist.iterrows()):
                pos = j - (k - 1)  # e.g., K=3 gives -2,-1,0
                row[f"hist{pos}_file"] = h["file"]
                row[f"hist{pos}_start_time"] = float(h["start_time"])
                row[f"hist{pos}_true_label_id"] = int(h["true_label_id"])
                row[f"hist{pos}_true_label_name"] = str(h["true_label_name"])
                row[f"hist{pos}_pred_label_id"] = int(h["pred_label_id"])
                row[f"hist{pos}_pred_label_name"] = str(h["pred_label_name"])
                row[f"hist{pos}_confidence"] = float(h["confidence"])
                row[f"hist{pos}_entropy"] = float(h["entropy"])
                row[f"hist{pos}_correct"] = int(h["correct"])

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing input file: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    required = [
        "split", "file", "participant", "task", "start_time", "end_time",
        "true_label_id", "pred_label_id", "true_label_name", "pred_label_name",
        "correct", "confidence", "entropy"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Basic safety checks
    df["start_time"] = pd.to_numeric(df["start_time"])
    df["end_time"] = pd.to_numeric(df["end_time"])

    summary = {
        "version": VERSION,
        "input_path": IN_PATH,
        "output_dir": OUT_DIR,
        "input_rows": int(len(df)),
        "input_split_counts": {str(k): int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "K_values": K_VALUES,
        "sorting": "participant + task + start_time + end_time + file",
        "target": "next_true_label_id / next_true_label_name = subtask(t+1)",
        "history": "hard SDM predicted labels from previous/current segments",
        "cross_participant_or_cross_task_history": "not allowed; groupby participant, task",
    }

    for k in K_VALUES:
        out = build_hard_history(df, k)

        out_path = os.path.join(OUT_DIR, f"v381_hard_history_chaining_K{k}.csv")
        out.to_csv(out_path, index=False)

        summary[f"K{k}_rows"] = int(len(out))
        summary[f"K{k}_split_counts"] = {str(a): int(b) for a, b in out["split"].value_counts().to_dict().items()}
        summary[f"K{k}_output_path"] = out_path

        print(f"[OK] K={k}: {out.shape} -> {out_path}")
        print(out[[
            "split", "participant", "task",
            "current_start_time", "current_pred_label_name",
            "next_start_time", "next_true_label_name"
        ]].head(3).to_string(index=False))

    summary_path = os.path.join(OUT_DIR, "build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    readme_path = os.path.join(OUT_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("V3.8.2 IPM hard-history chaining dataset\n")
        f.write("Source: V3.8.1 sorted SDM rich predictions.\n")
        f.write("Purpose: build hard predicted-state histories for IPM.\n")
        f.write("K=3 means [t-2, t-1, t] predicted current-subtask states -> true subtask(t+1).\n")
        f.write("K=4 means [t-3, t-2, t-1, t] predicted current-subtask states -> true subtask(t+1).\n")
        f.write("Rows never cross participant or task boundaries.\n")
        f.write("Rows are sorted by participant, task, start_time, end_time, file.\n")

    print("\nSaved summary:", summary_path)
    print("Saved README:", readme_path)


if __name__ == "__main__":
    main()
