import os
import json
import pandas as pd

IN_PATH = "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/V3.8.1_fusion_taskaware_merged34_GRU_E60_L64_sorted/rich_predictions/v381_sdm_rich_predictions_all_splits.csv"

OUT_DIR = "L_action/docs/results/V3_HDT/paper_benchmark_pipeline/V3.8.5_IPM_soft_history_chaining"
os.makedirs(OUT_DIR, exist_ok=True)

PROB_PREFIX = "prob_"


def build_soft_history(df, K):
    prob_cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    rows = []

    df = df.sort_values(["split", "participant", "task", "start_time"]).reset_index(drop=True)

    for (split, participant, task), g in df.groupby(["split", "participant", "task"], sort=False):
        g = g.sort_values("start_time").reset_index(drop=True)

        # Need K history states and one next target
        for i in range(K - 1, len(g) - 1):
            hist = g.iloc[i - K + 1:i + 1]
            current = g.iloc[i]
            nxt = g.iloc[i + 1]

            row = {
                "split": split,
                "participant": participant,
                "task": task,

                "current_file": current["file"],
                "current_start_time": float(current["start_time"]),
                "current_end_time": float(current["end_time"]),
                "current_true_label_id": int(current["true_label_id"]),
                "current_pred_label_id": int(current["pred_label_id"]),
                "current_true_label_name": current["true_label_name"],
                "current_pred_label_name": current["pred_label_name"],
                "current_confidence": float(current["confidence"]),
                "current_entropy": float(current["entropy"]),

                "next_file": nxt["file"],
                "next_start_time": float(nxt["start_time"]),
                "next_end_time": float(nxt["end_time"]),
                "next_true_label_id": int(nxt["true_label_id"]),
                "next_true_label_name": nxt["true_label_name"],
                "K": int(K),
            }

            # hist0 = current, hist-1 = previous, etc.
            for offset in range(K):
                h = hist.iloc[offset]
                rel = offset - (K - 1)  # e.g. K=4 -> -3,-2,-1,0
                prefix = f"hist{rel}"

                row[f"{prefix}_file"] = h["file"]
                row[f"{prefix}_start_time"] = float(h["start_time"])
                row[f"{prefix}_true_label_id"] = int(h["true_label_id"])
                row[f"{prefix}_pred_label_id"] = int(h["pred_label_id"])
                row[f"{prefix}_true_label_name"] = h["true_label_name"]
                row[f"{prefix}_pred_label_name"] = h["pred_label_name"]
                row[f"{prefix}_confidence"] = float(h["confidence"])
                row[f"{prefix}_entropy"] = float(h["entropy"])

                for pc in prob_cols:
                    row[f"{prefix}_{pc}"] = float(h[pc])

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(IN_PATH)
    prob_cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]

    summary = {
        "source": IN_PATH,
        "output_dir": OUT_DIR,
        "probability_columns": len(prob_cols),
        "description": "Soft-history IPM chaining CSVs from V3.8.1 SDM rich predictions. Each row uses K predicted SDM probability vectors plus confidence/entropy to predict the next true merged-34 subtask.",
        "versions": {}
    }

    for K in [3, 4]:
        out = build_soft_history(df, K)
        out_path = os.path.join(OUT_DIR, f"v381_soft_history_chaining_K{K}.csv")
        out.to_csv(out_path, index=False)

        summary["versions"][f"K{K}"] = {
            "path": out_path,
            "shape": list(out.shape),
            "split_counts": out["split"].value_counts().to_dict(),
        }

        print(f"[OK] K={K}: {out.shape} -> {out_path}")
        print(out[["split", "participant", "task", "current_start_time", "current_pred_label_name", "next_start_time", "next_true_label_name"]].head(3).to_string(index=False))

    with open(os.path.join(OUT_DIR, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write("V3.8.5 IPM soft-history chaining export\n")
        f.write("Source: V3.8.1 SDM rich predictions\n")
        f.write("Purpose: prepare K=3 and K=4 soft SDM history inputs for IPM training.\n")
        f.write("Soft input = SDM probability vectors + confidence + entropy across history steps.\n")
        f.write("Target = next true merged-34 subtask label.\n")

    print("\nSaved summary and README to:", OUT_DIR)


if __name__ == "__main__":
    main()
