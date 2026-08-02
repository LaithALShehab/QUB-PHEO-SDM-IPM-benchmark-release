import os
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd


class QUBPHEOTransitionFusionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        h5_root,
        split="train",
        train_ids=range(1, 49),
        val_ids=range(49, 60),
        test_ids=range(60, 71),
    ):
        df_full = pd.read_csv(csv_path, sep="\t")
        df_full["pid"] = df_full["participant"].str.extract(r"p(\d+)").astype(int)

        split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        df_full = df_full[df_full["pid"].isin(split_map[split])]

        df_full = df_full.sort_values(
            by=["participant", "task", "start_time"]
        ).reset_index(drop=True)

        df_full["next_subtask"] = (
            df_full.groupby(["participant", "task"])["subtask"].shift(-1)
        )

        df_full = df_full.dropna(subset=["next_subtask"]).reset_index(drop=True)

        self.df = df_full
        self.h5_root = h5_root

        self.subtasks = sorted(pd.read_csv(csv_path, sep="\t")["subtask"].unique())
        self.subtask2idx = {s: i for i, s in enumerate(self.subtasks)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        h5_name = row["filename"].replace(".mp4", ".h5")
        h5_path = os.path.join(self.h5_root, row["subtask"], h5_name)

        assert os.path.exists(h5_path), f"Missing H5 file: {h5_path}"

        with h5py.File(h5_path, "r") as f:
            left = f["left_landmarks"][..., :2]
            right = f["right_landmarks"][..., :2]

            hand = torch.cat(
                [torch.tensor(left), torch.tensor(right)], dim=1
            ).reshape(len(left), -1).float()

            if "norm_gaze" in f:
                gaze = torch.tensor(f["norm_gaze"][:]).float()
            else:
                gaze = torch.zeros((hand.shape[0], 2)).float()

            if gaze.ndim == 1:
                gaze = gaze.unsqueeze(-1)

            boxes = torch.tensor(f["rec_bboxes"][:]).float()
            boxes = boxes.view(boxes.shape[0], -1)

        T = min(hand.shape[0], gaze.shape[0], boxes.shape[0])
        fusion = torch.cat([hand[:T], gaze[:T], boxes[:T]], dim=1).float()

        label = torch.tensor(self.subtask2idx[row["next_subtask"]]).long()

        return fusion, label