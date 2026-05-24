# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BiosignalDataset(Dataset):
    def __init__(self, config, subject_ids=None):
        """
        If subject_ids is None → load ALL data (used for K-Fold)
        Otherwise → filter by subject_ids (used for LOSO / LMSO)
        """

        self.data = []

        data_path = config["paths"]["processed_data"]

        for file in os.listdir(data_path):
            if not file.endswith(".npy"):
                continue

            file_path = os.path.join(data_path, file)

            try:
                sample = np.load(file_path, allow_pickle=True).item()
            except Exception:
                continue

            subject_id = sample["subject_id"]

            # Filter if subject_ids provided
            if subject_ids is not None and subject_id not in subject_ids:
                continue

            signals = sample["signals"]
            labels = sample["labels"]

            for i in range(len(signals)):
                self.data.append({
                    "signal": signals[i],
                    "label": labels[i],
                    "subject_id": subject_id
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        x = torch.tensor(item["signal"], dtype=torch.float32)
        y = torch.tensor(item["label"], dtype=torch.long)

        return {
            "signal": x,   # (C, T)
            "label": y,
            "subject_id": item["subject_id"]
        }

    def get_all_subject_ids(self):
        return [item["subject_id"] for item in self.data]