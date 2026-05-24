# utils.py

import os
import numpy as np
from sklearn.model_selection import KFold


# -------------------------
# Subject utilities
# -------------------------

def get_subject_ids(data_path):
    subject_ids = set()

    for file in os.listdir(data_path):
        if not file.endswith(".npy"):
            continue

        try:
            data = np.load(os.path.join(data_path, file), allow_pickle=True).item()
            subject_ids.add(data["subject_id"])
        except Exception:
            continue

    return sorted(list(subject_ids))


# -------------------------
# Splitting strategies
# -------------------------

def loso_split(subject_ids):
    """
    Leave-One-Subject-Out
    """
    splits = []

    for test_sid in subject_ids:
        train_sids = [sid for sid in subject_ids if sid != test_sid]
        splits.append((train_sids, [test_sid]))

    return splits


def lmso_split(subject_ids, k=5):
    """
    Leave-Multiple-Subjects-Out
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = []

    for train_idx, test_idx in kf.split(subject_ids):
        train_sids = [subject_ids[i] for i in train_idx]
        test_sids = [subject_ids[i] for i in test_idx]

        splits.append((train_sids, test_sids))

    return splits


def kfold_split_indices(n_samples, k=5):
    """
    Conventional K-Fold (sample-level)
    WARNING: may cause subject leakage
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    splits = []
    indices = np.arange(n_samples)

    for train_idx, test_idx in kf.split(indices):
        splits.append((train_idx, test_idx))

    return splits


# -------------------------
# Misc
# -------------------------

def create_folders(config):
    for path in config["paths"].values():
        os.makedirs(path, exist_ok=True)


def check_no_leakage(train_sids, test_sids):
    overlap = set(train_sids).intersection(set(test_sids))
    assert len(overlap) == 0, f"Data leakage detected: {overlap}"