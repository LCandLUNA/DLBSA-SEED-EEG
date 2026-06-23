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

def get_session_files_by_subject(data_path):
    """
    Load processed_data, categorize files by subject_id
    returns a dict: {subject_id: [file1, file2, file3]} because for each subject(person), there are 3 sessions(files)
    """
    from collections import defaultdict
    
    subject_sessions = defaultdict(list)
    
    for file in sorted(os.listdir(data_path)):
        if not file.endswith(".npy"):
            continue
        data = np.load(os.path.join(data_path, file), allow_pickle=True).item()
        subj = str(data["subject_id"])
        sid = file.replace(".npy", "")
        subject_sessions[subj].append(sid)
    
    for subj in subject_sessions:
        subject_sessions[subj] = sorted(
            subject_sessions[subj],
            key=lambda x: int(x.replace("sample_", ""))
        )
    
    return dict(subject_sessions)
    
def subject_dependent_splits(data_path):
    subject_sessions = get_session_files_by_subject(data_path)
    
    splits = []
    for subj in sorted(subject_sessions.keys()):
        splits.append(([str(subj)], [str(subj)]))  # str而不是int
    
    return splits





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