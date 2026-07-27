# evaluate_metrics.py
# Standalone evaluation: loads existing checkpoints (NO re-training),
# runs inference on the test set of each fold, and computes:
#  - Marco F1 score
#  - Confusion Matrix(reveals which emotions get confused more)
# Aggregates predictions across all folds for an overall result

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from config import get_config
from model import get_model
from dataset import BiosignalDataset
from utils import get_subject_ids, loso_split, subject_dependent_splits, check_no_leakage

CLASS_NAMES = ["negative", "neutral", "positive"]   # labels 0, 1, 2


def get_test_dataset(cfg, split, protocol):
    """
    Rebuild the SAME test set used during training for one fold.
    - loso: test set is the held-out subject's data
    - subject_dependent: 80/20 split within the subject (seeded, matches training)
    """
    train_sids, test_sids = split

    if protocol == "loso":
        check_no_leakage(train_sids, test_sids)
        # Normalization stats come from training subjects (as in training.py)
        train_ds_raw = BiosignalDataset(cfg, train_sids)
        mean, std = train_ds_raw.compute_stats()
        test_dataset = BiosignalDataset(cfg, test_sids, mean=mean, std=std)

    else:  # subject_dependent
        full_dataset = BiosignalDataset(cfg, train_sids)
        n = len(full_dataset)
        train_size = int(0.8 * n)
        test_size = n - train_size
        # Same seed(42) as training so we evaluate on the same held-out 20%
        _, test_dataset = random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    return test_dataset


def evaluate_fold(cfg, fold, split, protocol, device):
    """Load one fold's checkpoint, run inference, return (labels, preds)."""
    mode = cfg["mode"]
    mtype = cfg["model"]["type"]
    ckpt = f'{cfg["paths"]["checkpoints"]}{mode}_{mtype}_{protocol}_fold{fold}.pt'
    if not os.path.exists(ckpt):
        print(f"  fold {fold}: checkpoint not found, skipping ({ckpt})")
        return None, None

    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    test_dataset = get_test_dataset(cfg, split, protocol)
    loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"],
                        shuffle=False)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"]
            out = model(x)                       # (B, 3)
            pred = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(y.numpy())

    return np.array(all_labels), np.array(all_preds)


def run(protocol="loso"):
    cfg = get_config()
    # --- set which model/data to evaluate here ---
    cfg["mode"] = "raw"              # "de" or "raw"
    cfg["model"]["type"] = "cnn1d"  # "cnn", "cnn1d", "dgcnn", "cnn_lstm", ...
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the same splits as training
    if protocol == "loso":
        ids = get_subject_ids(cfg["paths"][cfg["mode"]])
        splits = loso_split(ids)
    else:
        splits = subject_dependent_splits(cfg["paths"][cfg["mode"]])

    all_labels, all_preds = [], []
    for fold, split in enumerate(splits):
        print(f"Evaluating fold {fold}...")
        labels, preds = evaluate_fold(cfg, fold, split, protocol, device)
        if labels is not None:
            all_labels.extend(labels)
            all_preds.extend(preds)

    if not all_labels:
        print("No predictions produced. Check checkpoints exist for this config.")
        return

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # --- Macro F1 ---
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    per_class_f1 = f1_score(all_labels, all_preds, average=None)

    print("\n" + "=" * 50)
    print(f"Model: {cfg['mode']}_{cfg['model']['type']}  |  Protocol: {protocol}")
    print("=" * 50)
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("Per-class F1:")
    for name, f in zip(CLASS_NAMES, per_class_f1):
        print(f"  {name:8s}: {f:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalized

    tag = f'{cfg["mode"]}_{cfg["model"]["type"]}_{protocol}'
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({tag})")
    # annotate each cell with count + percentage
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, label="Row-normalized")
    plt.tight_layout()
    out = f'{cfg["paths"]["plots"]}confusion_matrix_{tag}.png'
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved confusion matrix to: {out}")
  


if __name__ == "__main__":
    run(protocol="loso")   # or "subject_dependent"