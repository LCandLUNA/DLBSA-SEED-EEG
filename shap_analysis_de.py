# shap_analysis.py
# SHAP-based interpretability analysis for the 2D CNN model trained on DE features
# (subject-dependent split). Loads pre-trained fold checkpoints (no re-training),
# computes SHAP values, and aggregates them into a (62 channels x 5 bands)
# importance map to reveal which brain regions / frequency bands drive predictions.

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

from config import get_config
from model import get_model
from dataset import BiosignalDataset
from utils import subject_dependent_splits

# The 5 DE frequency bands, in the fixed order used during preprocessing
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
N_CHANNELS = 62
N_BANDS = 5


def compute_shap_for_fold(cfg, fold, device="cpu", n_background=100, n_explain=50):
    """
    Compute SHAP values for one subject-dependent fold and return a (62, 5)
    importance map (mean absolute SHAP value per channel-band pair).

    cfg      : config dict (mode='de', model type='cnn')
    fold     : which fold / subject checkpoint to load
    Returns  : np.ndarray of shape (62, 5), or None if this fold fails
    """
    # --- 1. Load the trained model for this fold (no re-training) ---
    mode = cfg["mode"]
    mtype = cfg["model"]["type"]
    protocol = "subject_dependent"   # selection: subject-dependent, loso
    ckpt_path = f'{cfg["paths"]["checkpoints"]}{mode}_{mtype}_{protocol}_fold{fold}.pt'
    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    # --- 2. Rebuild the same data this fold was trained on ---
    splits = subject_dependent_splits(cfg["paths"][cfg["mode"]])
    train_sids, _ = splits[fold]
    dataset = BiosignalDataset(cfg, train_sids)

    # Not enough samples in this subject -> skip
    if len(dataset) < n_background + 10:
        print(f"  fold {fold}: too few samples ({len(dataset)}), skipping")
        return None

    # Stack all samples into one tensor: (N, 1, 62, 5)
    all_x = torch.stack([dataset[i]["signal"] for i in range(len(dataset))])

    # background: reference samples SHAP uses to estimate the expected output
    # explain_x : the samples we actually explain
    background = all_x[:n_background].to(device)
    end = min(n_background + n_explain, len(all_x))
    explain_x = all_x[n_background:end].to(device)

    # --- 3. Run SHAP. DeepExplainer suits neural nets; fall back to Gradient ---
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(explain_x)
    except Exception as e:
        print(f"  fold {fold}: DeepExplainer failed ({e}), using GradientExplainer")
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain_x)

    # --- 4. Normalize SHAP output into a single (62, 5) importance map ---
    # SHAP versions differ: shap_values may be a list (one array per class) OR
    # a single ndarray with the class axis at the end. We detect and handle both,
    # then reduce every axis except the (62, 5) channel-band axes.
    importance = _aggregate_to_channel_band(shap_values, fold)
    return importance


def _aggregate_to_channel_band(shap_values, fold):
    """
    Reduce arbitrary SHAP output into a (62, 5) mean-absolute-importance map.
    Handles both old (list-per-class) and new (class axis last) SHAP formats
    by locating the two axes of size 62 and 5 and averaging over all others.
    """
    # Convert list-of-arrays (one per class) into a single stacked array
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(v) for v in shap_values], axis=0)
    else:
        arr = np.asarray(shap_values)

    arr = np.abs(arr)  # magnitude of contribution, direction is not needed here

    # Debug print: check the real shape so aggregation is correct
    print(f"  fold {fold}: raw SHAP array shape = {arr.shape}")

    # Find the axis that equals 62 (channels) and the axis that equals 5 (bands)
    ch_axis = None
    band_axis = None
    for ax, size in enumerate(arr.shape):
        if size == N_CHANNELS and ch_axis is None:
            ch_axis = ax
        elif size == N_BANDS and band_axis is None:
            band_axis = ax

    if ch_axis is None or band_axis is None:
        raise ValueError(
            f"Could not locate (62, 5) axes in SHAP shape {arr.shape}. "
            "Inspect the array manually."
        )

    # Average over every axis except the channel and band axes
    other_axes = tuple(ax for ax in range(arr.ndim) if ax not in (ch_axis, band_axis))
    importance = arr.mean(axis=other_axes)  # -> shape (62, 5) or (5, 62)

    # Ensure orientation is (62, 5): channels as rows, bands as columns
    if importance.shape == (N_BANDS, N_CHANNELS):
        importance = importance.T
    return importance  # (62, 5)


def run_shap(folds_to_use=range(15)):
    cfg = get_config()
    cfg["mode"] = "de"
    cfg["model"]["type"] = "cnn"
    device = "cpu"  # CPU is simplest and reliable for SHAP

    # --- Compute per-fold importance and average across folds for robustness ---
    all_importance = []
    for f in folds_to_use:
        print(f"Processing fold {f}...")
        try:
            imp = compute_shap_for_fold(cfg, f, device=device)
            if imp is not None:
                all_importance.append(imp)
        except Exception as e:
            print(f"  fold {f} failed: {e}")

    if not all_importance:
        print("No folds produced SHAP values. Aborting.")
        return

    # Mean importance across folds -> a robust cross-subject importance map
    importance = np.mean(all_importance, axis=0)  # (62, 5)

    tag = f'{cfg["mode"]}_{cfg["model"]["type"]}' 
    # --- Heatmap: 62 channels x 5 bands ---
    plt.figure(figsize=(6, 12))
    plt.imshow(importance, aspect="auto", cmap="viridis")
    plt.colorbar(label="mean |SHAP value|")
    plt.xticks(range(N_BANDS), BANDS)
    plt.xlabel("Frequency Band")
    plt.ylabel("EEG Channel (0-61)")
    plt.title(f"SHAP Importance ({tag}, subject-dependent)")
    plt.tight_layout()
    out_heatmap = f'{cfg["paths"]["plots"]}shap_importance_{tag}.png'
    plt.savefig(out_heatmap, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved heatmap to: {out_heatmap}")

    # --- Bar chart ---
    band_imp = importance.mean(axis=0)
    plt.figure(figsize=(6, 4))
    plt.bar(BANDS, band_imp, color="steelblue")
    plt.ylabel("mean |SHAP value|")
    plt.title(f"Frequency-band importance ({tag})")
    plt.tight_layout()
    out_bar = f'{cfg["paths"]["plots"]}shap_band_importance_{tag}.png'
    plt.savefig(out_bar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved band bar chart to: {out_bar}")

    # --- Print numeric summary ---
    print("\nMean importance per frequency band:")
    for b, v in zip(BANDS, band_imp):
        print(f"  {b}: {v:.4f}")

    # Top-10 most important channel-band pairs
    flat = importance.flatten()
    top_idx = np.argsort(flat)[::-1][:10]
    print("\nTop-10 channel-band pairs:")
    for idx in top_idx:
        ch, band = idx // N_BANDS, idx % N_BANDS
        print(f"  channel {ch:2d}, {BANDS[band]:5s}: {flat[idx]:.4f}")


if __name__ == "__main__":
    # Start with 5 folds; change to range(15) once verified for full robustness
    run_shap(folds_to_use=range(5))
