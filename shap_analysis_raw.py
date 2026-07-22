# shap_analysis_raw.py
# SHAP interpretability for RAW-signal models (e.g. 1D/2D CNN on raw EEG).
# Raw input is (62 channels x 800 time-samples): there is NO frequency-band axis,
# so band-level analysis does not apply. Instead we produce:
#   (1) channel-level importance  -> which electrodes matter
#   (2) temporal importance curve -> which time points within the 4s window matter
# Loads pre-trained subject-dependent checkpoints (no re-training).

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

from config import get_config
from model import get_model
from dataset import BiosignalDataset
from utils import subject_dependent_splits

N_CHANNELS = 62
WINDOW_SIZE = 800   # raw time samples per 4s window (200Hz * 4s)


def compute_shap_for_fold(cfg, fold, device="cpu", n_background=50, n_explain=30):
    """
    Compute SHAP values for one subject-dependent fold on a RAW model.
    Returns a (62, 800) importance map (mean |SHAP| per channel-timepoint),
    or None if this fold fails.

    Note: raw SHAP is heavier than DE (800 time-points vs 5 bands), so we use
    smaller background/explain sets to keep it tractable.
    """
    mode = cfg["mode"]
    mtype = cfg["model"]["type"]
    protocol = "subject_dependent"

    # --- 1. Load the trained model for this fold ---
    ckpt_path = f'{cfg["paths"]["checkpoints"]}{mode}_{mtype}_{protocol}_fold{fold}.pt'
    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    # --- 2. Rebuild the same data this fold was trained on ---
    splits = subject_dependent_splits(cfg["paths"][cfg["mode"]])
    train_sids, _ = splits[fold]
    dataset = BiosignalDataset(cfg, train_sids)

    if len(dataset) < n_background + 10:
        print(f"  fold {fold}: too few samples ({len(dataset)}), skipping")
        return None

    # Raw samples: dataset returns (1, 62, 800) each -> stack to (N, 1, 62, 800)
    all_x = torch.stack([dataset[i]["signal"] for i in range(len(dataset))])
    background = all_x[:n_background].to(device)
    end = min(n_background + n_explain, len(all_x))
    explain_x = all_x[n_background:end].to(device)

    # --- 3. Run SHAP (GradientExplainer is more robust for raw/deep nets) ---
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain_x)
    except Exception as e:
        print(f"  fold {fold}: GradientExplainer failed ({e}), trying DeepExplainer")
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(explain_x)

    # --- 4. Reduce to a (62, 800) channel-time importance map ---
    importance = _aggregate_to_channel_time(shap_values, fold)
    return importance


def _aggregate_to_channel_time(shap_values, fold):
    """
    Reduce arbitrary SHAP output to a (62, 800) mean-absolute-importance map.
    Locates the axes of size 62 (channels) and 800 (time) and averages over
    all other axes (classes, samples, the dummy conv channel).
    """
    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(v) for v in shap_values], axis=0)
    else:
        arr = np.asarray(shap_values)

    arr = np.abs(arr)
    print(f"  fold {fold}: raw SHAP array shape = {arr.shape}")

    ch_axis, time_axis = None, None
    for ax, size in enumerate(arr.shape):
        if size == N_CHANNELS and ch_axis is None:
            ch_axis = ax
        elif size == WINDOW_SIZE and time_axis is None:
            time_axis = ax

    if ch_axis is None or time_axis is None:
        raise ValueError(
            f"Could not locate (62, 800) axes in SHAP shape {arr.shape}."
        )

    other_axes = tuple(ax for ax in range(arr.ndim) if ax not in (ch_axis, time_axis))
    importance = arr.mean(axis=other_axes)   # -> (62, 800) or (800, 62)

    if importance.shape == (WINDOW_SIZE, N_CHANNELS):
        importance = importance.T
    return importance   # (62, 800)


def run_shap(folds_to_use=(0, 1, 2, 3, 4)):
    cfg = get_config()
    cfg["mode"] = "raw"              # raw data
    cfg["model"]["type"] = "cnn1d"   # or "cnn" for 2D raw CNN
    device = "cpu"

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

    importance = np.mean(all_importance, axis=0)   # (62, 800)
    tag = f'{cfg["mode"]}_{cfg["model"]["type"]}'

    # --- (1) Channel-level importance: average over time -> (62,) ---
    channel_imp = importance.mean(axis=1)   # (62,)
    plt.figure(figsize=(10, 4))
    plt.bar(range(N_CHANNELS), channel_imp, color="steelblue")
    plt.xlabel("EEG Channel (0-61)")
    plt.ylabel("mean |SHAP value|")
    plt.title(f"Channel importance ({tag}, subject-dependent)")
    plt.tight_layout()
    out_ch = f'{cfg["paths"]["plots"]}shap_channel_importance_{tag}.png'
    plt.savefig(out_ch, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved channel importance to: {out_ch}")

    # --- (2) Temporal importance: average over channels -> (800,) ---
    time_imp = importance.mean(axis=0)   # (800,)
    time_axis_sec = np.arange(WINDOW_SIZE) / 200.0   # convert samples to seconds (200Hz)
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis_sec, time_imp, color="darkorange")
    plt.xlabel("Time within window (s)")
    plt.ylabel("mean |SHAP value|")
    plt.title(f"Temporal importance ({tag}, subject-dependent)")
    plt.tight_layout()
    out_time = f'{cfg["paths"]["plots"]}shap_temporal_importance_{tag}.png'
    plt.savefig(out_time, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal importance to: {out_time}")

    # --- (3) Full channel x time heatmap (optional, very wide) ---
    plt.figure(figsize=(14, 8))
    plt.imshow(importance, aspect="auto", cmap="viridis")
    plt.colorbar(label="mean |SHAP value|")
    plt.xlabel("Time sample (0-799)")
    plt.ylabel("EEG Channel (0-61)")
    plt.title(f"SHAP channel-time importance ({tag})")
    plt.tight_layout()
    out_map = f'{cfg["paths"]["plots"]}shap_channel_time_{tag}.png'
    plt.savefig(out_map, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved channel-time heatmap to: {out_map}")

    # --- Numeric summary: top-10 most important channels ---
    top_ch = np.argsort(channel_imp)[::-1][:10]
    print("\nTop-10 most important channels:")
    for ch in top_ch:
        print(f"  channel {ch:2d}: {channel_imp[ch]:.4f}")


if __name__ == "__main__":
    run_shap(folds_to_use=range(5))

    