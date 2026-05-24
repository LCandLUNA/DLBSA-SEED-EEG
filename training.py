# training.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import BiosignalDataset
from models import Model
from utils import (
    get_subject_ids,
    loso_split,
    lmso_split,
    kfold_split_indices,
    check_no_leakage
)


# -------------------------
# Training helpers
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)

        assert x.ndim == 3, f"Expected (B, C, T), got {x.shape}"

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def save_results(results, config):
    path = os.path.join(config["paths"]["results"], "results.json")
    with open(path, "w") as f:
        json.dump({"results": results}, f, indent=4)


# -------------------------
# Main experiment
# -------------------------

def run_experiment(config):
    device = config["training"]["device"]
    protocol = config["evaluation"]["protocol"]

    results = []

    # -------------------------
    # SPLIT CREATION
    # -------------------------

    if protocol in ["loso", "lmso"]:
        subject_ids = get_subject_ids(config["paths"]["processed_data"])

        if protocol == "loso":
            splits = loso_split(subject_ids)

        elif protocol == "lmso":
            splits = lmso_split(subject_ids, n_test_subjects=2)

    elif protocol == "kfold":
        full_dataset = BiosignalDataset(config, subject_ids=None)
        splits = kfold_split_indices(len(full_dataset), config["evaluation"]["num_folds"])

    else:
        raise ValueError("Unknown evaluation protocol")

    # -------------------------
    # TRAINING LOOP
    # -------------------------

    for fold, split in enumerate(splits):
        print(f"\nFold {fold+1}")

        # -------------------------
        # Dataset creation
        # -------------------------

        if protocol in ["loso", "lmso"]:
            train_sids, test_sids = split

            check_no_leakage(train_sids, test_sids)

            train_dataset = BiosignalDataset(config, train_sids)
            test_dataset = BiosignalDataset(config, test_sids)

        else:  # kfold
            train_idx, test_idx = split

            full_dataset = BiosignalDataset(config, subject_ids=None)

            train_dataset = Subset(full_dataset, train_idx)
            test_dataset = Subset(full_dataset, test_idx)

        # -------------------------
        # DataLoaders
        # -------------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False
        )

        # -------------------------
        # Model
        # -------------------------

        model = Model(config).to(device)

        # sanity check
        dummy = torch.randn(
            2,
            config["dataset"]["input_channels"],
            config["dataset"]["segment_length"]
        ).to(device)

        try:
            _ = model(dummy)
        except Exception as e:
            raise RuntimeError(f"Model forward failed: {e}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"]
        )

        criterion = nn.CrossEntropyLoss()

        # -------------------------
        # Training
        # -------------------------

        for epoch in range(config["training"]["epochs"]):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            acc = evaluate(model, test_loader, device)

            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

        # -------------------------
        # Save checkpoint
        # -------------------------

        torch.save(
            model.state_dict(),
            os.path.join(config["paths"]["checkpoints"], f"model_fold{fold}.pt")
        )

        results.append(acc)

    print("\nFinal Results:", results)
    print("Mean Accuracy:", sum(results) / len(results))

    save_results(results, config)

    return results