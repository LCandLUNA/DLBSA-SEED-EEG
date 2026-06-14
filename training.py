# training.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import BiosignalDataset
from model import get_model
from utils import (
    get_subject_ids,
    loso_split,
    lmso_split,
    kfold_split_indices,
    check_no_leakage
)

# -------------------------
# Training helpers
# define train_one_epoch to train the model for one epoch and return the average loss
# optimizer, criterion are defined in the run_experiment function, and passed to train_one_epoch for training
# this helper function will be called in the training loop for each epoch, and it will put gradients to zero, perform forward pass, calculate loss, perform backward pass, and update model parameters using the optimizer, and accumulate total loss for the epoch to return the average loss at the end of the epoch
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader) # average loss for the epoch

# -------------------------
# Evaluation helper
# define evaluate function to evaluate the model on the test set and return accuracy
# this function will be called in the training loop after each epoch to evaluate the model's performance on the test set, and it will put the model in evaluation mode, disable gradient calculation, perform forward pass on the test data, calculate predictions, and compare with true labels to calculate accuracy, which will be returned at the end of the evaluation
# -------------------------

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1) # find the index of the maximum value on the ouput matrix along class(column) dimension, which labelizes the predicted class for each sample in the batch

            correct += (preds == y).sum().item() # calculate the number of correct predictions by comparing the predicted labels with the true labels, and summing up the number of matches, and add to the total correct count
            total += y.size(0) # total number of samples in the batch

    return correct / total


def save_results(results, config):
    model_type = config["model"]["type"]
    filename = f"{model_type}_results.json" # distinguish results by model type 
    path = os.path.join(config["paths"]["results"], filename)
    with open(path, "w") as f:
        json.dump({"results": results}, f, indent=4)


# -------------------------
# Main experiment
# run_experiment function to run the entire training and evaluation process based on config.py
# this function will first create data splits based on the evaluation protocol defined in config, then for each fold
# it will create datasets and dataloaders for training and testing, initialize the model, define optimizer and loss function(criterion), run the training loop for the specified number of epochs, evaluate the model after each epoch, save the best model checkpoint for each fold, and finally save the results to a json file at the end of all folds
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

            train_dataset_raw = BiosignalDataset(config, train_sids)
            mean, std = train_dataset_raw.compute_stats() # compute stats from training dataset for normalization, to avoid data leakage from test set

            train_dataset = BiosignalDataset(config, train_sids, mean=mean, std=std) # normalize training dataset using its own stats
            test_dataset = BiosignalDataset(config, test_sids, mean=mean, std=std) # normalize test dataset using training stats to avoid data leakage
            

        else:  # if not loso or lmso, then it's kfold
            full_dataset_raw = BiosignalDataset(config, subject_ids=None)() 

            train_idx, test_idx = split

            # compute stats from training dataset only
            train_singals_only = full_dataset_raw.signals[train_idx] # get training signals using train indices
            mean = train_singals_only.mean(axis=0) # compute mean across samples for
            std = train_singals_only.std(axis=0) # compute std across samples for normalization

            full_dataset = BiosignalDataset(config, subject_ids=None, mean=mean, std=std) # create full dataset with normalization using training stats

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

        model = get_model(config).to(device) # initialize model based on config, and move to device (GPU or CPU)

        optimizer = torch.optim.Adam( # define optimizer for training, using Adam which is a popular choice for deep learning models due to its adaptive learning rate capabilities, and pass model parameters and learning rate from config    
            model.parameters(),
            lr=config["training"]["learning_rate"]
        )

        criterion = nn.CrossEntropyLoss() # define loss function for multi-class classification, which is appropriate for our 3-class

        # -------------------------
        # Training
        # -------------------------

        for epoch in range(config["training"]["epochs"]): # loop through epochs defined in config
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