# training.py

import os
import json
import torch
import copy
import torch.nn as nn
import statistics
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from dataset import BiosignalDataset
from model import get_model
from utils import (
    get_subject_ids,
    loso_split,
    lmso_split,
    kfold_split_indices,
    check_no_leakage,
    subject_dependent_splits
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

def evaluate(model, loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1) # find the index of the maximum value on the ouput matrix along class(column) dimension, which labelizes the predicted class for each sample in the batch

            correct += (preds == y).sum().item() # calculate the number of correct predictions by comparing the predicted labels with the true labels, and summing up the number of matches, and add to the total correct count
            total += y.size(0) # total number of samples in the batch

    return correct / total, total_loss / len(loader)


def save_results(results, config):
    model_type = config["model"]["type"]
    protocol = config["evaluation"]["protocol"] # get model type and evaluation protocol from config to distinguish results
    mode = config["mode"] # get mode from config to distinguish results
    filename = f"{mode}_{model_type}_{protocol}_results.json" # distinguish results by model type 
    path = os.path.join(config["paths"]["results"], filename)
    with open(path, "w") as f:
        json.dump({
            "mode": mode,
            "model": model_type,
            "protocol": protocol,
            "results": results,
            "mean_accuracy": sum(results) / len(results),
            "std_accuracy": statistics.stdev(results)
            }, f, indent=4)


# -------------------------
# Main experiment
# run_experiment function to run the entire training and evaluation process based on config.py
# this function will first create data splits based on the evaluation protocol defined in config, then for each fold
# it will create datasets and dataloaders for training and testing, initialize the model, define optimizer and loss function(criterion), run the training loop for the specified number of epochs, evaluate the model after each epoch, save the best model checkpoint for each fold, and finally save the results to a json file at the end of all folds
# -------------------------

def run_experiment(config):
    device = config["training"]["device"]
    protocol = config["evaluation"]["protocol"]
    model_type = config["model"]["type"]
    mode = config["mode"]

    results = []
    all_fold_train_losses = []
    all_fold_test_losses = []
    all_fold_train_accs = []
    all_fold_test_accs = []

    # -------------------------
    # SPLIT CREATION
    # -------------------------

    if protocol in ["loso", "lmso"]:
        subject_ids = get_subject_ids(config["paths"][config["mode"]]) # get subject ids from the processed data directory based on the mode (de or raw) defined in config

        if protocol == "loso":
            splits = loso_split(subject_ids)

        elif protocol == "lmso":
            splits = lmso_split(subject_ids, n_test_subjects=2)

    elif protocol == "kfold":
        full_dataset = BiosignalDataset(config, subject_ids=None)
        splits = kfold_split_indices(len(full_dataset), config["evaluation"]["num_folds"])
    
    elif protocol == "subject_dependent":
        splits = subject_dependent_splits(config["paths"][config["mode"]])

    else:
        raise ValueError("Unknown evaluation protocol")

    # -------------------------
    # TRAINING LOOP
    # -------------------------

    for fold, split in enumerate(splits):
        print(f"\nFold {fold+1}")

        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []

        # -------------------------
        # Dataset creation
        # -------------------------

        if protocol in ["loso", "lmso"]:
            train_sids, test_sids = split
            check_no_leakage(train_sids, test_sids)
            train_dataset_raw = BiosignalDataset(config, train_sids)
            print(f"train_dataset_raw size: {len(train_dataset_raw)}")
            mean, std = train_dataset_raw.compute_stats()
            train_dataset = BiosignalDataset(config, train_sids, mean=mean, std=std)
            test_dataset = BiosignalDataset(config, test_sids, mean=mean, std=std)
        
        elif protocol == "subject_dependent":
            train_sids, test_sids = split  # get train and test subject ids from the split
            train_dataset_raw = BiosignalDataset(config, train_sids)
            mean, std = train_dataset_raw.compute_stats()
            full_dataset = BiosignalDataset(config, train_sids, mean=mean, std=std)
            # 8:2 split of the full dataset for training and testing
            total = len(full_dataset)
            train_size = int(0.8 * total)
            test_size = total - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size],
                generator=torch.Generator().manual_seed(42) # add seed for reproducibility
            )



        else:  # then it's kfold
            full_dataset_raw = BiosignalDataset(config, subject_ids=None)

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

        if config["training"].get("optimizer", "adam") == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"].get("weight_decay", 1e-4)
            )
        else:
            optimizer = torch.optim.Adam( # define optimizer for training, using Adam which is a popular choice for deep learning models due to its adaptive learning rate capabilities, and pass model parameters and learning rate from config    
                model.parameters(),
                lr=config["training"]["learning_rate"]
            )

        criterion = nn.CrossEntropyLoss() # define loss function for multi-class classification, which is appropriate for our 3-class

        # -------------------------
        # Training
        # -------------------------
        best_acc = 0
        best_state = None
        # removed patience for early stopping here

        for epoch in range(config["training"]["epochs"]): # loop through epochs defined in config
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            train_acc, _ = evaluate(model, train_loader, device, criterion)
            test_acc, test_loss = evaluate(model, test_loader, device, criterion)

            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

            train_losses.append(loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_acc: # if current test accuracy is better than the best accuracy seen so far, update best accuracy and save the model checkpoint for this fold
                best_acc = test_acc
                best_state = copy.deepcopy(model.state_dict())
                # removed early stopping here
                

            final_test_acc = test_acc
        # -------------------------
        # Save checkpoint
        # -------------------------

        torch.save(
            best_state,
            os.path.join(config["paths"]["checkpoints"], f"{mode}_{model_type}_{protocol}_fold{fold}.pt")
        )

        all_fold_train_losses.append(train_losses)
        all_fold_test_losses.append(test_losses)
        all_fold_train_accs.append(train_accs)
        all_fold_test_accs.append(test_accs)

        results.append(final_test_acc) # append the final test accuracy for this fold to the results list
        
    print("\nFinal Results:", results)
    print("Mean Accuracy:", sum(results) / len(results))



# removed early stopping, keeping epochs fixed, and save the best model checkpoint for each fold, and save the results to a json file at the end of all folds
    train_loss_arr = np.array(all_fold_train_losses)
    test_loss_arr  = np.array(all_fold_test_losses)
    train_acc_arr  = np.array(all_fold_train_accs)
    test_acc_arr   = np.array(all_fold_test_accs)

    epochs_axis = np.arange(1, train_loss_arr.shape[1] + 1)

    def mean_std_plot(arr_train, arr_test, ylabel, fname):
        m_tr, s_tr = arr_train.mean(axis=0), arr_train.std(axis=0)
        m_te, s_te = arr_test.mean(axis=0),  arr_test.std(axis=0)

        plt.figure()
        plt.plot(epochs_axis, m_tr, label=f"Mean Train {ylabel} across folds")
        plt.fill_between(epochs_axis, m_tr - s_tr, m_tr + s_tr, alpha=0.2)
        plt.plot(epochs_axis, m_te, label=f"Mean Test {ylabel} across folds")
        plt.fill_between(epochs_axis, m_te - s_te, m_te + s_te, alpha=0.2)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(f"Mean {ylabel} (±std) - {mode} - {model_type} - {protocol}")
        plt.legend()
        plt.savefig(os.path.join(config["paths"]["plots"], fname))
        plt.close()

    mean_std_plot(train_loss_arr, test_loss_arr, "Loss",
                  f"{mode}_{model_type}_{protocol}_mean_loss_curve.png")
    mean_std_plot(train_acc_arr, test_acc_arr, "Accuracy",
                  f"{mode}_{model_type}_{protocol}_mean_accuracy_curve.png")

    save_results(results, config)

    return results