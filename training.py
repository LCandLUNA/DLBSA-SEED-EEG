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
    filename = f"{model_type}_{protocol}_results.json" # distinguish results by model type 
    path = os.path.join(config["paths"]["results"], filename)
    with open(path, "w") as f:
        json.dump({
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

    results = []

    # initialize lists to store average loss and accuracy for all folds before splitting the datasets, which will be used for plotting
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

        # initialize lists to store average loss and accuracy for this fold, which will be used for plotting
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
                full_dataset, [train_size, test_size]
            )



        else:  # then it's kfold
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
        patience = 10 # set patience for early stopping
        no_improve = 0 # counter to track number of epochs without improvement in test accuracy

        for epoch in range(config["training"]["epochs"]): # loop through epochs defined in config
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            train_acc, _ = evaluate(model, train_loader, device, criterion)
            test_acc, test_loss = evaluate(model, test_loader, device, criterion)

            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

            # store the average loss and accuracy for this epoch in the lists for this fold, which will be used for plotting
            train_losses.append(loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            

            if test_acc > best_acc: # if current test accuracy is better than the best accuracy seen so far, update best accuracy and save the model checkpoint for this fold
                best_acc = test_acc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0 # if there is improvement, reset the no_improve counter
            else: 
                no_improve += 1 # if there is no improvement, add 1 to the no_improve counter

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        # -------------------------
        # Save checkpoint
        # -------------------------

        torch.save(
            best_state,
            os.path.join(config["paths"]["checkpoints"], f"{model_type}_{protocol}_fold{fold}.pt")
        )

        # store all average loss and accuracy for this fold in the lists for all folds, which will be used for plotting
        all_fold_train_losses.append(train_losses)
        all_fold_test_losses.append(test_losses)
        all_fold_train_accs.append(train_accs)
        all_fold_test_accs.append(test_accs)

        results.append(best_acc)

    print("\nFinal Results:", results)
    print("Mean Accuracy:", sum(results) / len(results))

    min_epochs = min(len(f) for f in all_fold_train_losses) # find the minimum number of epochs across all folds to adjust the length of the lists for plotting
    mean_train_loss = np.mean([f[:min_epochs] for f in all_fold_train_losses], axis=0) 
    mean_test_loss = np.mean([f[:min_epochs] for f in all_fold_test_losses], axis=0) 
    mean_train_acc = np.mean([f[:min_epochs] for f in all_fold_train_accs], axis=0) 
    mean_test_acc = np.mean([f[:min_epochs] for f in all_fold_test_accs], axis=0) 

    # plot mean loss and accuracy curves across folds
    plt.figure()
    plt.plot(mean_train_loss, label="Mean Train Loss across folds")
    plt.plot(mean_test_loss, label="Mean Test Loss across folds")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Mean Loss Curve - {model_type} - {protocol}")
    plt.legend()
    plt.savefig(os.path.join(config["paths"]["plots"], f"{model_type}_{protocol}_mean_loss_curve.png"))
    plt.close()

    # plot mean accuracy curves across folds
    plt.figure()
    plt.plot(mean_train_acc, label="Mean Train Accuracy across folds")
    plt.plot(mean_test_acc, label="Mean Test Accuracy across folds")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Mean Accuracy Curve - {model_type} - {protocol}")
    plt.legend()
    plt.savefig(os.path.join(config["paths"]["plots"], f"{model_type}_{protocol}_mean_accuracy_curve.png"))
    plt.close() 

    save_results(results, config)

    return results