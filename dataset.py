# dataset.py
# Dataset class for loading preprocessed EEG data
# This file will read data according to the config, and transfer data into Pytorch tensors for training and evaluation

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BiosignalDataset(Dataset): # class inherit from torch.utils.data.Dataset

    def __init__(self, config, subject_ids=None, mean=None, std=None): # load all npy data files from processed_data path, filter by subject_ids if provided
        """
        If subject_ids is None → load ALL data (used for K-Fold)
        Otherwise → filter by subject_ids (used for LOSO / LMSO)
        mean, std: optional normalization stats (numpy arrays, shape (62, 5)).
        If provided, signals will be normalized using these stats.
        If not provided, no normalization is applied (raw signals returned).

        """
        # initiate empty lists to store signals, labels, and subject_ids from all npy files
        signal_list = []
        labels_list = []
        subject_id_list = []

        data_path = config["paths"]["processed_data"] # read from config

        for file in os.listdir(data_path): # loop through all 45 npy files 
            if not file.endswith(".npy"): # leave out non-npy files
                continue

            file_path = os.path.join(data_path, file)

            try:
                sample = np.load(file_path, allow_pickle=True).item() # allow pickle to load dict from npy file
            except Exception:
                continue

            subject_id = sample["subject_id"]

            # Filter if subject_ids provided
            if subject_ids is not None and subject_id not in subject_ids:
                continue

            signals = sample["signals"]
            labels = sample["labels"]

            # append all signals, labels, and subject_ids to numpy arrays for the dataset
            signal_list.append(signals)
            labels_list.append(labels)
            subject_id_list.extend([subject_id] * len(labels)) # repeat subject_id for each sample in the file

        if len(signal_list) > 0:
            self.signals = np.concatenate(signal_list, axis=0) # concatenate all signals into one numpy array, shape (total_samples, 62, 5)
            self.labels = np.concatenate(labels_list, axis=0) # concatenate all labels into one numpy array, shape (total_samples,)
        
        else:
            self.signals = np.zeros((0, 62, 5)) # if no data loaded, create empty array with correct shape
            self.labels = np.zeros((0,), dtype=np.int64)

        self.subject_ids = subject_id_list # list of subject_ids for each sample, length = total_samples

        # Normalization (if mean and std provided)
        self.mean = mean
        self.std = std

        if self.mean is not None and self.std is not None:
            self.signals = (self.signals - self.mean) / (self.std + 1e-8) # incase std is zero, add small value to avoid division by zero
        
    def compute_stats(self): # compute mean and std for normalization, return as numpy arrays with shape (62, 5)
        mean = self.signals.mean(axis=0) # (62,5)
        std = self.signals.std(axis=0) # (62,5)
        return mean, std


    def __len__(self):
        return len(self.signals) # calculate total number of samples in the dataset
    

    def __getitem__(self, idx): # load single sample by index, return signal, label, and subject_id as tensors

        x = torch.tensor(self.signals[idx], dtype=torch.float32) # shape (62, 5) where 62 channels and 5 frequency bands
        x = x.unsqueeze(0) # add in_channel dimension for Conv2d model input, -> shape (1, 62, 5)
        y = torch.tensor(self.labels[idx], dtype=torch.long) # 0, 1, or 2 for 3 classes (positive, neutral, negative)


        return {
            "signal": x,   # (1, 62, 5)
            "label": y,
            "subject_id": self.subject_ids[idx]
        }

    def get_all_subject_ids(self): 
        return self.subject_ids