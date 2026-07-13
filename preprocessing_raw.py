#Preprocessing script for EEG data without DE features
import os
import numpy as np
import scipy.io as sio
from pathlib import Path

FS = 200 # sampling rate
WINDOW_SEC = 4
WINDOW_SIZE = FS * WINDOW_SEC # 800 samples per window
N_CHANNELS = 62 # number of EEG channels

TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# remap labels to 0, 1, 2 for negative, neutral, positive so that they can be used as class indices in PyTorch
LABEL_MAPPING = {
    -1: 0,  # Negative emotion
    0: 1,   # Neutral emotion
    1: 2    # Positive emotion
}

# -----step 1: Load .mat files and extract EEG data and labels-----

def load_mat_file(file_path):
    """
    Load a .mat file and return EEG trial arrays sorted by trial index.

    Args:
        file_path (str): Path to the .mat file.
    
    Returns: 
        eeg_keys (list): List of EEG trial keys sorted by trial index. e.g.['ww_eeg1', 'ww_eeg2', 'ww_eeg3', ...]
        mat_data (dict): Dictionary containing the loaded .mat data.
    """
    mat_data = sio.loadmat(file_path)

    # filter out meta data in mat_data
    eeg_keys = [k for k in mat_data.keys() 
                if not k.startswith("__") and mat_data[k].shape[0] == N_CHANNELS
    ]
    
    # sort keys based on trial index
    eeg_keys.sort(key=lambda k: int(''.join(c for c in k if c.isdigit())))

    return eeg_keys, mat_data

#-----step 2Sliding window function to segment EEG data into 4-second windows-----

def sliding_window(signal):
    """
    cut one trial signal into non-overlapping 4-second windows

    Args:
        signal: np.ndarray, shape(62, T), 
        signal is one trial's EEG data loaded from mat_data[key] in process_one_file(),
        where key is one of 'ww_eeg1' ... 'ww_eeg15'.
    
    Returns:
        segments: np.ndarray, shape(N_segments, 62, WINDOW_SIZE),
        WINDOW_SIZE = 800 samples for 4 seconds at 200 Hz

        segments will be from [
            array(62, 800), # first 4-second window
            array(62, 800), # second 4-second window
            ...
        ]
        to:
        (N_segments, 62, 800) using np.stack(segments, axis=0)

    """
    total_time = signal.shape[1] # signal.shape: (62, T)
    segments = [] # initiate an empty list to store segments

    start = 0
    while start + WINDOW_SIZE <= total_time:
        segments.append(signal[:, start: start + WINDOW_SIZE]) # take all 62 channels and the next 800 samples as a segment
        start += WINDOW_SIZE # move the start index to the next window, non-overlapping
    
    if len(segments) == 0:
        print("Warning: No segments were created. Check the signal length.")
        return None
    
    return np.stack(segments, axis=0).astype(np.float32) # convert list of segments to a new numpy array with shape(N_segments, 62, WINDOW_SIZE(800))

#-----step 3: Per-channel normalization-----

def normalize(segments):
    """
    Z-score normalization for each segment independently per channel
    Uses z-score normalization, because it is a common technique in EEG proprecessing, which
    removes amplitutde differences across subjects and electrodes, making the data more comparable across trials and subjects.

    Args:
        segments: np.ndarray, shape(N_segments, 62, WINDOW_SIZE(800))
    
    Returns:
        normalized: np.ndarray, shape(N_segments, 62, WINDOW_SIZE(800))

    """
    mean = segments.mean(axis=2, keepdims=True)  # shape (N_segments, 62, 1), normalize on time dimension 
    std = segments.std(axis=2, keepdims=True) # shape (N_segments, 62, 1)
    return (segments - mean) / (std + 1e-8) # add a small value to std to avoid division by zero


#------step 4: Assign lables-----

def assign_labels(n_segments, trial_idx):
    """
    Because trials are segmented into 4-second windows, each segment should inherit the label of its correspoding trial

    Args:
        n_segments: int - number of segments from this trial
        trial_idx: int - trial index (0-14) in each mat file, used to get the label from TRIAL_LABELS, will be defined in process_one_file() when iterating through the 15 trials in each .mat file
    
    Returns:
        labels: np.ndarray, shape(n_segments,), with values {0, 1, 2} for negative, neutral, positive emotions
    """
    label = LABEL_MAPPING[TRIAL_LABELS[trial_idx]] # get the label for this trial
    return np.full((n_segments,), label, dtype=np.int64) # create an array of shape (n_segments,) for each trial with the same label, e.g. [0, 0, 0, 0] for 4 segments of a negative trial

#-----setp 5: Process one subject-session file-----

def process_one_file(file_path):
    """
    Run the full pipiline on one .mat file(one subject-session), including loading, segmenting, normalizing, and labeling.
    
    Args:
        file_path: str - path to the .mat file
    
    Returns:
        signals: np.ndarray, shape(N_total_segments, 62, WINDOW_SIZE(800)), all segments from this subject-session
        labels: np.ndarray, shape(N_total_segments,), all labels from this subject-session, e.g. [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...] for all segments from negative, neutral, positive trials

    """
    eeg_keys, mat_data = load_mat_file(file_path) # call load_mat_file() to get the sorted eeg_keys and mat_data from the .mat file

    # initiate empty lists to store all segments and labels for this subject-session
    all_signals = []
    all_labels = []

    for trial_idx, key in enumerate(eeg_keys): # iterate through the 15 trials in each .mat file
        signal = mat_data[key] # shape(62, T), extract signal value for each trial for the key, e.g. 'ww_eeg1', 'ww_eeg2', ..., 'ww_eeg15'
        segments = sliding_window(signal) # call sliding_window() to segment the signal into 4, shape (N_segments, 62, WINDOW_SIZE(800))

        if segments is None:
            print(f"Warning: No segments were created for trial {trial_idx} in file {file_path}. Skipping this trial.")
            continue

        segments = normalize(segments) # call normalize() to normalize each segment independently per channel, shape (N_segments, 62, WINDOW_SIZE(800))
        labels = assign_labels(len(segments), trial_idx) # call assign_labels() to assign labels for each segment, shape (N_segments,)

        all_signals.append(segments) # shape (N_segments, 62, WINDOW_SIZE(800)), append to the list of all segments for this subject-session, e.g. [array(62, 800), array(62, 800), ..., array(62, 800)] for all segments 
        all_labels.append(labels)
    
    # concatenate all segments and labels for this subject-session
    signals =  np.concatenate(all_signals, axis=0) # shape (N_total_segments, 62, WINDOW_SIZE(800))
    labels = np.concatenate(all_labels, axis=0) # shape (N_total_segments,)

    return signals, labels

#-----step 6: Process entire dataset-----

def preprocess_dataset(config):
    """
    Iterate over all .mat files in the preprocessed_EEG directory, 
    process each file using process_one_file(),
    and save the resulting signals and labels as .npy files in the processed_raw_data directory.
    """
    raw_path = config["raw_data"]
    save_path = config["paths"]["raw"]
    os.makedirs(save_path, exist_ok=True) # create the save directory if it doesn't exist

    print(f"Input :{raw_path}")
    print(f"Output: {save_path}")
    print(f"Window: {WINDOW_SEC}s, Sampling Rate: {FS}Hz, Window Size: {WINDOW_SIZE} samples")

    mat_files = sorted([
        f for f in os.listdir(raw_path)
        if f.endswith(".mat") and not f.startswith("label")
    ])
    print(f"Found {len(mat_files)} .mat files in {raw_path}")

    for file_counter, file_name in enumerate(mat_files): # use file_counter to mannually create a unique sample name for each subject-session, e.g. sample_0, sample_1, ..., sample_45
        file_path = os.path.join(raw_path, file_name)
        subject_id = file_name.split("_")[0] # extract subject_id from file name

        print(f"Processing: {file_name} ...", end=" ")

        signals, labels = process_one_file(file_path) # call process_one_file() to process the .mat file and get the signals and labels

        save_dict ={
            "signals": signals,
            "labels": labels,
            "subject_id": str(subject_id)
        }

        save_name = f"sample_{file_counter}.npy" # create a unique sample name for each subject-session, e.g. sample_0, sample_1, ..., sample_45
        np.save(os.path.join(save_path, save_name), save_dict) # save the signals and labels as a .npy file in the processed_raw_data directory

        print(f"-> signals {signals.shape}, labels {labels.shape}")

    print(f"\nDone. {len(mat_files)} files saved to {save_path}")


if __name__ == "__main__":
    config = {
        "raw_data": "/home/space/datasets/bsa03/SEED/Preprocessed_EEG",
        "mode": "raw", 
        "paths": {
            "de": "/home/bsa06/projects/DLBSA-SEED-EEG/processed_seed_4s", # when use DE features for training, this is the path to the processed DE features
            "raw": "/home/bsa06/projects/DLBSA-SEED-EEG/processed_raw_data"
        }
    }
    preprocess_dataset(config)


