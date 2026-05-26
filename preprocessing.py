# preprocessing.py
import os
import scipy.io as sio
import numpy as np
from pathlib import Path
from scipy.signal import stft

def extract_de_features_4s(signal, fs=200):
    """
    4s Pipeline: Extract Differential Entropy (DE) features from 4-second non-overlapping windows of the raw EEG signal.
    This function implements the core feature extraction steps as outlined in the assignment requirements.
    It takes a raw EEG signal of shape (62, time_points) and returns a feature array of shape (N_segments, 62, 5) where N_segments is the number of 4-second windows, 62 is the number of channels, and 5 corresponds to the five frequency bands (delta, theta, alpha, beta, gamma).
    N_segments = total_time_points / (fs * window_duration) = total_time_points / 800 for 4-second windows at 200 Hz sampling rate.   
    The steps include:
    1. Short-Time Fourier Transform (STFT) to get the time-frequency representation of the signal.
    2. Band-specific energy calculation for the defined frequency bands.
    3. Differential Entropy (DE) calculation for each band and channel.
    4. Temporal smoothing using a simple moving average to approximate the effect of a Linear Dynamic System (LDS) for smoothing the features over time.
    5. Feature normalization using Z-score normalization across the time axis for each channel and band to ensure that the features are on a comparable scale.
    The final output is a smoothed and normalized DE feature array that can be used for training.
    """
    window_duration = 4  # Duration of 4s windows
    nperseg = fs * window_duration  # Number of samples per segment (800 samples at 200 Hz)
    
    # 1. STFT to get time-frequency representation
    freqs, times, Zxx = stft(signal, fs=fs, window='hann', 
                             nperseg=nperseg, noverlap=0, axis=-1) # Use hann window, noverlap=0 for non-overlapping windows, and axis=-1 to apply STFT along the time dimension. freqs: array of sample frequencies, times: array of segment times, Zxx: STFT of the signal with shape (62, frequencies, N_segments)
    Zxx = np.abs(Zxx) # Get the absolute value of the STFT coefficients to represent the magnitude of the frequency components
    Zxx = np.transpose(Zxx, (2, 0, 1)) # Rearrange to (N_segments, 62, frequencies) for easier processing
    n_segments = Zxx.shape[0] # Total number of 2-second segments extracted from the signal
    
    bands = {
        'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 13), 'beta': (14, 30), 'gamma': (31, 50)
    } # Define the frequency bands of interest for EEG analysis according to the paper
    
    # 2. Band-specific energy calculation and DE computation

    de_features = np.zeros((n_segments, 62, 5)) # Initialize an array to hold the DE features for each segment, channel, and band. Shape: (N_segments, 62 channels, 5 bands)
    
    for t in range(n_segments): # Loop over each time segment
        for b_idx, (band_name, (low, high)) in enumerate(bands.items()):
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            band_energy = np.mean(Zxx[t][:, idx] ** 2, axis=-1) # Calculate the average energy in the current band for each channel by taking the mean of the squared magnitudes of the STFT coefficients across the frequencies that fall within the band
            de_band = 0.5 * np.log(2 * np.pi * np.e * band_energy + 1e-8)
            de_features[t, :, b_idx] = de_band # Store the DE features for the current band and all channels in the corresponding slice of the de_features array

    # 3. Temporal smoothing using a simple moving average to approximate the effect of a Linear Dynamic System (LDS)
    de_features_smoothed = np.zeros_like(de_features) # Initialize an array to hold the smoothed DE features. Shape: (N_segments, 62 channels, 5 bands)
    window_size = 5 # Use a window size of 5 segments for smoothing.
    for c in range(62): # Loop over each channel
        for b in range(5): # Loop over each band
            de_features_smoothed[:, c, b] = np.convolve( 
                de_features[:, c, b], np.ones(window_size)/window_size, mode='same' # "same" mode to keep the output the same length as the input. 
            )

    # 4. Feature normalization using Z-score normalization across the time axis for each channel and band
    mean = de_features_smoothed.mean(axis=0, keepdims=True)
    std = de_features_smoothed.std(axis=0, keepdims=True)
    de_features_smoothed = (de_features_smoothed - mean) / (std + 1e-8)
    
    return de_features_smoothed # Shape: (N_segments, 62, 5) - Smoothed and normalized DE features for each segment, channel, and band

def preprocess_dataset(config):
    """
    Main preprocessing function that reads raw SEED EEG data, applies the 4s feature extraction pipeline, and saves the processed features and labels in the required format.
    """
    raw_path = config["paths"]["raw_data"]
    save_path = config["paths"]["processed_data"]
    
    os.makedirs(save_path, exist_ok=True) # Ensure the output directory exists, if not, create it. 
    print("Running preprocessing...") 
    
    file_counter = 0 # Counter to keep track of the number of processed files for naming the output .npy files sequentially (e.g., sample_0.npy, sample_1.npy, etc.)
    FS = 200 # Predefined sampling frequency of the SEED dataset based on the dataset documentation

    # SEED dataset has 15 trials per subject, and the global labels for these trials are as follows (based on the SEED paper and dataset documentation):
    seed_global_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    

    for file_name in sorted(os.listdir(raw_path)): # Loop through each file in the raw data directory, sorted alphabetically to ensure consistent processing order
        if not file_name.endswith(".mat") or file_name.startswith("label"): # Left out label and readme files, only process .mat files
            continue
            
        file_path = os.path.join(raw_path, file_name)
        
        # 1. Extract subject ID from the file name (e.g., "1_20131027.mat" → subject_id = "1")
        subject_id = file_name.split("_")[0]
        
        # 2. Load the .mat file using scipy.io.loadmat, which returns a dictionary where keys are variable names and values are the corresponding data arrays
        mat_data = sio.loadmat(file_path)
        
        # 3. Initialize lists to hold all segments and labels for the current subject across all trials
        all_signals_list = []
        all_labels_list = []
        
        # 4. Filter out keys that correspond to EEG data 
        eeg_keys = [
            k for k in mat_data.keys()
            if not k.startswith("__") and mat_data[k].shape[0] == 62 # Filter keys that do not start with "__" (to exclude metadata such as "__header__", "__version__", etc.) and have a shape where the first dimension is 62 (indicating they are EEG data with 62 channels)
        ]

        # Sort the EEG keys based on the trial number extracted from the key name to ensure we process trials in the correct order (0 to 14)
        eeg_keys.sort(key=lambda k: int(''.join(c for c in k if c.isdigit())))

        # Loop through each trial's EEG data, apply the 2s DE feature extraction pipeline, and collect the features and corresponding labels
        for trial_idx, key in enumerate(eeg_keys):
            raw_signal = mat_data[key]  # Shape: (62, time_points) - Raw EEG signal for the current trial
            
            # Apply the 4s DE feature extraction pipeline to the raw signal of the current trial, which returns an array of shape (N_segments, 62, 5) containing the smoothed and normalized DE features for each 4-second segment
            trial_features_4s = extract_de_features_4s(raw_signal, fs=FS)
            
            n_segments = trial_features_4s.shape[0] # Number of 4-second segments extracted from the current trial's raw signal
            
            # Assign the global label for the current trial to all its segments. The global label is determined by the trial index and the predefined seed_global_labels list. We add +1 to convert the original labels from (-1, 0, 1) to (0, 1, 2) for compatibility with PyTorch's CrossEntropyLoss, which expects class indices starting from 0.
            # +1 is added to convert original labels from (-1, 0, 1) to (0, 1, 2) for compatibility with PyTorch's CrossEntropyLoss which expects class indices starting from 0.
            current_label = seed_global_labels[trial_idx] + 1
            trial_labels = np.full((n_segments,), current_label, dtype=np.int64)
            
            all_signals_list.append(trial_features_4s)
            all_labels_list.append(trial_labels)
        
        # After processing all trials for the current subject, concatenate the features and labels from all trials to create a single array of features and a corresponding array of labels for the entire subject. This will allow us to save one file per subject containing all their processed data.
        if len(all_signals_list) > 0:
            final_signals = np.concatenate(all_signals_list, axis=0)  # Shape: (Total segments N, 62, 5)
            final_labels = np.concatenate(all_labels_list, axis=0)    # Shape: (Total segments N,)
            
            # 5. Each saved file MUST contain the required dict structure
            save_dict = {
                "signals": final_signals,        # Shape: (N, C, T)
                "labels": final_labels,          # Shape: (N,)
                "subject_id": str(subject_id)    # Subject ID
            }
            
            # Save the processed features and labels for the current subject in a .npy file using numpy's save function. The file name includes the subject ID for easy identification (e.g., "sample_0.npy", "sample_1.npy", etc.). Each saved file contains a dictionary with the keys "signals", "labels", and "subject_id" as required by the assignment specifications.
            save_name = f"sample_{file_counter}.npy"
            np.save(os.path.join(save_path, save_name), save_dict)
            
            print(f" Saved {save_name} for subject {subject_id}: signals shape={final_signals.shape}, labels shape={final_labels.shape}")
            file_counter += 1

    print("Preprocessing complete.")


def extract_de_features_2s(signal, fs=200):
    """Backward-compatible alias for the 4-second DE feature extractor."""
    return extract_de_features_4s(signal, fs=fs)


if __name__ == "__main__":

    config = {
        "paths": {
            "raw_data": "/home/space/datasets/bsa03/SEED/Preprocessed_EEG",
            "processed_data": "/home/bsa03/processed_seed_4s"
        }
    }
    preprocess_dataset(config)