# config.py

def get_config():
    config = {
        "raw_data": "/home/space/datasets/bsa03/SEED/Preprocessed_EEG",
        
        # "de": use DE features(preprocessing.py)
        # "raw": use raw EEG signals(preprocessing_raw.py)

        "mode": "de", # options: "de", "raw"

        "paths": {
            "de": "/home/bsa06/projects/DLBSA-SEED-EEG/processed_seed_4s", # when use DE features for training, this is the path to the processed DE features
            "raw": "/home/bsa06/projects/DLBSA-SEED-EEG/processed_raw_data", # when use raw EEG signals for training, this is the path to the processed raw EEG signals

            "outputs": "/home/bsa06/projects/DLBSA-SEED-EEG/outputs/",
            "checkpoints": "/home/bsa06/projects/DLBSA-SEED-EEG/outputs/checkpoints/",
            "results": "/home/bsa06/projects/DLBSA-SEED-EEG/outputs/results/",
            "plots": "/home/bsa06/projects/DLBSA-SEED-EEG/outputs/plots/"
        },

        "dataset": {
            "name": "SEED",
            "eeg_channels": 62,    # 62 eeg channels
            "freq_bands": 5,     # 5 frequency bands
            "conv_in_channels": 1, # in_channels for Conv2d(number of feature types/DE features)
            "num_classes": 3, 
            "input_channels":1,
            "window_size": 800, # 4s * 200Hz = 800 samples
        },

        "model": {
            "type": "mlp_plus"      # model change
        },

        "training": {
            "batch_size": 32,
            "epochs": 80,
            "learning_rate": 1e-3,
            "optimizer": "adamw",  # optimizer change
            "weight_decay": 1e-4,
            "device": "cuda"  # change to "cpu" if needed
        },

        "evaluation": {
            "protocol": "loso",  # options: "loso", "lmso", "kfold", "subject_dependent"
            "num_folds": 5
        }
    }

    return config