# config.py

def get_config():
    config = {
        "raw_data": "/home/space/datasets/bsa03/SEED/Preprocessed_EEG",
        "paths": {
            "processed_data": "/home/cliu/processed_seed_4s",
            "outputs": "/home/cliu/DLBSA-SEED-EEG/outputs/",
            "checkpoints": "/home/cliu/DLBSA-SEED-EEG/outputs/checkpoints/",
            "results": "/home/cliu/DLBSA-SEED-EEG/outputs/results/",
            "plots": "/home/cliu/DLBSA-SEED-EEG/outputs/plots/"
        },

        "dataset": {
            "name": "SEED",
            "eeg_channels": 62,    # 62 eeg channels
            "freq_bands": 5,     # 5 frequency bands
            "conv_in_channels": 1, # in_channels for Conv2d(number of feature types/DE features)
            "num_classes": 3, 
            "input_channels":1
        },

        "training": {
            "batch_size": 32,
            "epochs": 80,
            "learning_rate": 1e-3,
            "device": "cuda"  # change to "cpu" if needed
        },

        "model": {
            "type": "cnn"
        },

        "evaluation": {
            "protocol": "loso",  # "loso" or "kfold"
            "num_folds": 5
        }
    }

    return config