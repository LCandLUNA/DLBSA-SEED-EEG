# config.py

def get_config():
    config = {
        "paths": {
            "raw_data": "/home/space/datasets/bsa03/SEED/Preprocessed_EEG",
            "processed_data": "/home/bsa03/processed_seed_4s",
            "outputs": "./outputs/",
            "checkpoints": "./outputs/checkpoints/",
            "results": "./outputs/results/",
            "plots": "./outputs/plots/"
        },

        "dataset": {
            "name": "SEED",
            "input_channels": 62,    # 62channels
            "segment_length": 5,     # 5 frequency bands
            "num_classes": 3, 
        },

        "training": {
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 1e-3,
            "device": "cuda"  # change to "cpu" if needed
        },

        "evaluation": {
            "protocol": "loso",  # "loso" or "kfold"
            "num_folds": 5
        }
    }

    return config