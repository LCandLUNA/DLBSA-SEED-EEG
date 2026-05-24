# config.py

def get_config():
    config = {
        "paths": {
            "raw_data": "./data/raw/",
            "processed_data": "./data/processed/",
            "outputs": "./outputs/",
            "checkpoints": "./outputs/checkpoints/",
            "results": "./outputs/results/",
            "plots": "./outputs/plots/"
        },

        "dataset": {
            "name": "example",
            "input_channels": 1,
            "segment_length": 1000,
            "num_classes": 2,
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