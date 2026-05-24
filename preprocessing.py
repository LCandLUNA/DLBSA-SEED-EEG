# preprocessing.py

import os
import numpy as np


def preprocess_dataset(config):
    """
    Preprocessing pipeline.

    Students should:
    - load raw data (BIDS or other format)
    - apply filtering / normalization
    - segment signals into (N, C, T)

    Each saved file MUST contain:
        {
            "signals": (N, C, T),
            "labels": (N,),
            "subject_id": str or int
        }

    One file can represent:
    - a run
    - a session
    - or any logical chunk
    """

    raw_path = config["paths"]["raw_data"]
    save_path = config["paths"]["processed_data"]

    os.makedirs(save_path, exist_ok=True)

    print("Running preprocessing...")

    file_counter = 0

    # -------------------------
    # Example structure (BIDS-like)
    # -------------------------
    for subject in os.listdir(raw_path):

        if not subject.startswith("sub-"):
            continue

        subject_id = subject
        subject_path = os.path.join(raw_path, subject)

        for root, _, files in os.walk(subject_path):
            for file in files:

                # -------------------------
                # TODO: filter valid data files (e.g. .snirf, .edf)
                # -------------------------

                file_path = os.path.join(root, file)

                # -------------------------
                # TODO: load real data here
                # -------------------------
                signals = np.random.randn(50, 1, 1000)
                labels = np.random.randint(0, 2, 50)

                save_name = f"sample_{file_counter}.npy"

                np.save(os.path.join(save_path, save_name), {
                    "signals": signals,
                    "labels": labels,
                    "subject_id": subject_id
                })

                file_counter += 1

    print("Preprocessing complete.")