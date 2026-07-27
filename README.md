# DL-BSA Project: SEED EEG Emotion Recognition

This repository implements an EEG-based emotion recognition pipeline for the SEED dataset. It supports three-class classification of emotions using either raw EEG signals or Differential Entropy (DE) features, with training and evaluation workflows built around deep learning models.

## Overview

The project includes:

- preprocessing pipelines for DE features and raw EEG signals
- dataset loading and subject-wise splitting
- training of multiple EEG classification models
- checkpoint-based evaluation with macro F1, weighted F1, per-class F1, and confusion matrices
- learning curve generation for loss and accuracy
- SHAP-based interpretability analysis for model explanation

## Project structure

- [main.py](main.py): entry point for running the training workflow
- [config.py](config.py): experiment configuration, including dataset paths, model choice, and evaluation protocol
- [preprocessing.py](preprocessing.py): DE feature extraction pipeline
- [preprocessing_raw.py](preprocessing_raw.py): raw EEG preprocessing pipeline
- [dataset.py](dataset.py): dataset loader for processed EEG samples
- [model.py](model.py): model definitions for CNN, CNN-1D, MLP, MLP-plus, LSTM, CNN-LSTM, DGCNN, and DANN-based variants
- [training.py](training.py): training loop, checkpoint saving, and learning curve generation
- [evaluation.py](evaluation.py): checkpoint-based evaluation with F1 metrics and confusion matrices
- [shap_analysis_de.py](shap_analysis_de.py): SHAP analysis for DE-feature models
- [shap_analysis_raw.py](shap_analysis_raw.py): SHAP analysis for raw-signal models
- [utils.py](utils.py): helper functions for subject IDs, splits, and leakage checks
- [submit_train.sh](submit_train.sh): SLURM job script for training on the Hydra cluster
- [submit_eval.sh](submit_eval.sh): SLURM job script for evaluation on the Hydra cluster
- [submit_preprocess.sh](submit_preprocess.sh): SLURM job script for preprocessing on the Hydra cluster

## Dataset and labels

This project is built for the SEED dataset with the following setup:

- 62 EEG channels
- 3 emotion classes
- 200 Hz sampling rate
- 4-second EEG windows

Label mapping used by the code:

- 0: Negative
- 1: Neutral
- 2: Positive

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the experiment

Edit [config.py](config.py) to choose:

- mode: "de" or "raw"
- model type: for example, "cnn", "cnn1d", "mlp", "mlp_plus", "lstm", "cnn_lstm", "dgcnn", or "dann_cnn_lstm"
- evaluation protocol: currently the main evaluation workflow uses "loso" and "subject_dependent"

### 3. Run preprocessing

For DE features:

```bash
python preprocessing.py
```

For raw EEG signals:

```bash
python preprocessing_raw.py
```

### 4. Train the model

```bash
python main.py
```

### 5. Evaluate checkpoints

```bash
python evaluation.py
```

### 6. Run SHAP analysis (optional)

For DE-feature models:

```bash
python shap_analysis_de.py
```

For raw-signal models:

```bash
python shap_analysis_raw.py
```

## Running on the Hydra cluster

This project is designed to run on the Hydra cluster using SLURM job scripts.

Typical commands:

```bash
sbatch submit_preprocess.sh
sbatch submit_train.sh
sbatch submit_eval.sh
```

These scripts launch preprocessing, training, and evaluation jobs with the required resources and write logs under the logs directory.

## Default configuration

The repository currently defaults to:

- raw EEG input
- LOSO evaluation
- AdamW optimizer
- 130 training epochs
- CUDA when available, otherwise CPU

## Outputs

Training and evaluation artifacts are written to:

- outputs/checkpoints/: model checkpoints
- outputs/results/: JSON result files with per-fold accuracies, mean accuracy, and standard deviation
- outputs/plots/: learning curves, confusion matrices, and SHAP interpretability plots
