# DL-BSA Project – SEED EEG Emotion Recognition 

This project explores EEG-based emotion recognition using the SEED dataset. The goal is to classify emotional states into three categories:
- Positive
- Neutral
- Negative

---

## Project Pipeline

```text
Dataset
(SEED)
    ↓
Preprocessing
(STFT + DE Features)
    ↓
Feature Dataset
(.npy files)
    ↓
Model Training
(CNN / MLP / TBD)
    ↓
Emotion Classification
(Positive / Neutral / Negative)
    ↓
Evaluation
(LOSO / LMSO)
```

---

## Workflow

1. `preprocessing.py`  
   Generate Differential Entropy (DE) features from SEED EEG recordings.
   Current preprocessing steps:
   - Short-Time Fourier Transform (STFT)
   - Frequency band decomposition
   - Differential Entropy (DE) feature extraction
   - Temporal smoothing
   - Z-score normalization

2. `dataset.py`  
   Load processed data and return samples.

3. `models.py`  
   Define your model.

4. `training.py`  
   Train and evaluate the model.

Run everything with:

```bash
python main.py
```

---

## Dataset

**Dataset:** SEED (SJTU Emotion EEG Dataset)

- 15 subjects
- 45 recording sessions
- 62 EEG channels
- Sampling rate: 200 Hz
- 3 emotion classes

### Label Mapping

| Emotion | Label |
|----------|--------|
| Negative | 0 |
| Neutral | 1 |
| Positive | 2 |

The project uses the official `Preprocessed_EEG` release of the SEED dataset.

---

## Preprocessing

The preprocessing pipeline includes:

- Short-Time Fourier Transform (STFT)
- Five-band frequency decomposition
  - Delta (1–3 Hz)
  - Theta (4–7 Hz)
  - Alpha (8–13 Hz)
  - Beta (14–30 Hz)
  - Gamma (31–50 Hz)
- Differential Entropy (DE) feature extraction
- Moving-average temporal smoothing
- Z-score normalization

---

## Output Format

Each processed sample is stored as:

```python
{
    "signals": ndarray,
    "labels": ndarray,
    "subject_id": str
}
```

Feature shape:

```python
signals.shape = (N, 62, 5)
```

where:

- `N` = number of EEG segments
- `62` = EEG channels
- `5` = frequency bands

---

## Model

You must implement your own model in `models.py`.

Requirements:
- input: `(B, C, T)` (or your adapted format)  
- output: task-dependent  

---

## Training

The provided `training.py` assumes a **classification task**:
- loss: CrossEntropyLoss  
- metric: accuracy  

If your task is different (e.g., regression or segmentation), you must modify:
- loss function  
- model output  
- evaluation metric  

---

## Evaluation

Supported protocols:
- LOSO (Leave-One-Subject-Out)  
- LMSO (Leave-Multiple-Subjects-Out)  

Subject-based splits are recommended for biomedical data.

---

## References

- W.-L. Zheng and B.-L. Lu, “Investigating critical frequency bands and channels for eeg-based emo tion recognition with deep neural networks”, IEEE Transactions on Autonomous Mental Development, vol. 7, no. 3, pp. 16