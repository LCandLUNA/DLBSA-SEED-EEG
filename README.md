# DL-BSA Project – Starter Template

This repository provides a **minimal starting point** for the Deep Learning for Biosignal Analysis (DL-BSA) project. It is intentionally simple and is meant to be **modified and extended**.

---

## General Note

This is **not a fixed framework**.

You are expected to:
- modify any script  
- add or remove components  
- adapt the pipeline to your dataset and task  

The template only defines a **basic structure**.

---

## Workflow

1. `preprocessing.py`  
   Load raw data (e.g., BIDS), clean and segment signals, and save processed data.

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

## Processed Data Format

Each saved file must contain:

```python
{
    "signals": (N, C, T),
    "labels": (N,),
    "subject_id": identifier
}
```

- One file can correspond to a run, session, or any logical unit.  
- Multiple files per subject are allowed.  
- File naming is not restricted.

---

## Dimensions

The template assumes input shape:

```python
(B, C, T)
```

However, depending on your preprocessing and model, this can change.

You are responsible for ensuring:
- consistency between preprocessing, dataset, and model  
- correct input/output dimensions throughout the pipeline  

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
- K-Fold (sample-level)  

Subject-based splits are recommended for biomedical data.

---

## Final Remark

This template is only a starting point.