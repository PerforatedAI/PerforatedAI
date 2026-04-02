# MedHallu Hallucination Classification

This repository contains a PyTorch implementation for detecting hallucinated answers in medical question-answer pairs. It uses the **MedHallu** datasets from Hugging Face and trains a simple feedforward neural network with TF-IDF features.  

---

## Dataset

Two datasets from [UTAustin-AIHealth/MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu):

1. **pqa_artificial**  ~9k rows used for training.
2. **pqa_labeled**  ~1k rows used for validation and testing.

Both datasets contain the following fields:

- `Question`: The medical question.
- `Knowledge`: Supporting context (can be a list of strings).
- `Ground Truth`: Correct answer.
- `Hallucinated Answer`: Incorrect or hallucinated answer.
- `Category of Hallucination`: Different categories of hallucination
---

## Preprocessing

- Each sample is converted into **two examples**:
  - Positive example (`label=0`): `[Question + Knowledge] [SEP] Ground Truth`
  - Negative example (`label=1`): `[Question + Knowledge] [SEP] Hallucinated Answer`
- TF-IDF vectorization is applied to convert text into numerical features.
- Features are scaled using `StandardScaler`.

---

## Model

We use a **simple feedforward neural network**:

```text
Input (TF-IDF 2000 features)
  └─ Linear(2000 -> 256) -> ReLU -> Dropout(0.2)
      └─ Linear(256 -> 128) -> ReLU
          └─ Linear(128 -> 2)
```
---

## Optimizer

Adam

---

## Loss

CrossEntropy with class weights to handle imbalance

---

## Metrics

During training and evaluation, the following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Loss

