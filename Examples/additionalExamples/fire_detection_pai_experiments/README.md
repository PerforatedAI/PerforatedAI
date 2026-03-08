# Wildfire Prediction from Satellite Data (California 2020)

This example demonstrates wildfire prediction using satellite-derived vegetation indices and land surface temperature, comparing:

- Baseline Neural Network (PyTorch)
- Neural Network with PerforatedAI dendrite restructuring

The goal is to evaluate whether dynamic dendrite growth improves classification performance under extreme class imbalance.

---

## Dataset

The dataset integrates three NASA Earthdata sources:

- FIRMS (VIIRS 375m) – Active fire detections  
- MODIS MOD13Q1 – NDVI & EVI vegetation indices (16-day composite)  
- MODIS MOD11A2 – Land Surface Temperature (8-day composite, averaged to 16-day windows)  

Final processed dataset:

- ~18M valid 1 km² pixels  
- ~0.08% fire pixels (extremely imbalanced)  
- Train / Val / Test split: 70 / 15 / 15  

For neural network experiments:

- Training uses a subsampled and balanced subset: all positives and a limited number of negatives, for stability and speed.
- Validation and test sets are also balanced, so precision, recall, and F1 metrics are meaningful.
- Full dataset imbalance is not used in this experiment — the model is trained on a balanced subset, but dendrite growth is still tested for its ability to improve classification.

---

## Model Architecture

```
Input (4 features)
->  Linear(4 -> 64)
->  ReLU
->  Linear(64 -> 64)
->  ReLU
->  Linear(64 -> 1)
```

- Output uses logits, with BCEWithLogitsLoss.
- pos_weight is dynamically set to balance classes during training, emphasizing positive fire pixels.
- Threshold for predictions: logits > 0 -> positive class.

The same architecture is used for:

- Baseline model
- Dendrite-enhanced model (PerforatedAI)

---

## Experiment

The script automatically runs two versions:

- `is_dendrite = False` ->  Standard MLP  
- `is_dendrite = True` ->  MLP with dynamic dendrite growth  

PerforatedAI uses fixed switching mode:

DOING_FIXED_SWITCH
fixed_switch_num = 4

This means dendrite restructuring is evaluated every 4 validation cycles, allowing the model to train sufficiently before architecture changes occur.

---

## Dendrite Growth Behavior

During training, three dendrites were added.

Switch events occurred at:

| Epoch | Event                                                                           |
| ----- | ------------------------------------------------------------------------------- |
| 4     | First dendrite added                                                            |
| 11    | Second dendrite added                                                           |
| 16    | Third dendrite added                                                            |
| 20    | Switch attempted but rejected because it did not improve validation performance |

When a switch occurs:

- The best-performing model checkpoint is restored
- A dendrite is added to increase model capacity
- The optimizer and scheduler are reset
- Training continues with the updated architecture

If a new dendrite does not improve validation performance, the previous architecture is restored.

### Example Results

| Model       | Test Accuracy | Precision | Recall | F1         |
| ----------- | ------------- | --------- | ------ | ---------- |
| Baseline    | 61.03%        | 0.6115    | 0.6053 | 0.6084     |
| + Dendrites | 59.76%        | 0.5693    | 0.8014 | **0.6657** |

Dendrites improve overall F1 and recall, dynamically increasing model capacity to better capture rare fire events.

---

### Interpretation

Although overall accuracy decreased slightly, dendrite restructuring significantly improved recall and F1 score.

Key changes:

| Metric   | Baseline | Dendrites  |
| -------- | -------- | ---------- |
| Recall   | 0.6053   | **0.8014** |
| F1 Score | 0.6084   | **0.6657** |

Recall improved substantially, meaning the model detects many more true fire events.

This is important because wildfire detection is a rare event problem. Missing a true fire pixel is typically far more costly than generating a few additional false positives.

Dendrites increase model capacity dynamically, allowing the network to capture patterns associated with rare fire events that the baseline architecture could not model effectively.

---

## Class Imbalance

Fire pixels represent <0.01% of total data.

Because of this:

- Balanced subsampling for training, validation, and test sets.
- BCEWithLogitsLoss with pos_weight to give more importance to positive samples.
- Recall is prioritized in evaluation metrics to ensure rare events are detected.

---

## How to Run

### Install Dependencies

```bash
pip install torch numpy scikit-learn perforatedai
```

### Dataset Location

Expected path:

```
fire_detection_pai_experiments/data/processed/modis_firms_train_val_test_dataset.npz
```

The file must contain:

```
X_train, y_train
X_val, y_val
X_test, y_test
```

### Launch Jupyter

From the project root directory:

```
jupyter notebook
```

Then open:

```
notebooks/fire_dendrite_vs_baseline.ipynb
```

Run all cells.

---

## Key Observations

- Baseline MLP achieves moderate performance under balanced training.
- PerforatedAI dynamically expands the network through dendrite growth.
- Three dendrites were added during training, increasing model capacity.
- This resulted in substantially improved recall and F1 score.
- Dynamic architecture growth allows the model to better capture rare wildfire patterns.

---

## Future Work

- Add meteorological features (wind, humidity, drought index)
- Explore spatio-temporal models (CNN / RNN / Transformers)
- Improve precision-recall tradeoff via threshold tuning
- Evaluate full imbalanced dataset without balancing
- Investigate dendrite growth behavior on larger architectures

---

## Folder Structure

```
fire_detection_pai_experiments/
│
├── README.md
├── data/
│   └── processed/
│       └── modis_firms_train_val_test_dataset.npz
│
└── notebooks/
    └── fire_dendrite_vs_baseline.ipynb
```

---

