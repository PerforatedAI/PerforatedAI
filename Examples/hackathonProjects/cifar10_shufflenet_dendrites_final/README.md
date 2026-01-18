
# CIFAR-10 — ShuffleNet with Perforated AI Dendrites

This project explores the application of **Perforated AI (PAI) dendritic growth**
to improve **accuracy–efficiency tradeoffs** on the CIFAR-10 image classification benchmark.

The work follows the official PerforatedAI workflow and demonstrates correct dendrite
integration, training behavior, and graph outputs.

---

## Dataset

**CIFAR-10**
- 60,000 RGB images (32×32)
- 10 object classes
- Standard train/test split
- Well-suited for efficiency-focused vision experiments

---

## Model Architecture

### Backbone
- **ShuffleNetV2** (lightweight CNN designed for efficiency)

### Experimental Variants

| Model | Description |
|------|------------|
| A | Larger baseline CNN (no dendrites) |
| B | Compressed ShuffleNet (no dendrites) |
| C | Compressed ShuffleNet + PAI dendrites |


---

## Regularization Methods

To reduce overfitting and improve generalization, the following techniques were applied:

### 1. Data Augmentation
- Random cropping with padding
- Random horizontal flipping
- Dataset normalization

**Effect:** Encourages invariance and prevents memorization.

---

### 2. Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Effect:**
- Reduces overconfident predictions
- Narrows training–validation accuracy gap
- Stabilizes learning after dendrite growth

---

### 3. Weight Decay (L2 Regularization)

```python
weight_decay = 5e-4
```

**Effect:**
- Penalizes large weights
- Encourages simpler representations
- Reduces overfitting in later epochs

---

### 4. Architectural Regularization
- Dropout in classifier layers
- Balanced capacity prior to dendrite growth

---

## Training Configuration

- Optimizer: SGD with momentum (PAI-managed)
- Scheduler: CosineAnnealingLR
- Learning Rate: 0.1
- Batch Size: 128
- Epoch Control: `while True` loop (PAI-controlled stopping)

---

## PAI Graph Outputs

PAI automatically generates diagnostic plots in:

```
PAI/PAI.png
```

These include:
- Training and validation accuracy curves
- Dendrite addition markers (vertical lines)
- Learning rate schedule
- Epoch timing comparison

---

## Interpreting the Graphs

- Dendrites are added after validation performance plateaus
- Training accuracy increases sharply after dendrite addition
- Validation accuracy improves steadily with reduced overfitting
- Parameter growth is controlled and justified by performance gains

---

## Key Observations

- Dendrites improve **capacity efficiency**
- Regularization is essential for stable dendritic learning
- Automatic stopping prevents unnecessary training
- Graphs confirm correct dendrite behavior

---

## Conclusion

This project demonstrates a complete and correct application of Perforated AI dendrites
on CIFAR-10, with improved efficiency, controlled overfitting, and interpretable
training dynamics.
