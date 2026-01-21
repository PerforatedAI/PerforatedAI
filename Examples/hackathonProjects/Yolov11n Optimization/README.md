# We made YOLOv11n more accurate with Dendrites

## Intro

**Description:**

This project demonstrates the application of PerforatedAI's dendritic neural network optimization to YOLOv11n object detection on the Pascal VOC2007 dataset. The goal is to prove that dendrite-enhanced networks can achieve **higher accuracy** on the same data compared to traditional neural networks.

We implement a custom training loop that integrates PAI's dendritic optimization with Ultralytics' YOLOv11n, enabling:
- Adaptive plateau detection (DOING_HISTORY mode) for optimal dendrite addition using Open Source PAI Dendrites
- Data efficiency experiments at multiple training set sizes (100%, 50%)

**Team:**

Neuron AI

## Project Impact

Object detection is fundamental to numerous real-world applications including autonomous vehicles, surveillance systems, robotics, and medical imaging. YOLOv11n represents the latest evolution in the YOLO family, designed for real-time detection with high accuracy.

Improving the accuracy of an object detection system matters because:

1. **Safety-Critical Applications**: In autonomous driving and medical imaging, even small improvements in detection accuracy can prevent accidents or catch diseases earlier, potentially saving lives.

2. **Data Efficiency**: Many real-world scenarios have limited labeled data. If dendrites can achieve higher accuracy with the same data (or match baseline accuracy with less data), this reduces annotation costs and enables deployment in data-scarce domains.

3. **Resource Optimization**: Better accuracy with the same model architecture means organizations don't need to scale up to larger, more expensive models to achieve their performance targets.

4. **Scalability**: Improvements on benchmark datasets like VOC2007 typically transfer to custom datasets, making this research applicable across industries from retail to agriculture to manufacturing.

---

## Usage Instructions

Download PASCAL VOC 2007 dataset and create it in datasets/VOC folder .
Yaml file is already provided in the repo. 

### Run Experiments

**Reproduce Our Exact Results (All 4 Experiments with Open Source GD):**

```bash
# Baseline 100% - No dendrites, full training data
python run_experiments.py --experiment baseline_100 --data-dir datasets/VOC

# Dendrite 100% - With dendrites (Open Source GD), full training data
python run_experiments.py --experiment dendrite_100 --data-dir datasets/VOC --opensource-gd

# Baseline 50% - No dendrites, 50% training data
python run_experiments.py --experiment baseline_50 --data-dir datasets/VOC

# Dendrite 50% - With dendrites (Open Source GD), 50% training data
python run_experiments.py --experiment dendrite_50 --data-dir datasets/VOC --opensource-gd
```

### Requirements

The main dependencies are:
- Python 3.10+
- PyTorch >= 2.0.0
- Ultralytics (YOLOv11)
- PerforatedAI (dendrite library)

---

## Results

This project demonstrates that **Dendritic Optimization significantly improves object detection accuracy** on Pascal VOC2007. Comparing the best traditional model to the best dendritic model:

### Full Data (100%) Results

| Model | Final Validation mAP50 | Notes |
|-------|------------------------|-------|
| Traditional (Baseline) | 55.77 | YOLOv11n, 100% training data |
| **Dendritic (PAI)** | **58.07** | YOLOv11n + Dendrites, 100% training data |

**Remaining Error Reduction: 5.20%**

> Calculation: Error drops from 44.23% (100 - 55.77) to 41.93% (100 - 58.07).
> That's a 2.30 percentage point drop = (2.30 / 44.23) × 100 = **5.20%** remaining error eliminated.

### Reduced Data (50%) Results

| Model | Final mAP50 | Notes |
|-------|------------------------|-------|
| Traditional (Baseline) | 45.41 | YOLOv11n, 50% training data |
| **Dendritic (PAI)** | **48.31** | YOLOv11n + Dendrites, 50% training data |

**Remaining Error Reduction: 5.31%**

> Calculation: Error drops from 54.59% to 51.69%.
> That's a 2.90 percentage point drop = (2.90 / 54.59) × 100 = **5.31%** remaining error eliminated.

### Summary Table

| Data % | Baseline mAP50 | Dendritic mAP50 | Improvement | Remaining Error Reduction |
|--------|----------------|-----------------|-------------|---------------------------|
| 100% | 55.77 | 58.07 | +2.30 | 5.20% |
| 50% | 45.41 | 48.31 | +2.90 | 5.31% |

### Key Findings

1. **Consistent Improvement**: Dendrites provide a consistent ~2-3 mAP50 improvement across different data percentages.

2. **Data Efficiency**: The dendritic model at 50% data (48.31) approaches the baseline at 100% data (55.77), demonstrating potential for data-efficient training.

3. **Scalable Benefits**: The improvement percentage remains stable (or slightly increases) as data decreases, suggesting dendrites are particularly valuable in data-constrained scenarios.

---

## Raw Results Graph

*PAI GRAPH FOR 100% DATA*

![Raw Results Graph](dendrite_100.png)

---
*PAI GRAPH FOR 50% DATA*

![Raw Results Graph](dendrite_50.png)

---

## Clean Results Graph

*COMPARISON GRAPH FOR 100% DATA*

![Raw Results Graph](Full.jpeg)

---
*COMPARISON GRAPH FOR 50% DATA*

![Raw Results Graph](Half.jpeg)

---


### Key Implementation Details

1. **Custom Training Loop**: We implement a custom training loop because Ultralytics' default `YOLO.train()` cannot be used with PAI - `add_validation_score()` must be called every epoch.

2. **Reproducibility**: All experiments use seed=42 with deterministic settings for reproducibility.

3. **Module Configuration**: We configure PAI to add dendrites to YOLO's feature extraction blocks (C3k2, C3k, C2PSA, Bottleneck, PSABlock) while tracking but not modifying normalization layers.

### Dataset

**Pascal VOC2007** (By Ultralytics):
- 20 object classes
- ~5,000 training images
- ~5,000 test images
- Standard object detection benchmark

---

## Acknowledgments

- [Perforated AI](https://perforatedai.com) for the dendritic optimization library
- [Ultralytics](https://ultralytics.com) for YOLOv11
- Pascal VOC dataset creators
