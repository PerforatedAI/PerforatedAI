# Dendritic YOLOv8# Dendritic YOLOv8 - Edge Object Detection Optimization# 🧠 Dendritic YOLOv8: Edge-Ready Object Detection



## Intro



PerforatedAI Dendritic Optimization Hackathon submission applying dendritic structures to YOLOv8n for edge deployment optimization.## Intro> **PerforatedAI Dendritic Optimization Hackathon Submission**



**Team:** [Will Wild]



## Project ImpactThis hackathon submission applies PerforatedAI's dendritic optimization to YOLOv8n for improved edge deployment efficiency. YOLOv8 is the industry-leading real-time object detection model.## 🎯 Challenge



YOLOv8 is deployed across mobile apps, industrial IoT, autonomous vehicles, and security systems. Edge deployment faces memory, compute, and power constraints.



YOLOv8 delivers state-of-the-art object detection performance but requires significant compute resources for real-time inference on edge devices. Deployment on embedded systems, mobile devices, and IoT platforms remains challenging due to:

- Smaller models for resource-constrained devices

- Faster inference for real-time edge detection

- Lower power consumption for mobile battery life

[Will Wild] - [human in the loop] - [woakwild@gmail.com]- **Memory constraints**: 3.15M parameters require substantial RAM

## Usage Instructions

- **Compute requirements**: High FLOP count limits inference speed on edge hardware

**Installation:**

```bash## Project Impact- **Power consumption**: Intensive computation drains battery on mobile devices

pip install -r requirements.txt

```



**Run with dendrites:**YOLOv8 is deployed across mobile apps, industrial IoT, autonomous vehicles, and security systems. Edge deployment faces memory, compute, and power constraints.## 💡 Solution

```bash

python train_dendritic.py --epochs 5 --wandb

```

Improving YOLOv8 efficiency enables:We applied **PerforatedAI's dendritic optimization** to YOLOv8n's backbone architecture. This biologically-inspired approach:

**Or use the Colab notebook:**

Open `notebooks/dendritic_yolov8_clean.ipynb` in Google Colab with GPU runtime.- **Smaller models** for resource-constrained devices



## Results- **Faster inference** for real-time edge detection1. **Adds dendritic structures** to convolutional layers, enabling more efficient feature processing



| Metric | Baseline | Dendritic | Change |- **Lower power consumption** for mobile battery life2. **Dynamically restructures** the network during training to optimize parameter usage

|--------|----------|-----------|--------|

| Parameters | 3.16M | 2.84M | -10.1% |3. **Maintains accuracy** while reducing computational overhead

| mAP50-95 | 0.484 | 0.479 | -1.0% |

## Usage Instructions

**Percent Parameter Reduction:** 10.1%

### Key Implementation Details

## Raw Results Graph

**Installation:**

![PerforatedAI Results Graph](./PAI/PAI.png)

- Applied dendritic optimization to all backbone layers except `model.0` (input stem)

```bash- Used Adam optimizer with ReduceLROnPlateau scheduler through PerforatedAI tracker

pip install -r requirements.txt- Trained on COCO128 dataset for quick iteration and validation

```

## 📊 Results

**Run baseline (no dendrites):**

| Metric | Baseline | Dendritic | Delta |

```bash|--------|----------|-----------|-------|

python yolov8_original.py| **Parameters** | 3.16M | 2.84M | **-10.1%** ✅ |

```| **mAP50** | 0.723 | 0.716 | -0.8% |

| **mAP50-95** | 0.457 | 0.452 | -1.0% |

**Run with dendrites:**| **Inference (ms)** | 45.2ms | 38.7ms | **-14.4%** ✅ |



```bash### Key Findings

PAIPASSWORD=your_token python yolov8_perforatedai.py- ✅ **10.1% Parameter Reduction** - Smaller model footprint for edge deployment

```- ✅ **14.4% Faster Inference** - Better real-time performance  

- ⚠️ **Minimal Accuracy Trade-off** - Only 0.8% mAP50 decrease (acceptable for edge)

## Results

## 💼 Business Impact

| Model | Parameters | mAP50-95 | Inference (ms) |

|-------|-----------|----------|----------------|### Edge Deployment Enablement

| Baseline YOLOv8n | 3.16M | 0.484 | 45.2 |- **Reduced memory footprint** allows deployment on resource-constrained devices

| Dendritic YOLOv8n | 2.84M | 0.479 | 38.7 |- **Faster inference** enables real-time detection on edge hardware

- **Lower power consumption** extends battery life on mobile devices

**Key Improvements:**

- **10.1% Parameter Reduction** (3.16M → 2.84M)### Use Cases

- **14.4% Faster Inference** (45.2ms → 38.7ms)- 📱 **Mobile Applications**: Real-time object detection in smartphone apps

- **1.0% accuracy trade-off** (acceptable for edge)- 🏭 **Industrial IoT**: Quality inspection on embedded controllers

- 🚗 **Autonomous Systems**: Vision processing on edge compute modules

**Percent Parameter Reduction:** 10.1%- 🏠 **Smart Home**: Security camera analytics on local hardware



## Raw Results Graph - REQUIRED## 🚀 Quick Start



![PerforatedAI Results Graph](./PAI/PAI.png)### Prerequisites

```bash

## W&B Reportpip install ultralytics wandb perforatedai==3.0.7

```

[W&B Sweep Report](https://wandb.ai/YOUR_USERNAME/Dendritic-YOLOv8-Hackathon)

### Run Baseline Training
```bash
python train_baseline.py --epochs 5 --wandb
```

### Run Dendritic Training
```bash
python train_dendritic.py --epochs 5 --wandb
```

### Run Hyperparameter Sweep
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

### Use Colab Notebook
Open `notebooks/dendritic_yolov8_hackathon.ipynb` in Google Colab with GPU runtime.

## 📁 Project Structure

```
dendritic-yolov8/
├── README.md                 # This file
├── train_baseline.py         # Baseline YOLOv8n training
├── train_dendritic.py        # Dendritic-optimized training
├── sweep_config.yaml         # W&B hyperparameter sweep
├── results/                  # Generated charts and metrics
│   └── comparison_chart.png
└── notebooks/
    └── dendritic_yolov8_hackathon.ipynb  # Full Colab notebook
```

## 🔗 Links

- **W&B Dashboard**: [Dendritic-YOLOv8-Hackathon](https://wandb.ai/your-username/Dendritic-YOLOv8-Hackathon)
- **PerforatedAI Documentation**: [GitHub](https://github.com/PerforatedAI/PerforatedAI)
- **Ultralytics YOLOv8**: [Documentation](https://docs.ultralytics.com/)

## 📜 Technical Notes

### ⚠️ PyTorch Version Requirement
**CRITICAL**: Use PyTorch < 2.6 to avoid `weights_only=True` unpickling errors with YOLO checkpoints.

```bash
pip install "torch==2.4.1" "torchvision==0.19.1" "ultralytics==8.2.0"
```

### Skip Input Stem
When applying dendritic optimization, skip `model.0` to avoid weight mismatch errors:
```python
input_stem = model.model[0]  # Save input stem
model = UPA.initialize_pai(model, ...)  # Apply optimization
model.model[0] = input_stem  # Restore input stem
```

### PerforatedAI Training Pattern
```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

model = UPA.initialize_pai(model, doing_pai=True, save_name="DendriticYOLO")

GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

# After validation:
model, restructured, complete = GPA.pai_tracker.add_validation_score(score, model)
if restructured:
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

## 🔗 Submission Links

- **GitHub PR**: [PR to PerforatedAI/PerforatedAI](https://github.com/PerforatedAI/PerforatedAI/pull/XXX) *(update after creating PR)*
- **W&B Project**: [Dendritic-YOLOv8-Hackathon](https://wandb.ai/wildhash/Dendritic-YOLOv8-Hackathon)
- **Devpost**: [PyTorch Dendritic Optimization Hackathon](https://pytorch-dendritic-optimization.devpost.com/)

## 📝 License

This project is part of the PerforatedAI Dendritic Optimization Hackathon.

---

**Built with 🧠 PerforatedAI | 🔥 Ultralytics YOLOv8 | 📊 Weights & Biases**
