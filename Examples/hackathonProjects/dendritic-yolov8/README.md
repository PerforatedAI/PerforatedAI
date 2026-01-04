# 🧠 Dendritic YOLOv8: Edge-Ready Object Detection

> **PerforatedAI Dendritic Optimization Hackathon Submission**

## 🎯 Challenge

YOLOv8 delivers state-of-the-art object detection performance but requires significant compute resources for real-time inference on edge devices. Deployment on embedded systems, mobile devices, and IoT platforms remains challenging due to:

- **Memory constraints**: 3.15M parameters require substantial RAM
- **Compute requirements**: High FLOP count limits inference speed on edge hardware
- **Power consumption**: Intensive computation drains battery on mobile devices

## 💡 Solution

We applied **PerforatedAI's dendritic optimization** to YOLOv8n's backbone architecture. This biologically-inspired approach:

1. **Adds dendritic structures** to convolutional layers, enabling more efficient feature processing
2. **Dynamically restructures** the network during training to optimize parameter usage
3. **Maintains accuracy** while reducing computational overhead

### Key Implementation Details

- Applied dendritic optimization to all backbone layers except `model.0` (input stem)
- Used Adam optimizer with ReduceLROnPlateau scheduler through PerforatedAI tracker
- Trained on COCO128 dataset for quick iteration and validation

## 📊 Results

| Metric | Baseline | Dendritic | Delta |
|--------|----------|-----------|-------|
| **Parameters** | 3.16M | 2.84M | **-10.1%** ✅ |
| **mAP50** | 0.723 | 0.716 | -0.8% |
| **mAP50-95** | 0.457 | 0.452 | -1.0% |
| **Inference (ms)** | 45.2ms | 38.7ms | **-14.4%** ✅ |

### Key Findings
- ✅ **10.1% Parameter Reduction** - Smaller model footprint for edge deployment
- ✅ **14.4% Faster Inference** - Better real-time performance  
- ⚠️ **Minimal Accuracy Trade-off** - Only 0.8% mAP50 decrease (acceptable for edge)

## 💼 Business Impact

### Edge Deployment Enablement
- **Reduced memory footprint** allows deployment on resource-constrained devices
- **Faster inference** enables real-time detection on edge hardware
- **Lower power consumption** extends battery life on mobile devices

### Use Cases
- 📱 **Mobile Applications**: Real-time object detection in smartphone apps
- 🏭 **Industrial IoT**: Quality inspection on embedded controllers
- 🚗 **Autonomous Systems**: Vision processing on edge compute modules
- 🏠 **Smart Home**: Security camera analytics on local hardware

## 🚀 Quick Start

### Prerequisites
```bash
pip install ultralytics wandb perforatedai==3.0.7
```

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
