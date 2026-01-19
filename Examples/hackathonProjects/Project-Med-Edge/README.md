# Project Med-Edge: Dendritic Optimization for Portable Dermatology

**Enabling skin cancer screening on low-cost microcontrollers through intelligent neural architecture growth**

---

## The Challenge

Skin cancer is the most common cancer globally, yet dermatologists are scarce in rural and low-resource settings. While AI models can assist with diagnosis, deploying them requires expensive hardware and cloud connectivity—both unavailable in remote clinics.

**The question:** Can we create a medical-grade skin lesion classifier that runs on ultra-low-cost microcontrollers while maintaining diagnostic accuracy?

---

## The Approach

We developed **DermoNet-Edge**, a Micro-CNN designed for the DermaMNIST dataset (7-class skin lesion classification, 28×28 images). Using Perforated AI's dendritic optimization, we improved classification accuracy while maintaining edge-deployable model sizes.

**Key Innovation:** We used a regularized architecture with two dropout layers (matching the MNIST example) and proper PAI configuration to achieve clean training curves with minimal overfitting.

---

## The Results

| Model | Val Accuracy | Parameters | Error Reduction |
|-------|--------------|------------|-----------------|
| **Baseline** | 76.57% | 15,229 | - |
| **Dendritic (PAI)** | **77.87%** | 63,201 | **5.55%** |

**Remaining Error Reduction (RER):** 5.55%  
**Dendrites Added:** 3  
**Overfitting Gap:** 0.22% (Train: 78.09%, Val: 77.87%)

### Perforated AI Training Graph

![PAI.png - Dendritic training visualization](./PAI/PAI.png)

*The graph shows: (1) Training progression with dendrite additions (vertical lines), (2) Learning rate decay with resets after dendrite additions (sawtooth pattern), (3) Layer-wise dendrite evaluations, and (4) Training times.*

---

## Business Impact

### Hardware Enablement
- **Target hardware:** ESP32-S3 or similar ARM Cortex-M7
- **Memory footprint:** ~247KB (63k parameters × 4 bytes)
- **Inference time:** <200ms on embedded CPU

### Use Case Unlocked
> "A village health worker in rural areas can screen patients for skin cancer using a portable device. The AI runs entirely on-device—no internet, no cloud, no privacy concerns."

### Economic Impact
- **Cost per screening:** $0.00 (vs $50-200 for specialist consultation)
- **Scalability:** Deployable to rural health centers globally
- **Accessibility:** Works offline in areas with zero connectivity

---

## Implementation Experience

**What worked:**
- Using **two dropout layers** (0.25 after conv, 0.5 after fc1) following the MNIST example
- **No data augmentation** for training (same as MNIST example) to avoid Val > Train
- **Proper PAI optimizer/scheduler setup** using `GPA.pai_tracker.setup_optimizer()`

**Time investment:**
- Initial setup: 2 hours
- Baseline training: 10 minutes
- Dendritic training: 15 minutes
- Total: ~3 hours from idea to results

**Key insight:**  
> "PAI not only improved accuracy but also reduced overfitting. The baseline had a 3.78% train-val gap, while the dendritic model had only 0.22% gap—showing that dendrites learned generalizable features."

---

## Technical Specifications

**Dataset:** DermaMNIST (HAM10000 subset)
- 10,015 images (7,007 train / 1,003 val / 2,005 test)
- 7 classes: melanoma, nevus, basal cell carcinoma, etc.
- Resolution: 28×28 RGB

**Architecture:** DermoNet-Edge
```python
Conv2d(3 → 8) + BN + ReLU + MaxPool
Conv2d(8 → 16) + BN + ReLU + MaxPool  
Conv2d(16 → 32) + BN + ReLU + MaxPool
Dropout(0.25)  # After conv layers
Linear(288 → 32) + ReLU
Dropout(0.5)   # After fc1 (dendritic layer)
Linear(32 → 7)
```

**PAI Configuration:**
```python
GPA.pc.set_improvement_threshold([0.01, 0.001, 0.0001, 0])
GPA.pc.set_candidate_weight_initialization_multiplier(0.01)
GPA.pc.set_pai_forward_function(torch.sigmoid)
GPA.pc.set_n_epochs_to_switch(8)
GPA.pc.set_max_dendrites(8)

# Proper optimizer setup (like MNIST example)
GPA.pai_tracker.set_optimizer(optim.Adam)
GPA.pai_tracker.set_scheduler(StepLR)
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

---

## Reproducibility

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline (Standard CNN)
python train.py

# Train with dendrites (Perforated AI)
python train.py --dendrites
```

### Hyperparameter Sweep (Bonus)
To optimize hyperparameters using W&B Sweeps:

1. Initialize the sweep:
```bash
wandb sweep sweep_config.yaml
```

2. Run the sweep agent (replace `SWEEP_ID` with ID from step 1):
```bash
wandb agent SWEEP_ID --count 20
```
This will run 20 trials exploring learning rates, dropout, and batch sizes to find the optimal configuration. The results are tracked in your W&B dashboard.

### W&B Tracking
All experiments tracked at:  
[https://wandb.ai/aakanksha-singh0205-kj-somaiya-school-of-engineering/Project-Med-Edge](https://wandb.ai/aakanksha-singh0205-kj-somaiya-school-of-engineering/Project-Med-Edge)

---

## Repository Structure

```
Project-Med-Edge/
├── train.py                 # Unified training script
├── src/
│   ├── model.py            # DermoNet-Edge architecture
│   └── dataset.py          # MedMNIST data loading
├── configs/
│   └── default.yaml        # Hyperparameters
├── PAI/
│   └── PAI.png             # Auto-generated training graph
├── sweep_config.yaml       # W&B sweep configuration
├── checkpoints/            # Saved models
└── README.md               # This file
```

---

## Key Findings

1. **Two dropout layers prevent overfitting while maintaining Train > Val**  
   Baseline gap: 3.78% → Dendritic gap: 0.22%

2. **PAI improves generalization, not just accuracy**  
   Dendrites learned features that transfer better to validation data

3. **Proper optimizer/scheduler setup is critical**  
   Using `setup_optimizer()` instead of direct creation enables LR decay visualization

---

## Future Work

- **Robustness testing:** Evaluate on noisy/blurry images (simulating cheap cameras)
- **Quantization:** INT8 deployment for further memory reduction
- **Multi-task learning:** Extend to other MedMNIST datasets (PathMNIST, OrganMNIST)
- **Clinical validation:** Partner with dermatologists for real-world testing

---

## Team

**Aakanksha Singh**  
Third Year Student

**Mihir Phalke**  
Third Year Student

---

## Acknowledgments

- **Perforated AI** for Dendrites 2.0 technology
- **MedMNIST** team for the curated medical imaging dataset
- **Hackathon organizers** for the opportunity

---

## License

MIT License - See LICENSE file for details
