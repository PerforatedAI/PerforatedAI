# 🔧 PerforatedAI Integration Guide for YOLOv8

Quick reference for how dendritic optimization was integrated into YOLOv8.

## Integration Overview

```
YOLOv8 Model
     ↓
Initialize PAI
     ↓
Custom Training Loop
     ↓
Add Validation Scores to PAI Tracker
     ↓
Dendritic Optimization Happens Automatically!
     ↓
Model Restructuring with Dendrites
     ↓
Continue Training
```

## Step-by-Step Integration

### 1. Imports
```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
```

### 2. Configure PAI Settings (Before Model Init)
```python
# Set global parameters
GPA.pc.set_testing_dendrite_capacity(False)  # True for testing, False for real runs
GPA.pc.set_weight_decay_accepted(True)
GPA.pc.set_verbose(True)

# Set improvement thresholds
GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])

# Set dendrite configuration
GPA.pc.set_max_dendrites(5)  # Maximum dendrite sets to add
GPA.pc.set_candidate_weight_initialization_multiplier(0.01)
GPA.pc.set_pai_forward_function(torch.sigmoid)

# Disable Perforated Backpropagation (using gradient descent dendrites)
GPA.pc.set_perforated_backpropagation(False)
```

### 3. Initialize Model with PAI
```python
from ultralytics import YOLO

# Load YOLOv8
yolo = YOLO('yolov8n.pt')
model = yolo.model

# Initialize PAI
model = UPA.initialize_pai(model, save_name='PAI')

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 4. Setup Optimizer via PAI Tracker
```python
# Set optimizer and scheduler
GPA.pai_tracker.set_optimizer(torch.optim.Adam)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

# Configure optimizer arguments
optimArgs = {
    'params': model.parameters(),
    'lr': 0.001,
    'weight_decay': 0.0005
}

schedArgs = {
    'mode': 'max',  # Maximize validation score
    'patience': 5
}

# Setup through PAI
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

### 5. Training Loop Integration
```python
for epoch in range(num_epochs):
    # Train for one epoch (using YOLO's built-in training)
    results = yolo.train(
        data='coco128.yaml',
        epochs=1,
        resume=epoch > 0,
        # ... other args
    )

    # Validate
    metrics = yolo.val()
    map50_95 = float(metrics.box.map)

    # ** KEY STEP ** - Add validation score to PAI tracker
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        map50_95 * 100,  # Convert to percentage
        model
    )

    # Move model back to device after potential restructuring
    model = model.to(device)

    # If model was restructured (dendrites added), reset optimizer
    if restructured and not training_complete:
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimArgs, schedArgs
        )
        # Update YOLO's reference to the model
        yolo.model = model

    # Check if training is complete
    if training_complete:
        print("Training complete!")
        break
```

### 6. Optional: Track Additional Metrics
```python
# During training, optionally track training accuracy
train_acc = calculate_training_accuracy()
GPA.pai_tracker.add_extra_score(train_acc, 'train')

# Track test scores (if you have a separate test set)
test_map = calculate_test_map()
GPA.pai_tracker.add_test_score(test_map, 'Test mAP')
```

## How It Works

### Automatic Dendritic Optimization

1. **Initial Training**: Model trains normally
2. **Plateau Detection**: PAI detects when validation score plateaus
3. **Dendrite Addition**: New dendrite modules added to the network
4. **Continued Training**: Training continues with expanded model
5. **Repeat**: Process repeats until no improvement from dendrites

### What PAI Does Automatically

- ✅ Detects which layers benefit from dendrites
- ✅ Adds dendritic modules at optimal locations
- ✅ Manages multiple dendrite sets
- ✅ Handles optimizer state during restructuring
- ✅ Generates training graphs (PAI.png)
- ✅ Tracks metrics in CSV files
- ✅ Implements early stopping

### What You Need to Do

- ✅ Call `initialize_pai()` on your model
- ✅ Setup optimizer through PAI tracker
- ✅ Call `add_validation_score()` after each validation
- ✅ Handle model restructuring (reset optimizer)
- ✅ Check for training completion

## Key Differences from Standard YOLOv8 Training

### Standard YOLOv8
```python
model = YOLO('yolov8n.pt')
model.train(data='coco128.yaml', epochs=100)
metrics = model.val()
```

### With Dendritic Optimization
```python
# Initialize PAI
model = YOLO('yolov8n.pt')
pytorch_model = model.model
pytorch_model = UPA.initialize_pai(pytorch_model)

# Custom training loop with PAI integration
for epoch in range(max_epochs):
    # Train one epoch
    model.train(epochs=1, resume=epoch>0)

    # Validate and add score to PAI
    metrics = model.val()
    pytorch_model, restructured, complete = GPA.pai_tracker.add_validation_score(
        metrics.box.map * 100, pytorch_model
    )

    # Handle restructuring
    if restructured:
        model.model = pytorch_model
        # Reset optimizer...

    if complete:
        break
```

## Configuration Options

### Improvement Thresholds
```python
# More aggressive (adds dendrites quickly)
GPA.pc.set_improvement_threshold([0.01, 0.001, 0])

# Conservative (waits for clear plateau)
GPA.pc.set_improvement_threshold([0])

# Balanced (recommended)
GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])
```

### Dendrite Forward Functions
```python
# Sigmoid (smooth, recommended for most cases)
GPA.pc.set_pai_forward_function(torch.sigmoid)

# ReLU (sparse activations)
GPA.pc.set_pai_forward_function(torch.relu)

# Tanh (symmetric activations)
GPA.pc.set_pai_forward_function(torch.tanh)
```

### Weight Initialization
```python
# Smaller values (more conservative)
GPA.pc.set_candidate_weight_initialization_multiplier(0.001)

# Larger values (more aggressive)
GPA.pc.set_candidate_weight_initialization_multiplier(0.1)

# Recommended
GPA.pc.set_candidate_weight_initialization_multiplier(0.01)
```

## Validation Checklist

Your integration is correct if:

- ✅ `PAI/PAI.png` is generated
- ✅ Graph shows multiple colored lines
- ✅ `PAIbest_test_scores.csv` has multiple rows
- ✅ Dendrite count increases over time
- ✅ Validation score improves with dendrites

Your integration has issues if:

- ❌ No PAI folder created
- ❌ Graph shows flat line
- ❌ Only one row in CSV
- ❌ Dendrite count stays at 0
- ❌ Errors about model restructuring

## Troubleshooting Integration Issues

### Issue: No dendrites added
**Cause**: Max dendrites set to 0 or mode is wrong
**Fix**:
```python
GPA.pc.set_max_dendrites(5)
GPA.pc.set_perforated_backpropagation(False)
```

### Issue: Optimizer errors after restructuring
**Cause**: Not resetting optimizer properly
**Fix**:
```python
if restructured and not training_complete:
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
        model, optimArgs, schedArgs
    )
```

### Issue: Model not improving
**Cause**: Learning rate too high/low for dendrites
**Fix**: Use PAI's automatic LR sweep or try lower LR

### Issue: CUDA out of memory
**Cause**: Dendrites increase model size
**Fix**:
- Reduce batch size
- Reduce image size
- Use gradient checkpointing

## Advanced: W&B Integration

```python
import wandb

# Initialize W&B
wandb.init(project="Dendritic YOLOv8", config=config)

# In training loop
wandb.log({
    "epoch": epoch,
    "mAP50-95": map50_95,
    "params": UPA.count_params(model),
    "dendrites": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
})

# Log architecture changes
if restructured:
    wandb.log({
        "arch_change": epoch,
        "new_dendrite_count": GPA.pai_tracker.member_vars["num_dendrites_added"]
    })
```

## Performance Tips

1. **Start small**: Test with COCO-128 before full COCO
2. **Use W&B sweeps**: Find optimal hyperparameters
3. **Monitor PAI graph**: Should show clear improvement cycles
4. **Patience matters**: Give dendrites time to train
5. **GPU memory**: Monitor usage, reduce batch if needed

## Files Generated

```
PAI/
├── PAI.png                           # Main results graph (REQUIRED)
├── PAIbest_test_scores.csv           # Metrics per dendrite cycle
├── PAI_validation_scores.csv         # All validation scores
└── PAI_extra_scores.csv              # Additional tracked metrics
```

## Next Steps

1. Review the generated `PAI.png`
2. Check `PAIbest_test_scores.csv` for metrics
3. Calculate Remaining Error Reduction
4. Update README with your results
5. Submit hackathon PR!

---

**Reference Implementation**: `train_yolov8_dendritic.py`

**Questions?** Check the [PerforatedAI Discord](https://discord.gg/Fgw3FG3Hzt)
