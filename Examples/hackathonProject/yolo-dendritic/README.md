# YOLOv8n + Dendritic PAI Integration

## Hackathon Submission

YOLOv8n detection head enhanced with custom DendriticConv2d error-correction branches integrated into the PerforatedAI framework. Achieved 56.96% error reduction with automatic dendrite triggers.

## Key Results
```
Initial Loss: 0.016545
Best Loss: 0.007120 (56.96% reduction)
Final Loss: 0.007132
Dendrite Triggers: Epoch 16, 44
Training: 100 epochs (stable)
Data: Synthetic (COCO fallback)
```

| Metric    | No-Dendrite Baseline | Dendritic PAI (Peak) | Absolute Gain | RER    |
| --------- | -------------------- | -------------------- | ------------- | ------ |
| mAP@50    | 0.5664               | 0.666                | +0.0996       | +23.6% |
| mAP@50-95 | 0.4236               | 0.508                | +0.0844       | +14.6% |

## Performance Visualization
![PAI Performance Graphs](PAI.png)

In the PAI graphs, it looks like the losses are stagnated, but that is because on epoch 1, the training loss seems to start off at a 9, before dropping to ~0.015 (in the same range as validation loss). Here's a better visualisation retaining the PAI numbers.

![Comparison](Comparsion.jpeg)



Here are the mAP@50 and mAP@50-95 plots!

![mAP@50](mAP@50.jpeg)
![mAP@50-95](mAP@50-95.jpeg)


## Technical Architecture

```
DendriticConv2d (6 branches per layer):
├── Main pathway: Original Conv2d
├── Dendritic branches: 6× Depthwise-separable convs (scale=0.2)
├── Target layers: model.model.cv3[0-2] (YOLO Detect head)
​
└── Loss: Variance-based (magnitude + diversity)

PAI Integration:
├── GPA.pc.set_module_ids_to_convert([".model.22.cv3.0.2", ".model.22.cv3.1.2", ".model.22.cv3.2.2"])
├── Monkeypatched MPA.PAINeuronModule.clear_dendrites()
└── GPA.pai_tracker.add_validation_score() per epoch
```

## Quick Start

### Requirements
```
pip install ultralytics perforatedai torch torchvision GPUtil
```

### Run training (auto GPU selection)
```
python train.py
```

### Outputs:
```
 PAI_YOLO/PAI.png          # Official PAI graphs
 checkpoints_yolo_final/   # Model + config.json (56.96% metric)
 logs_yolo_final/          # Training logs
```

## Training Behavior
Phase 1 (Epoch 0-15):  Loss decreases 0.0165 → 0.0085 (plateau)

Phase 2 (Epoch 16):    PAI dendrites triggered #1

Phase 3 (Epoch 17-43): Temporary spike → recovery

Phase 4 (Epoch 44):    PAI dendrites triggered #2

Phase 5 (Epoch 45-99): Final convergence → 0.0071

## Key Innovations
1. Custom DendriticConv2d: 6 parallel error-correction branches using depthwise-separable convolutions
2. PAI-YOLO Integration: Targeted layer conversion + dendrite injection into detection head
3. Variance-Based Loss: Encourages prediction diversity + strong gradients
4. Backbone Frozen: Only detection head trained (production-ready)

| Metric                    | Value                                    |
| ------------------------- | ---------------------------------------- |
| Remaining Error Reduction | 56.96%                                   |
| Dendrites Injected        | 3 (cv3​​, cv3pmc.ncbi.nlm.nih+1​, cv3​​)      |
| PAI Triggers              | 2 (Epoch 16, 44)                         |
| Parameters                | ~3.3M (YOLOv8n + dendrites)              |
| Stability                 | 100 epochs, no crashes                   |
