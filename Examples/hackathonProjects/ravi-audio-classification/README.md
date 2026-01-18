# ESC-50 Audio Classification with Dendrites

## Intro

Description:

This project shows an improvement when adding artificial dendrites to a CNN14 audio classification model on the ESC-50 dataset. ESC-50 contains 2,000 environmental audio recordings across 50 classes (dog barking, glass breaking, rain, chainsaw, etc.). We compare a baseline CNN14 model against the same architecture enhanced with PerforatedAI dendrites to show measurable accuracy improvements.

Team:

Ravi Rai - ML/AI Engineer - https://www.linkedin.com/in/ravibrai/

## Project Impact

Environmental sound classification is useful for numerous real-world applications including smart cities (detecting traffic patterns, emergency vehicles), wildlife monitoring (identifying species by calls), healthcare (detecting distress sounds, fall detection for elderly care), and assistive technology for hearing-impaired individuals. Improved accuracy in sound classification ensures better reliability in detecting critical sounds like alarms, glass breaking, or baby crying, which can trigger appropriate responses in automated safety and monitoring systems. Even small improvements in accuracy can significantly impact the reliability of these systems in high-stakes scenarios. A 15% error reduction means 15% fewer missed detections or false alarms in safety-critical applications.

## Usage Instructions - Required

Installation:

```bash
cd Examples/hackathonProjects/ravi-audio-classification
pip install -r requirements.txt
```

Dataset Download:

```bash
mkdir -p data && cd data
curl -L -o esc50.zip https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip esc50.zip && mv ESC-50-master ESC-50 && rm esc50.zip
cd ..
```

Verify Setup (optional):

```bash
python verify_setup.py
```

Run:

```bash
# Step 1: Preprocess audio to mel-spectrograms (~2 minutes)
python preprocess.py

# Step 2: Train baseline model (~1-2 hours)
python train_baseline.py

# Step 3: Train with dendrites (~2-3 hours)
python train_perforatedai.py

# Step 4: Generate comparison visualizations
python compare_results.py
```

## Results - Required

This ESC-50 project shows that Dendritic Optimization can improve accuracy on environmental sound classification. Comparing the best traditional model to the best dendritic model:

| Model        | Test Accuracy | Validation Accuracy | Parameters | Notes |
|--------------|---------------|---------------------|------------|-------|
| Traditional  | 70.50%        | 80.31%              | 1,577,394  | Baseline CNN14, early stopping at epoch 116 |
| Dendritic    | 75.00%        | 85.00%              | 4,733,392  | CNN14 + 2 dendrites, 200 epochs |

This provides a **Remaining Error Reduction of 15.25%**.

The error dropped from 29.5% to 25.0%, meaning dendrites eliminated 15.25% of the original classification errors. This is a significant improvement on ESC-50, which is known to be a challenging environmental sound classification benchmark with 50 classes.

## Raw Results Graph - Required

![PAI Training Output](./PAI_CNN14.png)

The graph shows the complete training progression:
- **Initial baseline training (epochs 0-70):** Model learns from ~13% to ~76% validation accuracy
- **First vertical blue line (~epoch 70):** First dendrite added, validation improves to ~85%
- **Second vertical blue line (~epoch 130):** Second dendrite attempted
- **Training continues to epoch 200:** PAI completes training cycle

The top-left subplot shows validation scores (orange/blue lines) and training scores (green line). The clear performance jump after the first dendrite demonstrates the effectiveness of dendritic optimization.

## Clean Results Graph - Optional

![Accuracy Improvement](./Accuracy%20Improvement.png)

This visualization compares the final test accuracy and error reduction between baseline and dendritic models. The 15.25% error reduction demonstrates that dendrites eliminated over 15% of the remaining classification errors.

## Additional Files

Code Structure:

```
ravi-audio-classification/
├── config.py                   # Centralized configuration
├── preprocess.py               # Audio to mel-spectrogram conversion
├── train_baseline.py           # Train baseline CNN14
├── train_perforatedai.py       # Train CNN14 with dendrites
├── compare_results.py          # Generate comparison artifacts
├── verify_setup.py             # Environment verification script
├── requirements.txt            # Python dependencies
├── PAI_CNN14/                  # PAI output files and graphs
│   └── PAI_CNN14.png           # Required raw results graph
└── utils/
    ├── data_utils.py           # Dataset loading utilities
    ├── metrics.py              # Evaluation metrics
    └── pretrained_model.py     # CNN14 model architecture
```

Model Architecture:

The CNN14 architecture uses Sequential blocks (Conv2d + BatchNorm2d + ReLU + AvgPool2d) which are optimal for PerforatedAI integration. This design allows dendrites to be added cleanly to convolutional blocks while preserving batch normalization behavior.

Dataset Details:

- **Dataset:** ESC-50 (Environmental Sound Classification - 50 classes)
- **Size:** 2,000 audio clips (40 clips per class)
- **Duration:** 5 seconds per clip
- **Sample Rate:** 22,050 Hz
- **Features:** 128-band mel-spectrograms
- **Splits:** Train (1,280), Validation (320), Test (400)
- **Standard test fold:** Fold 5 (as per ESC-50 benchmark)

Technical Notes:

- Uses Adam optimizer with ReduceLROnPlateau scheduler
- Baseline uses early stopping (patience=15)
- PAI automatically manages dendrite addition based on validation plateau detection
- Dendrites use sigmoid activation and are initialized with small random weights (0.01 multiplier)
- Random seed 42 used throughout for reproducibility
- Training performed on Apple M4 Mac using MPS (Metal Performance Shaders) acceleration
