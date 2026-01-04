# Emotion Recognition from Speech with Dendritic Optimization

## Intro - Required

This project applies **PerforatedAI's Dendritic Optimization** to emotion recognition from speech audio. We use a CNN architecture to classify 8 emotions from Mel spectrograms extracted from the RAVDESS dataset.

**Key Innovation**: Adding artificial dendrites to audio classification demonstrates that dendritic optimization benefits extend beyond traditional image classification to the crucial domain of affective computing.

### Team

**Kamalesh** - Developer | Hackathon Participant

---

## Project Impact - Required

Emotion recognition from speech has critical real-world applications:

- **Mental Health Monitoring**: Early detection of depression, anxiety, and emotional distress through voice analysis can enable timely interventions
- **Accessibility**: Helps neurodivergent individuals who struggle to interpret vocal emotional cues
- **Customer Service**: Automated detection of frustrated or angry customers enables better service routing
- **Human-Computer Interaction**: More empathetic AI assistants that respond appropriately to user emotional state

Improving the accuracy of emotion recognition even by a few percentage points can mean the difference between correctly identifying someone in distress versus missing critical warning signs. **Every reduction in error rate has direct humanitarian impact.**

---

## Usage Instructions - Required

### Installation

```bash
# Clone the repository
git clone https://github.com/PerforatedAI/PerforatedAI.git
cd PerforatedAI/Examples/hackathonProjects/emotion-recognition

# Install dependencies
pip install -r requirements.txt

# Install PerforatedAI
cd ../../../
pip install -e .
cd Examples/hackathonProjects/emotion-recognition

# Download RAVDESS dataset
python download_data.py
```

### Training with Dendrites + W&B Logging

```bash
# Standard training with PerforatedAI dendrites and W&B logging
python main.py --data_dir ./data/ravdess --epochs 50

# Run W&B hyperparameter sweep
python main.py --sweep --count 10

# Disable W&B logging
python main.py --data_dir ./data/ravdess --no-wandb
```

---

## Results - Required

This project demonstrates that **Dendritic Optimization improves emotion recognition accuracy** on the RAVDESS dataset. Dendrites are **dynamically added** during training based on improvement thresholds.

### Dynamic Dendrite Addition

| Switch | Epoch | Parameters | Change |
|--------|-------|------------|--------|
| 0 | 0 | 422,728 | Initial model |
| 2 | 20 | 845,112 | **+1 Dendrite added** |
| 4 | 27 | 1,269,344 | **+1 Dendrite added** |

### Accuracy Comparison

| Model | Param Count | Best Validation Accuracy | Notes |
|-------|-------------|--------------------------|-------|
| Traditional CNN | 422,728 | 50.00% | Baseline without dendrites |
| Dendritic CNN (1 dendrite) | 845,112 | **52.59%** | With PerforatedAI optimization |
| Dendritic CNN (2 dendrites) | 1,269,344 | 49.14% | Additional capacity testing |

### Remaining Error Reduction

$$RER = \frac{52.59 - 50.00}{100 - 50.00} \times 100 = \textbf{5.18\%}$$

The dendritic optimization reduced the remaining error by **5.18%**, demonstrating that artificial dendrites can improve emotion recognition from speech spectrograms.

---

## Raw Results Graph - Required

The PerforatedAI library automatically generates this graph during training, saved to `PAI/PAI.png`.

![PerforatedAI Results Graph](./PAI/PAI.png)

---

## Weights and Biases Sweep Report - Optional

All training metrics, including dynamic dendrite additions, are logged to Weights & Biases:

[**View W&B Dashboard →**](https://wandb.ai/kamaleshgehlot0022-chennai-institute-of-technology/emotion-recognition-pai)

Tracked metrics include:
- Training/Validation accuracy per epoch
- Loss curves
- Dendrite addition events  
- Parameter count over time
- Learning rate schedule

---

## Additional Files

- `main.py` - Main training script with PerforatedAI + W&B integration
- `model.py` - CNN and ResNet model architectures
- `dataset.py` - RAVDESS dataset loader with spectrogram conversion
- `download_data.py` - Helper script to download RAVDESS dataset
- `requirements.txt` - Python dependencies

### Dataset

The [RAVDESS dataset](https://zenodo.org/record/1188976) contains 1,440 audio files from 24 actors expressing 8 emotions:
- Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

### Architecture

```
Input: Mel Spectrogram (1, 128, 128)
    ↓
Conv Block 1: Conv2d(1→32) → BatchNorm → ReLU → MaxPool
    ↓
Conv Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool
    ↓
Conv Block 3: Conv2d(64→128) → BatchNorm → ReLU → MaxPool
    ↓
Conv Block 4: Conv2d(128→256) → BatchNorm → ReLU → MaxPool
    ↓
Global Average Pooling
    ↓
FC: 256 → 128 → 8 (emotions)
           ↓
    🌳 Dendrites added dynamically here!
```
