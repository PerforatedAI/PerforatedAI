# ESC-50 Audio Classification with Dendritic Networks

## Intro - Required

**Description:**

This hackathon project demonstrates the impact of adding artificial dendrites to a CNN14 audio classification model on the ESC-50 dataset. ESC-50 contains 2,000 environmental audio recordings across 50 classes (dog barking, glass breaking, rain, chainsaw, etc.). We compare a baseline CNN14 model against the same architecture enhanced with PerforatedAI dendrites to show measurable accuracy improvements.

**Team:**

Ravi Rai - [Your Position/Affiliation] - [Your Contact Info]

## Project Impact - Required

Environmental sound classification is crucial for numerous real-world applications including smart cities (detecting traffic patterns, emergency vehicles), wildlife monitoring (identifying species by calls), healthcare (detecting distress sounds, fall detection for elderly care), and assistive technology for hearing-impaired individuals. Improved accuracy in sound classification ensures better reliability in detecting critical sounds like alarms, glass breaking, or baby crying, which can trigger appropriate responses in automated safety and monitoring systems. Even small improvements in accuracy can significantly impact the reliability of these systems in high-stakes scenarios.

## Usage Instructions - Required

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- PerforatedAI library installed (`pip install -e .` from repository root)

### Dataset Download

Download the ESC-50 dataset:

```bash
cd Examples/hackathonProjects/ravi-audio-classification
mkdir -p data
cd data
curl -L -o esc50.zip https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip esc50.zip
mv ESC-50-master ESC-50
rm esc50.zip
cd ..
```

This will create `data/ESC-50/` with all audio files and metadata.

### Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

**Step 1: Preprocess the audio data (one-time, ~2-3 minutes):**

```bash
python preprocess.py
```

This converts audio files to mel-spectrograms and creates train/val/test splits.

**Step 2: Train baseline model (no dendrites):**

```bash
python train_baseline.py
```

Expected time: ~1-2 hours on M4 Mac (MPS) or CUDA GPU. Trains until early stopping.

**Step 3: Train with PerforatedAI dendrites:**

```bash
python train_perforatedai.py
```

Expected time: ~2-3 hours. Automatically adds dendrites when validation plateaus.

**Step 4: Generate comparison visualizations:**

```bash
python compare_results.py
```

This copies the PAI graph and generates the clean comparison graph for submission.

### Optional Arguments

Both training scripts support:
- `--batch_size 32` (default: 32)
- `--lr 0.0001` (learning rate, default: 0.0001)
- `--weight_decay 1e-5` (default: 1e-5)
- `--epochs 200` (max epochs, default: 200)
- `--patience 15` (early stopping patience for baseline only)

PAI training additional options:
- `--max_dendrites 5` (default: 5)

## Results - Required

Comparing baseline CNN14 to dendritic CNN14 on ESC-50 test set:

| Model        | Test Accuracy | Val Accuracy | Parameters | Dendrites | Epochs |
|--------------|--------------|--------------|------------|-----------|--------|
| Baseline CNN14     | 70.0%        | 80.94%       | 1,577,394  | 0         | 104    |
| Dendritic CNN14    | 73.5%        | 85.31%       | 4,733,392  | 2         | 200    |
| **Improvement** | **+3.5%** | **+4.37%**  | +3,155,998 | **+2**    | +96    |

**Remaining Error Reduction: 11.67%**

The error dropped from 30% to 26.5%, meaning dendrites eliminated 11.67% of the original error. This is a significant improvement on ESC-50, which is known to be a challenging environmental sound classification benchmark.

**Key Findings:**
- First dendrite added at epoch 76 when baseline plateaued at ~79% validation accuracy
- Validation accuracy jumped to ~86% after first dendrite addition
- Second dendrite attempted at epoch 128 but provided minimal benefit
- PAI automatically stopped training when no further improvement was detected
- Final test accuracy improved by 3.5 absolute percentage points with only 2 dendrites

## Raw Results Graph - Required

![PAI Training Output](./PAI_CNN14.png)

The graph shows the complete training progression:
- **Initial baseline training (epochs 0-76):** Model learns from 13% to ~79% validation accuracy
- **First vertical blue line (epoch 76):** First dendrite added, validation jumps to ~86%
- **Second vertical blue line (epoch 128):** Second dendrite attempted
- **Automatic stopping (epoch 137):** PAI determined convergence

The top-left subplot shows validation scores (orange/red lines) and training scores (green line). The clear performance jump after the first dendrite demonstrates the effectiveness of dendritic optimization.

## Clean Results Graph - Optional

![Accuracy Improvement](./Accuracy%20Improvement.png)

This visualization compares the final test accuracy, model size (parameters), and error reduction between baseline and dendritic models. The 11.67% error reduction demonstrates that dendrites eliminated over 10% of the remaining classification errors.

## Additional Files

### Code Structure

```
ravi-audio-classification/
├── config.py                   # Centralized configuration
├── preprocess.py              # Audio to mel-spectrogram conversion
├── train_baseline.py          # Train baseline CNN14
├── train_perforatedai.py      # Train CNN14 with dendrites
├── compare_results.py         # Generate comparison artifacts
├── requirements.txt           # Python dependencies
└── utils/
    ├── data_utils.py          # Dataset loading utilities
    ├── metrics.py             # Evaluation metrics
    └── pretrained_model.py    # CNN14 model architecture
```

### Model Architecture

The CNN14 architecture uses Sequential blocks (Conv2d + BatchNorm2d + ReLU + AvgPool2d) which are optimal for PerforatedAI integration. This design allows dendrites to be added cleanly to convolutional blocks while preserving batch normalization behavior.

### Dataset Details

- **Dataset:** ESC-50 (Environmental Sound Classification - 50 classes)
- **Size:** 2,000 audio clips (40 clips per class)
- **Duration:** 5 seconds per clip
- **Sample Rate:** 22,050 Hz
- **Features:** 128-band mel-spectrograms
- **Splits:** Train (1,280), Validation (320), Test (400)
- **Standard test fold:** Fold 5 (as per ESC-50 benchmark)

### Technical Notes

- Uses Adam optimizer with ReduceLROnPlateau scheduler
- Baseline uses early stopping (patience=15)
- PAI automatically manages dendrite addition based on validation plateau detection
- Dendrites use sigmoid activation and are initialized with small random weights (0.01 multiplier)
- Training performed on Apple M4 Mac using MPS (Metal Performance Shaders) acceleration

## Reproducibility

To exactly reproduce these results:
1. Use the same random seed (42) set in `config.py`
2. Use the same data split (test_fold=5 for ESC-50)
3. Run with the provided hyperparameters
4. Expected variance: ±1-2% due to random initialization and hardware differences

All hyperparameters are stored in `config.py` for easy reproduction.
