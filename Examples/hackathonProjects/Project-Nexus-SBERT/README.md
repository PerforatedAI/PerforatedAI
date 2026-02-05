# Project NEXUS: Dendritic SBERT Study

**Investigating Perforated Backpropagation on pre-optimized NLP architectures.**

![PAI Training Graph](PAI/PAI.png)

## Findings Summary

We applied dendritic optimization to Sentence-BERT (all-MiniLM-L6-v2) on the STS Benchmark. Our 22-experiment study reveals that the baseline architecture is already Pareto-optimal for this task.

| Configuration | Best Val Spearman | Epochs to Best | Status |
|---------------|-------------------|----------------|--------|
| **Baseline** | **89.18%** | 0 | Immediate convergence |
| **Dendritic** | 89.02% | 15 | No significant gain |

**Conclusion:** The standard SBERT architecture requires no additional capacity for STS. Dendritic optimization is best reserved for underfitting models or domain adaptation tasks.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline
python src/train_nexus.py --epochs 10 --batch_size 32 --lr 2e-5 --save_dir experiments/baseline

# Run dendritic
python src/train_nexus.py --use_dendrites --epochs 10 --batch_size 32 --lr 2e-5 --warmup_epochs 4 --save_dir experiments/dendritic
```

## Technical Configuration
*   **Model:** `all-MiniLM-L6-v2`
*   **Target:** Final Dense Layer
*   **Setup:** Learning Rate: 2e-5 | Batch: 32 | Warmup: 4

**[View W&B Experiments](https://wandb.ai/aakanksha-singh0205-kj-somaiya-school-of-engineering/PerforatedAI-Examples_hackathonProjects_Project-Nexus-SBERT_src)**

## Repository Structure
*   `CASE_STUDY.md`: Analysis of results.
*   `src/`: Training and evaluation scripts.
*   `PAI/`: Visualization artifacts.
*   `sweep_grid.yaml`: Hyperparameter definitions.

**Team:** Aakanksha Singh & Mihir Phalke (K.J. Somaiya College of Engineering)  
**License:** MIT
