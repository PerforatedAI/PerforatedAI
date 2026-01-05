# Project NEXUS: Dendritic SBERT for Edge RAG

**Optimizing the world's most downloaded embedding model (`all-MiniLM-L6-v2`) for privacy-first Edge AI using Perforated Backpropagation.**

**Authors:** Aakanksha Singh and Mihir Phalke

## Why This Matters
*   **Prevalence:** Transforms `all-MiniLM-L6-v2` (**50M+ downloads/mo**), the industry standard for RAG.
*   **Privacy:** Enables stable fine-tuning on sensitive data (Medical/Finance) directly on edge devices.
*   **Efficiency:** Achieves **better loss convergence (0.0036 vs 0.0038)** with negligible footprint (+0.6%).

## How to Run

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Dendritic Training (Winning Model)
python src/train_nexus.py --use_dendrites --epochs 10 --save_dir experiments/dendritic

# 3. (Optional) Run Baseline for Comparison
python src/train_nexus.py --epochs 10 --save_dir experiments/baseline
```

## Results: Validated & Meaningful

**MANDATORY VERIFICATION:** The graph below confirms successful dendritic activation (Green Dot = Global Optima).

![Perforated AI Training Graph](assets/PAI.png)

| Metric | Baseline | **NEXUS (Dendritic)** | Impact |
| :--- | :--- | :--- | :--- |
| **Final Loss** | 0.0038 | **0.0036** | **5.3% Error Reduction** <br>*(1 - 0.0036/0.0038)* |
| **Stability** | Plateaus Early | **Continuous Optimization** | "Dendritic Switch" Effect |
| **Model Size** | 88.13 MB | **88.72 MB** | **+0.6% (Safe Adaptation)** |

## Deep Dives & Reports

*   **[Case Study One-Pager](docs/CASE_STUDY.md)**: Full narrative, business case, and hardware/HIPAA analysis.
*   **[W&B Sweep Report](docs/SWEEP_REPORT.md)**: Hyperparameter exploration and parallel coordinates visualization.

---
*Submitted for the PyTorch Dendritic Optimization Hackathon 2025*
