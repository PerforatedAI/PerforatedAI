# Architectural Limits: SBERT & Dendritic Optimization

**Team:** Aakanksha Singh & Mihir Phalke (K.J. Somaiya College of Engineering)  
**Model:** all-MiniLM-L6-v2 | **Task:** STS Benchmark

## Research Goal
To determine if Perforated Backpropagation can enhance the performance of a highly optimized, industry-standard model (Sentence-BERT) used in healthcare and finance RAG systems.

## Experiments
We conducted a comprehensive sweep (22 runs) varying:
*   Learning rates (1e-5 to 5e-5)
*   Batch sizes (16, 32)
*   Warmup epochs (2, 4, 6)

## Key Findings

**1. Baseline Efficiency:**
The baseline SBERT model achieved optimal performance (**89.18% Spearman**) at **Epoch 0**. This indicates the architecture is already perfectly parameterized for the STS task.

**2. Dendritic Behavior:**
The dendritic system successfully activated and grew architecture, but validation scores did not improve beyond the baseline. This confirms that adding capacity to an already-optimal model yields diminishing returns.

## Conclusion & Learnings
This study defines clear boundaries for dendritic optimization:
*   **Best Use Case:** Underfitting models or domain adaptation (e.g., medical/legal specific tasks).
*   **Limited Utility:** Fine-tuning architectures that are already at the efficiency frontier (like SBERT on STS).

This negative result is valuable for practitioners, saving compute resources by identifying where standard architectures suffice vs. where dendritic growth is necessary.

---
**Links:**
*   [W&B Report](https://wandb.ai/aakanksha-singh0205-kj-somaiya-school-of-engineering/PerforatedAI-Examples_hackathonProjects_Project-Nexus-SBERT_src)
*   [Repository Code](Examples/hackathonProjects/Project-Nexus-SBERT/)
