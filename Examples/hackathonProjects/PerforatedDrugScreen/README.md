# Perforated Drug Screening with GIN on MoleculeNet BBBP

## Intro - Required

**Description:**

This hackathon project demonstrates the application of **Perforated AI’s Dendritic Optimization** to a **Graph Isomorphism Network (GIN)** trained on the **MoleculeNet BBBP (Blood–Brain Barrier Penetration)** dataset. BBBP is a standard benchmark in molecular property prediction and drug discovery, where accurately identifying whether a compound can cross the blood–brain barrier is critical for CNS drug development.

We compare:
- a **baseline GIN** trained with standard backpropagation, and
- the **same GIN architecture** enhanced with **dendritic optimization** (PerforatedAI),

to evaluate whether dendrites can improve predictive performance and learning dynamics in **small, noisy biomedical graph datasets**, which are common in real-world drug discovery pipelines.

This submission is structured to be **fully reproducible**, with both baseline and dendritic runs included.

**Team:**

- **Abhishek Nandy** – Principal Machine Learning Engineer / Independent Researcher  
  (Drug Discovery, Graph ML, Systems Optimization)

---

## Project Impact - Required

Predicting **blood–brain barrier penetration** is a high-impact problem in pharmaceutical R&D. BBB penetration failure is a common reason promising candidates are discarded late in development, resulting in significant time and cost loss.

Even small improvements in predictive accuracy can:
- reduce late-stage drug attrition,
- enable earlier elimination of non-viable compounds, and
- lower experimental and computational screening costs.

Dendritic optimization is particularly relevant in this domain because:
- drug discovery datasets are often **small and noisy**, and
- graph models are costly to scale,
so improvements in accuracy and/or convergence efficiency can translate into faster iteration cycles and reduced compute needs.

---

## Usage Instructions - Required

**Installation:**

Create and activate a conda environment:

```bash
conda create -n perforated-drugscreen python=3.11 -y
conda activate perforated-drugscreen
```

Install dependencies:

```bash
pip install torch torch-geometric wandb perforatedai
```

> Note: Torch Geometric optional CUDA extensions are not required for this experiment. CPU execution is supported.

**Run - Baseline (No Dendrites):**

```bash
python bbbp_original.py \
  --hidden_dim 64 \
  --num_layers 4 \
  --epochs 40 \
  --weight_decay 0.0 \
  --seed 0
```

**Run - Dendritic Optimization (PerforatedAI):**

```bash
python bbbp_perforatedai_wandb.py \
  --hidden_dim 64 \
  --num_layers 4 \
  --epochs 40 \
  --weight_decay 0.0 \
  --seed 0 \
  --doing_pai \
  --wandb \
  --wandb_project PerforatedDrugScreen \
  --wandb_run_name BBBP_dendrites_hd64_L4_seed0
```

Both scripts use the **same architecture, dataset split, optimizer configuration, and random seed**, ensuring a fair comparison.

---

## Results - Required

This BBBP example shows that **Dendritic Optimization** can improve predictive performance on a graph-based drug discovery benchmark. Comparing the best baseline run to the best dendritic run:

| Model | Best Val AUC | Test AUC @ Best Val | Parameters |
|------|--------------:|--------------------:|-----------:|
| Baseline GIN (No Dendrites) | 0.8591 | 0.8269 | 68,482 |
| GIN + Dendritic Optimization | 0.9220 | 0.9083 | 103,044 |

### Remaining Error Reduction

Using Test AUC as the primary score:

- Baseline error = 1 − 0.8269 = 0.1731  
- Dendritic error = 1 − 0.9083 = 0.0917  

Remaining Error Reduction:

\[
(0.1731 - 0.0917) / 0.1731 \approx 47.0\%
\]

Dendritic optimization eliminated **~47% of the remaining error** compared to the baseline GIN model.

### Notes on Stability / Ablation

Additional runs suggest that on small, near-saturated datasets like BBBP, unconstrained dendritic growth can increase capacity without always improving generalization. This highlights the importance of selecting the correct dendritic operating regime (e.g., accuracy-seeking vs. compression-seeking) for biomedical graph learning workloads.

---

## Raw Results Graph - Required

When running the Perforated AI library, output graphs are automatically generated (by default in the `PAI/` folder). The final graph produced by the dendritic training run is required for judging, because it verifies that dendrites were correctly added and trained.

**This submission includes the required PerforatedAI output graph below:**

![Perforated AI output graph](./PAI/PAI.png)

