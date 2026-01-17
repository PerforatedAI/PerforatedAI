Perforated Drug Screening with GIN on MoleculeNet BBBP
Intro (Required)
Description

This hackathon project demonstrates the application of Perforated AI’s Dendritic Optimization to a Graph Neural Network (GIN) trained on the MoleculeNet BBBP (Blood–Brain Barrier Penetration) dataset. BBBP is a standard benchmark in molecular property prediction and drug discovery, where accurately identifying whether a compound can cross the blood–brain barrier is critical for CNS drug development.

We compare a baseline GIN model trained using standard backpropagation against an identical architecture enhanced with dendritic optimization. The goal is to evaluate whether dendrites can improve predictive accuracy and learning dynamics in data-limited, noisy biomedical graph datasets, a common challenge in real-world drug discovery pipelines.

This submission is structured to be fully reproducible, with both baseline and dendritic runs included.

Team

Abhishek Nandy – Principal Machine Learning Engineer / Independent Researcher
(Drug Discovery, Graph ML, Systems Optimization)

Project Impact (Required)

Predicting blood–brain barrier penetration is a high-impact problem in pharmaceutical research. Failed BBB penetration is a major reason why otherwise promising drug candidates are discarded late in development, resulting in significant financial and time losses.

Even small improvements in predictive accuracy can:

Reduce late-stage drug attrition

Enable earlier elimination of non-viable compounds

Lower experimental and computational screening costs

Dendritic optimization is particularly relevant in this domain because:

Drug datasets are often small and noisy

Graph models are expensive to scale

Improving accuracy or convergence efficiency can unlock lower-cost hardware and faster iteration cycles

Usage Instructions (Required)
Installation

Create and activate a conda environment:

conda create -n perforated-drugscreen python=3.11 -y
conda activate perforated-drugscreen


Install dependencies:

pip install torch torch-geometric wandb perforatedai


⚠️ Note: Torch Geometric optional CUDA extensions are not required for this experiment and CPU execution is supported.

Run – Baseline (No Dendrites)
python bbbp_original.py \
  --hidden_dim 64 \
  --num_layers 4 \
  --epochs 40 \
  --weight_decay 0.0 \
  --seed 0

Run – Dendritic Optimization
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


Both scripts use the same architecture, optimizer, dataset split, and random seed, ensuring a fair comparison.

Results (Required)
Accuracy Comparison
Model	Best Val AUC	Test AUC @ Best Val	Parameters
Baseline GIN (No Dendrites)	0.8591	0.8269	68,482
GIN + Dendritic Optimization	0.9220	0.9083	103,044
Remaining Error Reduction

Baseline error = 1 − 0.8269 = 0.1731
Dendritic error = 1 − 0.9083 = 0.0917

Remaining Error Reduction:

(0.1731 − 0.0917) / 0.1731 ≈ 47.0%


Dendritic optimization eliminated ~47% of the remaining error compared to the baseline GIN model.

Notes on Stability & Ablation

Additional runs showed that on small, near-saturated datasets like BBBP, unconstrained dendritic growth can increase capacity without improving generalization. This highlights the importance of choosing the correct dendritic operating regime (accuracy-seeking vs compression-seeking) — a key insight from this project.