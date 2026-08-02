# PAI Pancreas — Cell Type Annotation with Dendritic Transformers

Applies [PerforatedAI](https://github.com/PerforatedAI/PerforatedAI) dendritic augmentation to cross-technology single-cell RNA-seq cell type annotation on the Human Pancreas scib benchmark (Luecken et al. Nature Methods 2022).

---

## Results

### Accuracy improvement (full 256-dim model)

| Model | Params | Test Acc |
|-------|--------|----------|
| Baseline (3-seed mean) | 2,135,822 | 95.38% ± 0.92% |
| PAI best snapshot | 5,349,758 | **97.05%** |

**Δ = +1.68pp** · Bootstrap 95% CI: [+0.40%, +2.55%] · **Statistically significant**  
Relative error reduction: **36.2%**

### Compression experiment (128-dim small model)

| Model | Params | Test Acc | vs Full Vanilla |
|-------|--------|----------|-----------------|
| Full vanilla (256-dim) | 2,135,822 | 95.38% | — |
| Small vanilla (128-dim) | 543,630 | 94.95% | −0.42% |
| **Small + PAI (best snap)** | **1,086,776** | **95.78%** | **+0.40%** |

**Small + PAI uses 49% of full vanilla's parameters and achieves higher accuracy.**

---

## Dataset

Human Pancreas scib benchmark (`human_pancreas_norm_complexBatch.h5ad`)  
Source: Luecken et al. Nature Methods 2022 · Figshare article 12420968  
16,382 cells × 19,093 genes · 14 cell types · pre-normalised

**Cross-technology split** (original 4-study benchmark):
- Train: Baron (inDrop1-4) + Muraro (celseq2) + Segerstolpe (smartseq2) — 13,248 cells
- Test: Xin (smarter) — 1,492 cells
- Training uses a 10% stratified subsample (rare cell type stress test)

---

## Architecture

```
2000 HVGs
  → 50 chunks of 40 genes
  → Linear(40, d_model)       # gene chunk embedding
  + learnable positional embedding (50 positions)
  → 4 × TransformerEncoderLayer(d_model, nhead=4)
  → mean pool → LayerNorm
  → Linear(d_model, 14)       # classifier
```

PAI perforates all 14 `nn.Linear` layers. Full model: `d_model=256, d_ff=512`. Small model: `d_model=128, d_ff=256`.

---

## Setup

```bash
conda activate env_scrna
pip install --no-deps git+https://github.com/PerforatedAI/PerforatedAI.git
```

The `.h5ad` file is downloaded automatically on first run (~301 MB, cached locally).

---

## Usage

```bash
# Full run: baseline (3 seeds) + PAI
python main.py

# Baseline only (saves baseline_cache_full.pkl)
python main.py --baseline-only

# PAI only (requires baseline cache)
python main.py --pai-only

# Compression experiment: 128-dim small model
python main.py --small-model --baseline-only   # saves baseline_cache_small.pkl
python main.py --small-model --pai-only        # requires both caches for 4-model table

# CPU / specific device
python main.py --device cpu
```

> **Important**: Restart the Python process between PAI re-runs. PAI uses global state that must be cleared.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters |
| `data.py` | Download, preprocessing, DataLoader construction |
| `model.py` | `GeneTransformer` architecture |
| `train.py` | Training loop, evaluation, bootstrap CI |
| `baseline.py` | 3-seed baseline experiment |
| `pai_experiment.py` | PAI training with DOING_HISTORY dendrite insertion |
| `plots.py` | Figures and summary tables |
| `main.py` | CLI entry point |
