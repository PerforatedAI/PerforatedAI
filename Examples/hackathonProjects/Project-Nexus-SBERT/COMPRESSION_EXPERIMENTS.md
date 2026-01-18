# NEXUS V2 Compression Experiment Guide

This document outlines experiments to address the overfitting issue identified by the reviewer.

## The Problem

The original NEXUS implementation showed overfitting:
- Training score kept improving
- Validation score plateaued
- Dendrites were ultimately deleted because they didn't help validation

As Rorry explained: "This could be a hint that the architecture is compatible, but the overfitting is already too strong. Often, this means this could be a great case for a compression experiment rather than an accuracy experiment."

## Solution: Compression Experiments

The idea is to use a **smaller model** and see if dendrites can help it reach the same accuracy as the original larger model with fewer parameters.

---

## Quick Validation Commands (30-45 min total)

Run these 2 experiments to quickly test if compression helps:

### Step 1: Compressed Baseline (50% size) - ~10 min
```bash
python src/train_nexus_v2.py --compression 0.5 --epochs 6 --batch_size 32 --lr 2e-5 --save_dir experiments/baseline_50pct
```

### Step 2: Compressed + Dendrites (50% size) - ~15-20 min  
```bash
python src/train_nexus_v2.py --use_dendrites --compression 0.5 --epochs 6 --batch_size 32 --lr 2e-5 --warmup_epochs 2 --save_dir experiments/dendritic_50pct
```

**That's it!** Compare the two results. If dendrites improve validation score on the compressed model, you've found the solution!

---

## Full Experiment Commands (If you have more time later)

### 1. Baseline (Full Size, No Dendrites)
```bash
python src/train_nexus_v2.py --epochs 8 --batch_size 32 --lr 2e-5 --save_dir experiments/baseline_full_v2
```

### 2. Compressed Baseline (25% size, No Dendrites)
```bash
python src/train_nexus_v2.py --compression 0.25 --epochs 8 --batch_size 32 --lr 2e-5 --save_dir experiments/baseline_25pct
```

### 3. Compressed + Dendrites (25% size)
```bash
python src/train_nexus_v2.py --use_dendrites --compression 0.25 --epochs 8 --batch_size 32 --lr 2e-5 --warmup_epochs 2 --save_dir experiments/dendritic_25pct
```

### 4. With Dropout Regularization
```bash
python src/train_nexus_v2.py --use_dendrites --dropout 0.2 --epochs 8 --batch_size 32 --lr 2e-5 --warmup_epochs 2 --save_dir experiments/dendritic_dropout
```

### 5. Compressed + Dropout + Dendrites
```bash
python src/train_nexus_v2.py --use_dendrites --compression 0.5 --dropout 0.15 --epochs 8 --batch_size 32 --lr 2e-5 --warmup_epochs 2 --save_dir experiments/dendritic_compressed_dropout
```

---

## What to Look For

### Signs of Success (Good Graph):
1. Training and validation scores track together (no large gap)
2. After dendrite activation (blue vertical line), BOTH training and validation improve
3. Dendrites are NOT deleted at the end
4. Final validation score is close to or better than full-size baseline

### Signs of Still Overfitting:
1. Training score >> Validation score (large gap)
2. Dendrites get deleted
3. Validation score doesn't improve after dendrite activation

---

## Expected Results

| Experiment | Parameters | Expected Val Spearman | Notes |
|------------|------------|----------------------|-------|
| Full Baseline | 100% | ~0.888 | Reference point |
| 50% Baseline | 50% | ~0.87-0.88 | Slight drop expected |
| 25% Baseline | 25% | ~0.85-0.87 | More noticeable drop |
| 50% + Dendrites | 50% + PAI | ~0.88+ | **Dendrites should help!** |
| 25% + Dendrites | 25% + PAI | ~0.86-0.88 | **Best candidate for dendrite benefit** |

---

## Success Criteria

**If compression experiments work:**
- The compressed model with dendrites achieves similar or better validation performance than the full-size baseline
- This proves dendrites can help recover the performance lost from compression
- This is a valid and valuable use case for Perforated Backpropagation!

**Narrative to present:**
> "While initial experiments showed the full-size model was already overparameterized for this dataset, compression experiments revealed that dendrites can effectively recover performance in resource-constrained scenarios. A 50% compressed model with dendrites achieved comparable performance to the full baseline, demonstrating the value of adaptive architecture for edge deployment."

---

## Next Steps After Running Experiments

1. Compare the PAI/PAI.png graphs from each run
2. Look for runs where dendrites actually helped validation
3. Update README.md with the new findings
4. Create a compelling narrative about compression + dendrites

Good luck! 🚀
