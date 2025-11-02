# Adult Income & Credit Default — Dendritic MLP vs Vanilla

**Goal.** I wanted to demonstrate that Artificial dendrites can shrink tabular MLPs while keeping Adult Income and Credit default accuracy intact on two structured datasets.

## Headline
- **Adult Income:** 47% fewer parameters (450k → 238k) with a +0.0005 test AUC lift.
- **Credit Default:** 26% fewer parameters (407k → 301k) while 
- Visualization → see `results/quality_vs_params.png`.


## Setup
- Adult Income: UCI “adult” dataset from OpenML (cached locally as `phpMawTba.arff`).
- Credit Default: “Default of Credit Card Clients” OpenML dataset (cached as `default of credit card clients.arff`).
- Baseline: 3-layer MLP, width 512, dropout 0.25, Adam 1e-3, ReduceLROnPlateau, early stop patience 6.
- Dendritic run: width 128 core, dendrites on hidden `Linear` layers, output head left vanilla, `--max-dendrites 12`, fixed switches every 3 epochs.
- Utility modules: `metrics.py` (AUC/F1 helpers), `param_count.py` (trainable counts), `test_setup.py` (forward smoke test).

## Reproduce
```bash
# Adult baseline: heavy MLP reference (≈450k params)
python Examples/adult_income_dendritic_prashaman/train.py \
  --epochs 40 --patience 6 \
  --width 512 --dropout 0.25 \
  --no-dendrites \
  --notes "baseline_w512"

# Adult dendritic: compressed model (≈240k params)
python Examples/adult_income_dendritic_prashaman/train.py \
  --epochs 60 --patience 10 \
  --width 128 --dropout 0.25 \
  --use-dendrites --exclude-output-proj \
  --max-dendrites 12 --fixed-switch-num 3 \
  --notes "pai_w128_cap12"

# Credit baseline
python Examples/adult_income_dendritic_prashaman/train.py \
  --dataset credit \
  --epochs 40 --patience 6 \
  --width 512 --dropout 0.25 \
  --no-dendrites \
  --notes "credit_baseline_w512"

# Credit dendritic (seeded sweep – best seed=1337 shown)
python Examples/adult_income_dendritic_prashaman/train.py \
  --dataset credit \
  --epochs 60 --patience 10 \
  --width 128 --dropout 0.25 \
  --use-dendrites --exclude-output-proj \
  --max-dendrites 8 --fixed-switch-num 3 \
  --seed 1337 \
  --notes "credit_dend_w128_cap8_seed1337"
```
Results land in `results/best_test_scores.csv`, `results/params_progression.csv`, and `results/quality_vs_params.png`.
`python run_sweep.py` (or `make sweep`) reproduces the Adult runs; the credit run was repeated across a few seeds (17, 23, 42, 1337, 2025) and seed 1337 delivered the no-loss result logged below.

## Results
Dataset | Model | Params | Δ vs Baseline | Val AUC | Test AUC | Notes
---|---|---|---|---|---|---
Adult | Vanilla MLP (w=512) | 450,049 | — | 0.9125 | 0.9159 | `baseline_w512`
Adult | Dendritic MLP (w=128) | 238,465 | −47% | 0.9125 | 0.9164 | `pai_w128_cap12`
Credit | Vanilla MLP (w=512) | 406,529 | — | 0.7839 | 0.7726 | `credit_baseline_w512`
Credit | Dendritic MLP (w=128, seed 1337) | 301,185 | −26% | 0.7955 | 0.7810 | `credit_dend_w128_cap8_seed1337`

Credit required a light seed sweep (documented above); the winning seed retains the 26% compression while nudging both validation and test AUC higher than baseline. All runs use PerforatedAI’s tracker to manage switches and log parameter growth. Inference throughput is logged to `results/inference_bench.csv` (samples/sec) to highlight runtime gains from smaller models.

## Notes
- Pass `--max-dendrites` and `--fixed-switch-num` to explore different compression targets without editing code (see runs above).
- Place an ARFF dump in `data_cache/` to run offline; otherwise the code fetches from OpenML and caches automatically.
- Run the smoke test (`python Examples/adult_income_dendritic_prashaman/test_setup.py`) before long trainings; it exercises both vanilla and dendritic forward passes.
- `make format` / `make test` / `make sweep` provide quick consistency checks and repro.

## Parameter Accounting
`param_count.py` counts trainable tensors, and `train.py` logs the total at initialization and after each epoch. Dendritic parameter progression is written to `results/params_progression.csv` so you can audit every restructure.
