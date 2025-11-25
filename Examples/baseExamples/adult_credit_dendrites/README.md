# Adult & Credit Tabular Compression with Dendrites

This example adds AI dendrites to the Adult Income and Credit Default tabular benchmarks to show how parameter counts drop without losing AUC.

## What’s inside this folder?
- `train.py`: single entry point with dataset flag, dendrite toggles, and logging utilities.
- `run_sweep.py` / `Makefile`: helper shortcuts for the adult runs.
- `metrics.py`, `param_count.py`, `test_setup.py`: light utilities for metrics, parameter counting, and smoke testing.
- `results/`: CSVs, comparison chart, and the final PAI graph.

## Installation
```bash
pip install -r Examples/baseExamples/adult_credit_dendrites/requirements.txt
```

## Running the experiments
All commands assume repo root. (Optional) Set `MPLCONFIGDIR` to avoid font-cache warnings on macOS:
```bash
export MPLCONFIGDIR="$(pwd)/Examples/baseExamples/adult_credit_dendrites/results"
```

### 1. Adult Income baseline (≈450k params)
```bash
python Examples/baseExamples/adult_credit_dendrites/train.py \
  --dataset adult --epochs 1000 --patience 1000 \
  --width 512 --dropout 0.25 \
  --no-dendrites \
  --notes adult_original_base
```

### 2. Adult Income shrunk + dendritic
```bash
python Examples/baseExamples/adult_credit_dendrites/train.py \
  --dataset adult --epochs 1000 --patience 1000 \
  --width 64 --dropout 0.50 \
  --use-dendrites --exclude-output-proj \
  --max-dendrites 8 --fixed-switch-num 50 \
  --seed 1337 \
  --notes adult_shrunk_dend
```

### 3. Credit Default baseline
```bash
python Examples/baseExamples/adult_credit_dendrites/train.py \
  --dataset credit \
  --epochs 1000 --patience 1000 \
  --width 128 --dropout 0.25 \
  --no-dendrites \
  --notes credit_compact_base
```

### 4. Credit Default dendritic 
```bash
python Examples/baseExamples/adult_credit_dendrites/train.py \
  --dataset credit \
  --epochs 1000 --patience 1000 \
  --width 64 --dropout 0.50 \
  --use-dendrites --exclude-output-proj \
  --max-dendrites 8 --fixed-switch-num 50 \
  --seed 1337 \
  --notes credit_dend_w64_hist_seed1337
```

### Sweep helper (width × dropout × dendrites)
```bash
for dataset in adult credit; do
  for width in 32 48 64 128 256; do
    for dropout in 0.25 0.50; do
      for use_dendrites in true false; do
        notes="${dataset}_w${width}_d${dropout}_$( [ "$use_dendrites" = true ] && echo dend || echo base )"
        python Examples/baseExamples/adult_credit_dendrites/train.py \
          --dataset $dataset \
          --epochs 1000 --patience 1000 \
          --width $width --dropout $dropout \
          $( [ "$use_dendrites" = true ] && echo "--use-dendrites --exclude-output-proj --max-dendrites 8 --fixed-switch-num 50" || echo "--no-dendrites" ) \
          --seed 1337 \
          --notes "$notes"
      done
    done
  done
done
```
Seed 1337 provided the best results in our sweeps.

### Smoke test
```bash
python Examples/baseExamples/adult_credit_dendrites/test_setup.py
```

## Datasets
- **Adult Income** (`phpMawTba.arff`): pulled automatically from OpenML (`adult`, version 2). The script caches it under `data_cache/openml/`.
- **Default of Credit Card Clients** (`default of credit card clients.arff`): also fetched via OpenML (ID 42477). If network is disabled, download the ARFF manually, drop it into `data_cache/openml/`, and rerun the commands above.

## Outcomes

- Compression plots (test AUC left, parameters right):  
  - `results/compression_adult.png` (Adult baseline → shrunk → shrunk+dendrites)  
  - `results/compression_credit.png` (Credit baseline → shrunk → shrunk+dendrites)
- Test AUC bars: `results/bar_adult.png`, `results/bar_credit.png`
- PAI plots: `results/pai_credit_seed.png` (credit dendritic) and `results/pai_adult.png` (adult dendritic)

## Results summary (val / test AUC)
Dataset | Model | Params | Δ vs baseline | Val AUC | Test AUC | Notes
---|---|---|---|---|---|---
Adult | Baseline (w=512) | 450,049 | — | 0.9125 | 0.9159 | `adult_original_base`
Adult | Shrunk baseline (w=64) | 13,249 | −97% | 0.9122 | 0.9161 | `adult_shrunk_base`
Adult | Shrunk + dendrites (w=64, seed 1337) | 127,201 | −72% | 0.9163 | 0.9161 | `adult_shrunk_dend`
Credit | Baseline (w=128) | 27,905 | — | 0.7947 | 0.7804 | `credit_compact_base`
Credit | Shrunk baseline (w=32) | 2,369 | −92% | 0.7802 | 0.7653 | `credit_shrunk_base`
Credit | Shrunk + dendrites (w=64, seed 1337) | 89,521 | — vs shrunk | 0.8008 | 0.7829 | `credit_best_dend`

`results/best_test_scores.csv` stores these headline rows. The full sweep is in `results/best_test_scores_full.csv`. `results/inference_bench.csv` holds throughput numbers, and `results/params_progression.csv` logs dendrite growth over time.

## Tips & troubleshooting
- Change `--max-dendrites` / `--fixed-switch-num` to explore other compression targets. Everything is logged so you can audit each restructure.
- For offline usage, copy the two ARFFs into `data_cache/openml/`; the loader automatically prefers local files.
- Every dendritic run now emits the standard Perforated AI plot bundle (`<save_name>/*.png`). Attach the final `PAI.png` when you share results.
- Use `make sweep` to recreate the adult baseline+dendritic pair in one go.
