#!/usr/bin/env python3
"""
main.py — end-to-end runner: baseline → PAI → figures → report

Usage
-----
    python main.py                      # full run (baseline + PAI), full model
    python main.py --small-model        # compression experiment: 128-dim model
    python main.py --baseline-only      # skip PAI
    python main.py --pai-only           # skip baseline (requires cached baseline)
    python main.py --device cpu         # override device
    python main.py --pai-seed 0         # change PAI seed

Compression experiment (--small-model)
---------------------------------------
    "smaller model + dendrites ≈ full-size vanilla"

    Full vanilla   (256-dim, no dendrites) → reference accuracy & params
    Small vanilla  (128-dim, no dendrites) → lower accuracy, fewer params
    Small + PAI    (128-dim + dendrites)   → closes the gap, fewer params than full vanilla

Important
---------
    Restart the Python process between PAI re-runs.
    PAI uses global state (GPA) that corrupts subsequent runs if not cleared.
"""

import argparse
import os
import pickle
import sys
import warnings

warnings.filterwarnings('ignore')

import torch


def _detect_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='PAI Pancreas experiment')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--pai-only',      action='store_true')
    parser.add_argument('--small-model',   action='store_true',
                        help='Use 128-dim model (compression experiment)')
    parser.add_argument('--test-tech',     default='smarter',
                        choices=['smarter', 'smartseq2', 'celseq2',
                                 'inDrop1', 'inDrop2', 'inDrop3', 'inDrop4'],
                        help='Technology to hold out as test set '
                             '(default: smarter=Xin; smartseq2=Segerstolpe is harder)')
    parser.add_argument('--device',        default=None)
    parser.add_argument('--pai-seed',      type=int, default=42)
    args = parser.parse_args()

    from config import D_MODEL, D_FF, D_MODEL_SMALL, D_FF_SMALL
    d_model = D_MODEL_SMALL if args.small_model else D_MODEL
    d_ff    = D_FF_SMALL    if args.small_model else D_FF
    size_tag = 'small' if args.small_model else 'full'
    tag      = f'{size_tag}_{args.test_tech}'
    if args.small_model:
        print(f'Model: SMALL ({d_model}-dim, {d_ff}-dim ff)  [compression experiment]')
    else:
        print(f'Model: FULL  ({d_model}-dim, {d_ff}-dim ff)')
    print(f'Test technology: {args.test_tech}')

    device = args.device or _detect_device()
    print(f'Device: {device}')

    # ── Data ─────────────────────────────────────────────────────────────
    from data import download_pancreas, prepare_pancreas
    raw                             = download_pancreas()
    train_adata, test_adata, label_col = prepare_pancreas(raw, test_tech=args.test_tech)

    # ── Baseline ──────────────────────────────────────────────────────────
    BASELINE_CACHE = f'baseline_cache_{tag}.pkl'
    baseline_results = None

    if args.pai_only and os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, 'rb') as f:
            baseline_results = pickle.load(f)
        print(f'Loaded cached baseline: {baseline_results["mean_acc"]*100:.2f}% ± '
              f'{baseline_results["std_acc"]*100:.2f}%')

    if not args.pai_only and baseline_results is None:
        from baseline import run_baseline
        baseline_results = run_baseline(
            train_adata, test_adata, label_col, device,
            d_model=d_model, d_ff=d_ff,
        )

        with open(BASELINE_CACHE, 'wb') as f:
            pickle.dump(baseline_results, f)
        print(f'Baseline cached to {BASELINE_CACHE}')

        from plots import plot_baseline
        plot_baseline(baseline_results, tag=tag)

    # ── PAI ───────────────────────────────────────────────────────────────
    if not args.baseline_only:
        if baseline_results is None:
            print('ERROR: no baseline results found. '
                  f'Run without --pai-only first (creates {BASELINE_CACHE}).',
                  file=sys.stderr)
            sys.exit(1)

        from pai_experiment import run_pai
        pai_results = run_pai(
            train_adata, test_adata, label_col,
            baseline_results, device, seed=args.pai_seed,
            d_model=d_model, d_ff=d_ff,
        )

        from plots import plot_pai, print_progressive_table, print_final_summary

        # For the compression experiment, load the full vanilla baseline
        # so we can show the 4-model comparison table and chart.
        full_baseline = None
        if args.small_model:
            full_cache = f'baseline_cache_full_{args.test_tech}.pkl'
            if os.path.exists(full_cache):
                with open(full_cache, 'rb') as f:
                    full_baseline = pickle.load(f)
                print(f'Loaded full vanilla baseline for compression table: '
                      f'{full_baseline["mean_acc"]*100:.2f}%')
            else:
                print(f'WARNING: {full_cache} not found — run without '
                      f'--small-model first to get the full vanilla reference.')

        plot_pai(pai_results, baseline_results, tag=tag)
        print_progressive_table(pai_results, baseline_results)
        print_final_summary(pai_results, baseline_results,
                            full_baseline=full_baseline, test_tech=args.test_tech)

        if args.small_model and full_baseline is not None:
            from plots import plot_compression_comparison
            plot_compression_comparison(full_baseline, baseline_results, pai_results,
                                        test_tech=args.test_tech)


if __name__ == '__main__':
    main()
