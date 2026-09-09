#!/usr/bin/env python3
"""
Process pruning sweep output directories and generate a scatter plot.

Input directory structure:
    <sweep_dir>/
        run0/
            pruning_results.csv          # 10 rows: iterations 0-9 of pruning
            PAI_prune1_<timestamp>/
                PAI_prune1_<timestamp>_best_arch_scores.csv
            PAI_prune2_<timestamp>/
                PAI_prune2_<timestamp>_best_arch_scores.csv
            ...
            PAI_prune9_<timestamp>/
                PAI_prune9_<timestamp>_best_arch_scores.csv
        run1/
            ...

pruning_results.csv columns: iteration, param_count, best_f1
best_arch_scores.csv columns: Param Counts, Max Valid Scores, Train Person f1
  - Row 0: starting pruned architecture (no dendrites added)
  - Rows 1+: architectures after adding dendrites

Output:
    scatter.png  -- param count (x) vs accuracy (y), colored by series:
        - Pruned baseline  (pruning_results.csv)
        - Dendrite row 0   (first row of each best_arch_scores, across all runs)
        - Dendrite row 1   (second row, ...)
        - ...

Example:
    python process_pruning_sweep_output.py --folder thoro_pruning_sweep/
    python process_pruning_sweep_output.py --folder thoro_pruning_sweep/ --output my_output/
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


_FONT_SCALE = 1.5


def _scale_rc_value(value, fallback: float) -> float:
    if isinstance(value, (int, float)):
        return float(value) * _FONT_SCALE
    return fallback * _FONT_SCALE


plt.rcParams.update({
    "font.size": _scale_rc_value(plt.rcParams.get("font.size"), 10.0),
    "axes.titlesize": _scale_rc_value(plt.rcParams.get("axes.titlesize"), 12.0),
    "axes.labelsize": _scale_rc_value(plt.rcParams.get("axes.labelsize"), 10.0),
    "xtick.labelsize": _scale_rc_value(plt.rcParams.get("xtick.labelsize"), 10.0),
    "ytick.labelsize": _scale_rc_value(plt.rcParams.get("ytick.labelsize"), 10.0),
    "legend.fontsize": _scale_rc_value(plt.rcParams.get("legend.fontsize"), 10.0),
    "figure.titlesize": _scale_rc_value(plt.rcParams.get("figure.titlesize"), 12.0),
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pruning_results(run_dir: str) -> List[Tuple[float, float]]:
    """Load pruning_results.csv and return list of (param_count, accuracy)."""
    path = os.path.join(run_dir, "pruning_results.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Accept either 'param_count' or 'param_counts' (case-insensitive)
    param_col = next((c for c in df.columns if c.lower() in ("param_count", "param_counts")), None)
    acc_col = next((c for c in df.columns if c.lower() in ("best_f1", "accuracy", "best_acc", "acc")), None)
    if param_col is None or acc_col is None:
        print(f"  WARNING: Could not identify columns in {path}; got {list(df.columns)}", file=sys.stderr)
        return []
    results = []
    for _, row in df.iterrows():
        try:
            p = float(row[param_col])
            a = float(row[acc_col])
            results.append((p, a))
        except (ValueError, TypeError):
            continue
    return results


def _load_best_arch_scores(subfolder: str, subfolder_name: str) -> List[Tuple[float, float]]:
    """Load <subfolder_name>_best_arch_scores.csv and return list of (param_count, accuracy)."""
    path = os.path.join(subfolder, f"{subfolder_name}_best_arch_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected best_arch_scores file not found: {path}")

    rows: List[Tuple[float, float]] = []
    try:
        df = pd.read_csv(path, header=0)
        df.columns = [c.strip() for c in df.columns]
        # Handle files that accidentally have a duplicate header row
        param_col = next((c for c in df.columns if "param" in c.lower()), None)
        acc_col = next((c for c in df.columns if "valid" in c.lower() or "score" in c.lower() or "acc" in c.lower()), None)
        if param_col is None or acc_col is None:
            print(f"  WARNING: Could not identify columns in {path}; got {list(df.columns)}", file=sys.stderr)
            return []
        for _, row in df.iterrows():
            try:
                p = float(row[param_col])
                a = float(row[acc_col])
                rows.append((p, a))
            except (ValueError, TypeError):
                # Skips duplicate header rows or malformed entries
                continue
    except Exception as exc:
        print(f"  WARNING: Failed to read {path}: {exc}", file=sys.stderr)
    return rows


def _load_run(run_dir: str) -> Tuple[List[Tuple[float, float]], Dict[int, List[Tuple[float, float]]]]:
    """Load one run directory.

    Returns:
        pruning_points: list of (param, acc) from pruning_results.csv
        dendrite_rows:  dict mapping row-index -> list of (param, acc) from
                        all PAI_prune* best_arch_scores files
    """
    pruning_points = _load_pruning_results(run_dir)

    dendrite_rows: Dict[int, List[Tuple[float, float]]] = {}

    for entry in sorted(os.listdir(run_dir)):
        sub_path = os.path.join(run_dir, entry)
        if not os.path.isdir(sub_path):
            continue
        if not entry.startswith("PAI_prune"):
            continue
        points = _load_best_arch_scores(sub_path, entry)
        for row_idx, point in enumerate(points):
            dendrite_rows.setdefault(row_idx, []).append(point)

    return pruning_points, dendrite_rows


def _load_all_runs(sweep_dir: str) -> Tuple[List[Tuple[float, float]], Dict[int, List[Tuple[float, float]]]]:
    """Walk all run* subdirectories and aggregate data."""
    all_pruning: List[Tuple[float, float]] = []
    all_dendrite: Dict[int, List[Tuple[float, float]]] = {}

    run_dirs = sorted(
        os.path.join(sweep_dir, d)
        for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d)) and d.startswith("run")
    )

    if not run_dirs:
        print(f"WARNING: No run* subdirectories found in {sweep_dir}", file=sys.stderr)

    for run_dir in run_dirs:
        pruning_pts, dendrite_rows = _load_run(run_dir)
        all_pruning.extend(pruning_pts)
        for row_idx, pts in dendrite_rows.items():
            all_dendrite.setdefault(row_idx, []).extend(pts)

    return all_pruning, all_dendrite


# ---------------------------------------------------------------------------
# Grouped data loading (for average and candlestick graphs)
# ---------------------------------------------------------------------------

def _load_pruning_results_with_iteration(run_dir: str) -> List[Tuple[int, float, float]]:
    """Load pruning_results.csv and return list of (iteration, param_count, accuracy)."""
    path = os.path.join(run_dir, "pruning_results.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    param_col = next((c for c in df.columns if c.lower() in ("param_count", "param_counts")), None)
    acc_col = next((c for c in df.columns if c.lower() in ("best_f1", "accuracy", "best_acc", "acc")), None)
    iter_col = next((c for c in df.columns if c.lower() == "iteration"), None)
    if param_col is None or acc_col is None:
        print(f"  WARNING: Could not identify columns in {path}; got {list(df.columns)}", file=sys.stderr)
        return []
    results = []
    for idx, row in df.iterrows():
        try:
            p = float(row[param_col])
            a = float(row[acc_col])
            it = int(row[iter_col]) if iter_col is not None else int(idx)
            results.append((it, p, a))
        except (ValueError, TypeError):
            continue
    return results


def _load_all_runs_grouped(
    sweep_dir: str,
) -> Tuple[
    Dict[int, Dict[str, Tuple[float, float]]],
    Dict[Tuple[int, int], Dict[str, Tuple[float, float]]],
    int,
]:
    """Load all runs and group data by (prune_idx, row_idx) pair, keyed by run name.

    Returns:
        pruning_by_iter:       {iteration_idx: {run_name: (param_count, acc)}}
        dendrite_by_prune_row: {(prune_idx, row_idx): {run_name: (param_count, acc)}}
        total_runs:            number of run* directories found
    """
    run_dirs = sorted(
        os.path.join(sweep_dir, d)
        for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d)) and d.startswith("run")
    )
    total_runs = len(run_dirs)

    pruning_by_iter: Dict[int, Dict[str, Tuple[float, float]]] = {}
    dendrite_by_prune_row: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        for it, param, acc in _load_pruning_results_with_iteration(run_dir):
            pruning_by_iter.setdefault(it, {})[run_name] = (param, acc)
        for entry in sorted(os.listdir(run_dir)):
            sub_path = os.path.join(run_dir, entry)
            if not os.path.isdir(sub_path) or not entry.startswith("PAI_prune"):
                continue
            m = re.match(r"PAI_prune(\d+)", entry)
            if not m:
                continue
            prune_idx = int(m.group(1))
            points = _load_best_arch_scores(sub_path, entry)
            for row_idx, (param, acc) in enumerate(points):
                # Last sorted PAI_prune* folder for the same prune_idx wins on collision.
                dendrite_by_prune_row.setdefault((prune_idx, row_idx), {})[run_name] = (param, acc)

    return pruning_by_iter, dendrite_by_prune_row, total_runs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_SERIES_COLORS = [
    "#2196F3",  # blue       — pruned baseline
    "#FF5722",  # deep orange — dendrite row 0 (initial PAI start)
    "#4CAF50",  # green       — dendrite row 1
    "#9C27B0",  # purple      — dendrite row 2
    "#FF9800",  # amber       — dendrite row 3
    "#00BCD4",  # cyan        — dendrite row 4
    "#E91E63",  # pink        — dendrite row 5
    "#795548",  # brown       — dendrite row 6
]


def _series_color(idx: int) -> str:
    return _SERIES_COLORS[idx % len(_SERIES_COLORS)]


def _make_scatter(
    dendrite_rows: Dict[int, List[Tuple[float, float]]],
    output_path: str,
) -> None:
    """Create and save the scatter plot and a companion CSV."""
    fig, ax = plt.subplots(figsize=(14, 8))

    legend_handles: List[Line2D] = []
    csv_rows: List[Dict] = []

    # --- dendrite rows ---
    for row_idx in sorted(dendrite_rows.keys()):
        pts = dendrite_rows[row_idx]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = _series_color(row_idx + 1)
        ax.scatter(xs, ys, c=color, s=60, alpha=0.85, zorder=3, marker="^")
        if row_idx == 0:
            label = "PAI: 0 dendrites added"
        else:
            label = f"PAI: {row_idx} dendrite step{'s' if row_idx > 1 else ''} added"
        legend_handles.append(
            Line2D([0], [0], marker="^", color="w", markerfacecolor=color,
                   markersize=10, label=label)
        )
        for param, acc in pts:
            csv_rows.append({"group_type": "dendrite", "dendrite_row": row_idx, "param_count": param, "score": acc})

    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pruning Sweep: Parameters vs Accuracy")
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")



# ---------------------------------------------------------------------------
# Average line+scatter graph
# ---------------------------------------------------------------------------

def _make_average_plot(
    dendrite_by_prune_row: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]],
    total_runs: int,
    min_percent: float,
    output_path: str,
) -> str:
    """Create an average-per-group line+scatter plot and write a companion CSV.

    Dendrite row 0 points (one per prune_idx, averaged across runs) are
    connected by a line as the baseline.  Rows 1+ are individual scatter dots.
    Groups whose run fraction is below min_percent are omitted.

    Returns the path to the companion CSV.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    legend_handles: List[Line2D] = []
    csv_rows: List[Dict] = []

    all_row_indices = sorted(set(k[1] for k in dendrite_by_prune_row))

    # --- Row 0: baseline line, connected across prune steps ---
    if 0 in all_row_indices:
        color = _series_color(1)
        row0_pts: List[Tuple[float, float]] = []
        for prune_idx in sorted(k[0] for k in dendrite_by_prune_row if k[1] == 0):
            run_data = dendrite_by_prune_row[(prune_idx, 0)]
            unique_runs = len(run_data)
            if total_runs > 0 and (unique_runs / total_runs) < min_percent:
                continue
            avg_p = float(np.mean([v[0] for v in run_data.values()]))
            avg_a = float(np.mean([v[1] for v in run_data.values()]))
            row0_pts.append((avg_p, avg_a))
            csv_rows.append({
                "prune_idx": prune_idx,
                "dendrite_row": 0,
                "n_runs": unique_runs,
                "avg_param_count": avg_p,
                "avg_score": avg_a,
            })
        if row0_pts:
            sorted_pts = sorted(row0_pts, key=lambda p: p[0])
            ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                    color=color, linewidth=2, zorder=2, alpha=0.8)
            ax.scatter([p[0] for p in row0_pts], [p[1] for p in row0_pts],
                       c=color, s=80, zorder=3, marker="o", alpha=0.9)
            legend_handles.append(
                Line2D([0], [0], marker="o", color=color, markerfacecolor=color,
                       markersize=10, label="PAI: 0 dendrites added (avg per prune step)")
            )

    # --- Rows 1+: individual scatter dots per (prune_idx, row_idx) pair ---
    for row_idx in all_row_indices:
        if row_idx == 0:
            continue
        color = _series_color(row_idx + 1)
        any_plotted = False
        for prune_idx in sorted(k[0] for k in dendrite_by_prune_row if k[1] == row_idx):
            run_data = dendrite_by_prune_row[(prune_idx, row_idx)]
            unique_runs = len(run_data)
            if total_runs > 0 and (unique_runs / total_runs) < min_percent:
                continue
            avg_p = float(np.mean([v[0] for v in run_data.values()]))
            avg_a = float(np.mean([v[1] for v in run_data.values()]))
            ax.scatter([avg_p], [avg_a], c=color, s=80, zorder=3, marker="^", alpha=0.9)
            csv_rows.append({
                "prune_idx": prune_idx,
                "dendrite_row": row_idx,
                "n_runs": unique_runs,
                "avg_param_count": avg_p,
                "avg_score": avg_a,
            })
            any_plotted = True
        if any_plotted:
            lbl = f"PAI: {row_idx} dendrite{'s' if row_idx > 1 else ''} added"
            legend_handles.append(
                Line2D([0], [0], marker="^", color="w", markerfacecolor=color,
                       markersize=10, label=lbl)
            )

    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pruning Sweep: Average per (Pruning Step × Dendrite Count)")
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Candlestick graph
# ---------------------------------------------------------------------------

def _make_candlestick_plot(
    dendrite_by_prune_row: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]],
    total_runs: int,
    min_percent: float,
    output_path: str,
) -> str:
    """Create a candlestick (box-plot) graph with x-axis = parameter count.

    One box per (prune_idx, row_idx) pair that passes min_percent.
    Boxes are positioned by average param_count.

    Returns the path to the companion CSV.
    """
    boxes: List[Dict] = []
    csv_rows: List[Dict] = []

    # --- Dendrite groups (one box per (prune_idx, row_idx) pair) ---
    for (prune_idx, row_idx) in sorted(dendrite_by_prune_row.keys()):
        run_data = dendrite_by_prune_row[(prune_idx, row_idx)]
        unique_runs = len(run_data)
        if total_runs > 0 and (unique_runs / total_runs) < min_percent:
            continue
        params = [v[0] for v in run_data.values()]
        accs = [v[1] for v in run_data.values()]
        if not accs:
            continue
        s = pd.Series(accs)
        avg_p = float(np.mean(params))
        boxes.append({
            "x": avg_p,
            "group_type": "dendrite",
            "prune_idx": prune_idx,
            "dendrite_row": row_idx,
            "n_runs": unique_runs,
            "color": _series_color(row_idx + 1),
            "whislo": float(s.min()),
            "q1": float(s.quantile(0.25)),
            "med": float(s.median()),
            "q3": float(s.quantile(0.75)),
            "whishi": float(s.max()),
        })
        csv_rows.append({
            "group_type": "dendrite",
            "prune_idx": prune_idx,
            "dendrite_row": row_idx,
            "n_runs": unique_runs,
            "avg_param_count": avg_p,
            "min": float(s.min()),
            "q1": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q3": float(s.quantile(0.75)),
            "max": float(s.max()),
        })

    if not boxes:
        print("WARNING: No data to plot in candlestick graph.", file=sys.stderr)
        return ""

    boxes.sort(key=lambda b: b["x"])
    x_positions = [b["x"] for b in boxes]
    x_min = min(x_positions)
    x_max = max(x_positions)
    x_span = max(1.0, x_max - x_min)

    unique_x = sorted(set(x_positions))
    gaps = [unique_x[i + 1] - unique_x[i] for i in range(len(unique_x) - 1) if unique_x[i + 1] > unique_x[i]]
    width = (min(gaps) * 0.4) if gaps else (x_span * 0.02)

    fig, ax = plt.subplots(figsize=(14, 7))
    x_pad = max(1.0, x_span * 0.05)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    bxp_stats = [
        {"whislo": b["whislo"], "q1": b["q1"], "med": b["med"], "q3": b["q3"], "whishi": b["whishi"]}
        for b in boxes
    ]
    artists = ax.bxp(
        bxp_stats,
        positions=x_positions,
        widths=width,
        showfliers=False,
        manage_ticks=False,
        patch_artist=True,
    )

    for i, b in enumerate(boxes):
        color = b["color"]
        artists["boxes"][i].set_facecolor(to_rgba(color, 0.4))
        artists["boxes"][i].set_edgecolor(color)
        artists["boxes"][i].set_linewidth(1.5)
        artists["medians"][i].set_color(color)
        artists["medians"][i].set_linewidth(2.0)
        for wi in [2 * i, 2 * i + 1]:
            artists["whiskers"][wi].set_color(color)
            artists["caps"][wi].set_color(color)

    ax.set_title("Pruning Sweep: Score Distribution per (Pruning Step × Dendrite Count)")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    legend_handles: List[Line2D] = []
    for row_idx in sorted(set(b["dendrite_row"] for b in boxes)):
        color = _series_color(row_idx + 1)
        lbl = "PAI: 0 dendrites added" if row_idx == 0 else f"PAI: {row_idx} dendrite{'s' if row_idx > 1 else ''} added"
        legend_handles.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor=color,
                   markersize=10, label=lbl)
        )
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return csv_path



# ---------------------------------------------------------------------------
# "If at most N dendrites are allowed" filled average graph
# ---------------------------------------------------------------------------

def _make_filled_average_plot(
    dendrite_by_prune_row: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]],
    output_path: str,
) -> str:
    """Average graph where missing higher-dendrite scores are filled forward.

    For each run at a given prune step, if it only reached dendrite count M,
    its score for all counts > M is treated as the score at M.  This produces
    an "if at most N dendrites are allowed" view using every run in every bar.

    Row 0 is drawn as a connected line across prune steps; rows 1+ are dots.
    Returns the path to the companion CSV.
    """
    prune_indices = sorted(set(k[0] for k in dendrite_by_prune_row))
    max_dend_row = max((k[1] for k in dendrite_by_prune_row), default=0)

    # Build filled scores: {(prune_idx, row_idx): {run_name: (param, score)}}
    filled: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}

    for prune_idx in prune_indices:
        base_runs = dendrite_by_prune_row.get((prune_idx, 0), {})
        if not base_runs:
            continue
        for run_name, (base_param, base_score) in base_runs.items():
            best_param = base_param
            best_score = base_score
            for row_idx in range(0, max_dend_row + 1):
                actual = dendrite_by_prune_row.get((prune_idx, row_idx), {}).get(run_name)
                if actual is not None:
                    best_param, best_score = actual
                filled.setdefault((prune_idx, row_idx), {})[run_name] = (best_param, best_score)

    fig, ax = plt.subplots(figsize=(14, 8))
    legend_handles: List[Line2D] = []
    csv_rows: List[Dict] = []

    all_row_indices = sorted(set(k[1] for k in filled))

    # Row 0: connected line
    if 0 in all_row_indices:
        color = _series_color(1)
        row0_pts: List[Tuple[float, float]] = []
        for prune_idx in sorted(k[0] for k in filled if k[1] == 0):
            run_data = filled[(prune_idx, 0)]
            avg_p = float(np.mean([v[0] for v in run_data.values()]))
            avg_a = float(np.mean([v[1] for v in run_data.values()]))
            row0_pts.append((avg_p, avg_a))
            csv_rows.append({
                "prune_idx": prune_idx,
                "dendrite_row": 0,
                "n_runs": len(run_data),
                "avg_param_count": avg_p,
                "avg_score": avg_a,
            })
        if row0_pts:
            sorted_pts = sorted(row0_pts, key=lambda p: p[0])
            ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                    color=color, linewidth=2, zorder=2, alpha=0.8)
            ax.scatter([p[0] for p in row0_pts], [p[1] for p in row0_pts],
                       c=color, s=80, zorder=3, marker="o", alpha=0.9)
            legend_handles.append(
                Line2D([0], [0], marker="o", color=color, markerfacecolor=color,
                       markersize=10, label="PAI: 0 dendrites (filled avg)")
            )

    # Rows 1+: individual dots
    for row_idx in all_row_indices:
        if row_idx == 0:
            continue
        color = _series_color(row_idx + 1)
        any_plotted = False
        for prune_idx in sorted(k[0] for k in filled if k[1] == row_idx):
            run_data = filled[(prune_idx, row_idx)]
            avg_p = float(np.mean([v[0] for v in run_data.values()]))
            avg_a = float(np.mean([v[1] for v in run_data.values()]))
            ax.scatter([avg_p], [avg_a], c=color, s=80, zorder=3, marker="^", alpha=0.9)
            csv_rows.append({
                "prune_idx": prune_idx,
                "dendrite_row": row_idx,
                "n_runs": len(run_data),
                "avg_param_count": avg_p,
                "avg_score": avg_a,
            })
            any_plotted = True
        if any_plotted:
            lbl = f"PAI: up to {row_idx} dendrite{'s' if row_idx > 1 else ''} (filled avg)"
            legend_handles.append(
                Line2D([0], [0], marker="^", color="w", markerfacecolor=color,
                       markersize=10, label=lbl)
            )

    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pruning Sweep: \"If at Most N Dendrites Allowed\" Average Score")
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Improvement stacked bar graph
# ---------------------------------------------------------------------------

def _make_improvement_bar_plot(
    dendrite_by_prune_row: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]],
    output_path: str,
) -> str:
    """Create a stacked bar graph showing average score improvement from d0 with filled forward.

    Uses "filled forward" logic: if a run only reaches dendrite M, its score at M
    is used for all dendrite counts > M. This shows "if at most N dendrites allowed".

    One bar per pruning iteration (prune 8 leftmost, prune 0 rightmost).
    Each bar is stacked by dendrite count: bottom segment = avg(d1-d0),
    next segment = avg(d2-d0) - avg(d1-d0), etc.

    Returns the path to the companion CSV.
    """
    prune_indices = sorted(set(k[0] for k in dendrite_by_prune_row), reverse=True)
    max_dend_row = max((k[1] for k in dendrite_by_prune_row), default=0)

    # Build filled scores: for each run, fill forward its best score
    filled: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}
    for prune_idx in prune_indices:
        base_runs = dendrite_by_prune_row.get((prune_idx, 0), {})
        if not base_runs:
            continue
        for run_name, (base_param, base_score) in base_runs.items():
            best_param = base_param
            best_score = base_score
            for row_idx in range(0, max_dend_row + 1):
                actual = dendrite_by_prune_row.get((prune_idx, row_idx), {}).get(run_name)
                if actual is not None:
                    best_param, best_score = actual
                filled.setdefault((prune_idx, row_idx), {})[run_name] = (best_param, best_score)

    bars: List[Dict] = []
    csv_rows: List[Dict] = []

    for prune_idx in prune_indices:
        base_runs = filled.get((prune_idx, 0), {})
        if not base_runs:
            continue

        # Per-run diffs: {row_idx: [diff_run0, diff_run1, ...]}
        per_dend_diffs: Dict[int, List[float]] = {}

        for run_name, (_, base_score) in base_runs.items():
            for row_idx in range(1, max_dend_row + 1):
                key = (prune_idx, row_idx)
                if key not in filled or run_name not in filled[key]:
                    break
                _, score = filled[key][run_name]
                per_dend_diffs.setdefault(row_idx, []).append((score - base_score) * 100.0)

        if not per_dend_diffs:
            continue

        avg_diffs = {row_idx: float(np.mean(diffs)) for row_idx, diffs in per_dend_diffs.items()}
        bars.append({"prune_idx": prune_idx, "avg_diffs": avg_diffs})
        for row_idx, avg_diff in sorted(avg_diffs.items()):
            csv_rows.append({
                "prune_idx": prune_idx,
                "dendrite_row": row_idx,
                "avg_improvement_from_d0": avg_diff,
            })

    if not bars:
        raise ValueError("No improvement data to plot.")

    all_dend_rows = sorted(set(row_idx for b in bars for row_idx in b["avg_diffs"]))

    fig, ax = plt.subplots(figsize=(max(10, len(bars) * 1.2), 7))
    x = list(range(len(bars)))
    x_labels = [str(b["prune_idx"]) for b in bars]
    bottoms = [0.0] * len(bars)

    for row_idx in all_dend_rows:
        heights = []
        for b in bars:
            cum = b["avg_diffs"].get(row_idx, 0.0)
            prev_cum = b["avg_diffs"].get(row_idx - 1, 0.0) if row_idx > 1 else 0.0
            heights.append(max(0.0, cum - prev_cum))
        color = _series_color(row_idx + 1)
        ax.bar(x, heights, bottom=bottoms, color=color, alpha=0.85,
               label=f"d{row_idx - 1} \u2192 d{row_idx}")
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Pruning Iteration")
    ax.set_ylabel("Average Improvement from d0 (percentage points)")
    ax.set_title("Average Improvement by Adding Up To N Dendrites per Pruning Iteration")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Error Reduction stacked bar graph
# ---------------------------------------------------------------------------

def _make_error_reduction_bar_plot(
    dendrite_by_prune_row: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]],
    output_path: str,
) -> str:
    """Create a stacked bar graph showing average Remaining Error Reduction from d0 with filled forward.

    Uses "filled forward" logic: if a run only reaches dendrite M, its score at M
    is used for all dendrite counts > M. This shows "if at most N dendrites allowed".

    Measures the reduction in remaining error rather than raw percentage points.
    For example, 90->91 is 10% error reduction (from 10% error to 9% error),
    while 80->81 is only 5% error reduction (from 20% error to 19% error).

    One bar per pruning iteration (prune 8 leftmost, prune 0 rightmost).
    Each bar is stacked by dendrite count: bottom segment = avg(d1-d0) error reduction,
    next segment = avg(d2-d0) - avg(d1-d0) error reduction, etc.

    Returns the path to the companion CSV.
    """
    prune_indices = sorted(set(k[0] for k in dendrite_by_prune_row), reverse=True)
    max_dend_row = max((k[1] for k in dendrite_by_prune_row), default=0)

    # Build filled scores: for each run, fill forward its best score
    filled: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}
    for prune_idx in prune_indices:
        base_runs = dendrite_by_prune_row.get((prune_idx, 0), {})
        if not base_runs:
            continue
        for run_name, (base_param, base_score) in base_runs.items():
            best_param = base_param
            best_score = base_score
            for row_idx in range(0, max_dend_row + 1):
                actual = dendrite_by_prune_row.get((prune_idx, row_idx), {}).get(run_name)
                if actual is not None:
                    best_param, best_score = actual
                filled.setdefault((prune_idx, row_idx), {})[run_name] = (best_param, best_score)

    bars: List[Dict] = []
    csv_rows: List[Dict] = []

    for prune_idx in prune_indices:
        base_runs = filled.get((prune_idx, 0), {})
        if not base_runs:
            continue

        # Per-run error reductions: {row_idx: [error_reduction_run0, error_reduction_run1, ...]}
        per_dend_error_reductions: Dict[int, List[float]] = {}

        for run_name, (_, base_score) in base_runs.items():
            base_error = 1.0 - base_score
            if base_error <= 0:
                # If baseline is already 100%, can't measure error reduction
                continue
            
            for row_idx in range(1, max_dend_row + 1):
                key = (prune_idx, row_idx)
                if key not in filled or run_name not in filled[key]:
                    break
                _, score = filled[key][run_name]
                # Calculate error reduction: (improvement / baseline_error) * 100
                # This gives the percentage of remaining error that was eliminated
                error_reduction = ((score - base_score) / base_error) * 100.0
                per_dend_error_reductions.setdefault(row_idx, []).append(error_reduction)

        if not per_dend_error_reductions:
            continue

        avg_error_reductions = {row_idx: float(np.mean(reductions)) 
                                for row_idx, reductions in per_dend_error_reductions.items()}
        bars.append({"prune_idx": prune_idx, "avg_error_reductions": avg_error_reductions})
        for row_idx, avg_reduction in sorted(avg_error_reductions.items()):
            csv_rows.append({
                "prune_idx": prune_idx,
                "dendrite_row": row_idx,
                "avg_error_reduction_from_d0": avg_reduction,
            })

    if not bars:
        raise ValueError("No error reduction data to plot.")

    all_dend_rows = sorted(set(row_idx for b in bars for row_idx in b["avg_error_reductions"]))

    fig, ax = plt.subplots(figsize=(max(10, len(bars) * 1.2), 7))
    x = list(range(len(bars)))
    x_labels = [str(b["prune_idx"]) for b in bars]
    bottoms = [0.0] * len(bars)

    for row_idx in all_dend_rows:
        heights = []
        for b in bars:
            cum = b["avg_error_reductions"].get(row_idx, 0.0)
            prev_cum = b["avg_error_reductions"].get(row_idx - 1, 0.0) if row_idx > 1 else 0.0
            heights.append(max(0.0, cum - prev_cum))
        color = _series_color(row_idx + 1)
        ax.bar(x, heights, bottom=bottoms, color=color, alpha=0.85,
               label=f"d{row_idx - 1} \u2192 d{row_idx}")
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Pruning Iteration")
    ax.set_ylabel("Average Remaining Error Reduction (%)")
    ax.set_title("Average Remaining Error Reduction by Adding Up To N Dendrites per Pruning Iteration")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a scatter plot from a pruning sweep output directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--folder", required=True,
        help="Path to the pruning sweep directory (contains run0/, run1/, ...).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for the scatter PNG. "
             "Defaults to a folder named after the sweep dir in the same location.",
    )
    parser.add_argument(
        "--min-percent", type=float, default=0.0,
        help="Minimum fraction of runs (0.0–1.0) that must have data for a dendrite-row "
             "group before it is included in the average and candlestick graphs.  "
             "Default 0.0 includes all groups regardless of run coverage.",
    )
    parser.add_argument(
        "--min-prune", type=int, default=None,
        help="Minimum prune index (inclusive) to include across all graphs.  "
             "Filters pruning iterations and dendrite prune-folder indices.",
    )
    parser.add_argument(
        "--max-prune", type=int, default=None,
        help="Maximum prune index (inclusive) to include across all graphs.  "
             "Filters pruning iterations and dendrite prune-folder indices.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sweep_dir = os.path.abspath(args.folder)

    if not os.path.isdir(sweep_dir):
        print(f"ERROR: {sweep_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        parent = os.path.dirname(sweep_dir)
        stem = os.path.basename(sweep_dir.rstrip("/"))
        output_dir = os.path.join(parent, stem + "_output")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading sweep data from: {sweep_dir}")
    pruning_by_iter, dendrite_by_prune_row, total_runs = _load_all_runs_grouped(sweep_dir)

    # Apply prune-index range filter.
    # "Prune N" refers to pruning_results iteration N and PAI_pruneN folder.
    min_prune = args.min_prune
    max_prune = args.max_prune
    if min_prune is not None:
        pruning_by_iter = {k: v for k, v in pruning_by_iter.items() if k >= min_prune}
        dendrite_by_prune_row = {k: v for k, v in dendrite_by_prune_row.items() if k[0] >= min_prune}
    if max_prune is not None:
        pruning_by_iter = {k: v for k, v in pruning_by_iter.items() if k <= max_prune}
        dendrite_by_prune_row = {k: v for k, v in dendrite_by_prune_row.items() if k[0] <= max_prune}

    # Reconstruct flat scatter data from the (already filtered) grouped data.
    dendrite_rows: Dict[int, List[Tuple[float, float]]] = {}
    for (prune_idx, row_idx), run_data in dendrite_by_prune_row.items():
        for (param, acc) in run_data.values():
            dendrite_rows.setdefault(row_idx, []).append((param, acc))

    total_dendrite = sum(len(v) for v in dendrite_rows.values())
    for row_idx in sorted(dendrite_rows.keys()):
        print(f"  Dendrite row {row_idx} points: {len(dendrite_rows[row_idx])}")
    print(f"  Total dendrite points: {total_dendrite}")

    scatter_path = os.path.join(output_dir, "scatter.png")
    _make_scatter(dendrite_rows, scatter_path)

    min_percent = args.min_percent

    average_path = os.path.join(output_dir, "average.png")
    _make_average_plot(dendrite_by_prune_row, total_runs, min_percent, average_path)

    candlestick_path = os.path.join(output_dir, "candlestick.png")
    _make_candlestick_plot(dendrite_by_prune_row, total_runs, min_percent, candlestick_path)

    improvement_path = os.path.join(output_dir, "improvement_bars.png")
    _make_improvement_bar_plot(dendrite_by_prune_row, improvement_path)

    error_reduction_path = os.path.join(output_dir, "error_reduction_bars.png")
    _make_error_reduction_bar_plot(dendrite_by_prune_row, error_reduction_path)

    filled_average_path = os.path.join(output_dir, "filled_average.png")
    _make_filled_average_plot(dendrite_by_prune_row, filled_average_path)


if __name__ == "__main__":
    main()
