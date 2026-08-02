#!/usr/bin/env python3
"""
Process by-dendrite-separate CSV output and generate summary plots.

This script currently supports CSV files produced by:
    get_wandb_results.py --mode by-dendrite-separate

Input format expectations:
- Row 1: metadata labels (contains entries like "param_count <column_name>")
- Row 2: metadata values (parameter counts aligned to columns)
- Row 3: header row
- Row 4+: data rows

Outputs:
- Creates a folder with the same stem as the input CSV in the same directory.
- Writes two PNG files:
  1) candlestick_by_pair.png
     Candlestick-style box summaries by model/dendrite pair label.
  2) candlestick_by_param_count.png
     Same summaries, but positioned on the x-axis by metadata param_count.

Example:
    python process_csv_output.py --csv my_sweep_by_dendrite_separate.csv
    python process_csv_output.py --csv my_sweep_by_dendrite_separate.csv --output my_output_folder
    python process_csv_output.py --csv my_sweep_by_dendrite_separate.csv --x-break 13000000,22000000
"""

import argparse
import colorsys
import csv
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


_FONT_SCALE = 1.5


def _scale_rc_value(value: Any, fallback: float) -> float:
    """Scale numeric matplotlib rc values safely."""
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


def _safe_float(value: str) -> Optional[float]:
    """Parse a float value from string, returning None on empty/invalid input."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _dendrite_sort_key(column_name: str) -> Tuple[int, int, str, int]:
    """Sort known model_*_dendrite_M columns first, with val before test."""
    match = re.match(r"(model_.+?)_dendrite_(\d+)_max_(val|test)$", column_name)
    if match:
        model_id = match.group(1)
        dendrite_idx = int(match.group(2))
        metric = match.group(3)
        metric_order = 0 if metric == "val" else 1
        numeric_model = re.match(r"model_(\d+)$", model_id)
        if numeric_model:
            model_rank = int(numeric_model.group(1))
            return (0, metric_order, "", model_rank * 100000 + dendrite_idx)
        # Keep non-numeric models (e.g. model_unknown) ordered after numeric ones.
        return (0, metric_order, model_id, dendrite_idx)
    return (1, 2, column_name, 0)


def _display_label(column_name: str, model_name_map: Optional[Dict[str, str]] = None) -> str:
    """Build a compact display label for a dendrite metric column."""
    match = re.match(r"(model_.+?)_dendrite_(\d+)_max_(?:val|test)$", column_name)
    if match:
        model_id = match.group(1)
        model_label = model_name_map.get(model_id, model_id) if model_name_map else model_id
        return f"{model_label} / dendrite_{match.group(2)}"
    return column_name.replace("_max_val", "").replace("_max_test", "")


def _load_model_info(base_dir: str) -> Tuple[Dict[str, str], Optional[str]]:
    """Load model_info.csv and return (model_name_map, initial_model_id).

    model_info.csv must have model_id and model_name columns.
    Optionally, mark one row with is_initial_model=true to flag the baseline.
    """
    model_info_path = os.path.join(base_dir, "model_info.csv")
    if not os.path.exists(model_info_path):
        return {}, None

    try:
        model_info_df = pd.read_csv(model_info_path)
    except Exception:
        return {}, None

    if not {"model_id", "model_name"}.issubset(set(model_info_df.columns)):
        return {}, None

    model_name_map: Dict[str, str] = {}
    initial_model_id: Optional[str] = None

    for _, row in model_info_df.iterrows():
        model_id = str(row["model_id"]).strip()
        model_name = str(row["model_name"]).strip()
        if not model_id or model_id.lower() == "nan":
            continue
        if model_name and model_name.lower() != "nan":
            model_name_map[model_id] = model_name
        if "is_initial_model" in model_info_df.columns:
            flag = str(row["is_initial_model"]).strip().lower()
            if flag in ("true", "1", "yes"):
                initial_model_id = model_id

    return model_name_map, initial_model_id


def _load_model_name_map(base_dir: str) -> Dict[str, str]:
    """Convenience wrapper — returns only the name map."""
    name_map, _ = _load_model_info(base_dir)
    return name_map


def _parse_model_and_dendrite(column_name: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract model_id and dendrite index from a standard dendrite metric column."""
    match = re.match(r"(model_.+?)_dendrite_(\d+)_max_(?:val|test)$", column_name)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def _model_sort_key(model_id: str) -> Tuple[int, int, str]:
    """Sort numeric model_N first, then non-numeric model IDs."""
    match = re.match(r"model_(\d+)$", model_id)
    if match:
        return (0, int(match.group(1)), "")
    return (1, 0, model_id)


def _extract_data_percent_from_run_name(run_name: str) -> Tuple[Optional[str], str]:
    """Extract data-percent color key from known run_name conventions.

    Supported conventions:
    - data_percent_<N>
    - subj_<N>_samp_<M>
    - subj_<N> or samp_<N>
    """
    text = str(run_name)

    match_data_percent = re.search(r"data_percent_([0-9]+(?:\.[0-9]+)?)", text)
    if match_data_percent:
        value = match_data_percent.group(1)
        return f"data_percent_{value}", f"{float(value):g}% data"

    match_subj = re.search(r"subj_([0-9]+(?:\.[0-9]+)?)", text)
    match_samp = re.search(r"samp_([0-9]+(?:\.[0-9]+)?)", text)

    if match_subj and match_samp:
        subj_pct = match_subj.group(1)
        samp_pct = match_samp.group(1)
        return f"subj_{subj_pct}_samp_{samp_pct}", f"subj={float(subj_pct):g}%, samp={float(samp_pct):g}%"

    if match_subj:
        subj_pct = match_subj.group(1)
        return f"subj_{subj_pct}", f"subj={float(subj_pct):g}%"

    if match_samp:
        samp_pct = match_samp.group(1)
        return f"samp_{samp_pct}", f"samp={float(samp_pct):g}%"

    return None, "unknown"


def _extract_subject_sample_from_run_name(run_name: str) -> Tuple[Optional[str], str, Optional[str], str]:
    """Extract subject/sample keys and labels from run_name.

    Returns:
        (subject_key, subject_label, sample_key, sample_label)
    """
    text = str(run_name)

    match_subj = re.search(r"subj_([0-9]+(?:\.[0-9]+)?)", text)
    match_samp = re.search(r"samp_([0-9]+(?:\.[0-9]+)?)", text)
    if match_subj and match_samp:
        subj_value = match_subj.group(1)
        samp_value = match_samp.group(1)
        return (
            f"subj_{subj_value}",
            f"subj={float(subj_value):g}%",
            f"samp_{samp_value}",
            f"samp={float(samp_value):g}%",
        )

    data_percent_key, data_percent_label = _extract_data_percent_from_run_name(text)
    if data_percent_key is not None:
        # Fallback for naming schemes that only provide a single data_percent axis.
        return data_percent_key, data_percent_label, data_percent_key, data_percent_label

    return None, "unknown", None, "unknown"


def _extract_split_label(run_name: str) -> str:
    """Extract split strategy label from run_name for coloring."""
    text = str(run_name)
    match = re.search(r"split_([^_]+)", text)
    if match:
        return match.group(1)
    return "split_unknown"


def _percent_key_sort_key(percent_key: str) -> Tuple[int, float, str]:
    """Sort keys like samp_50, subj_75, data_percent_100 by numeric value."""
    text = str(percent_key)
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", text)
    if match:
        return (0, float(match.group(1)), text)
    return (1, float("inf"), text)


def _percent_key_to_float(percent_key: str) -> Optional[float]:
    """Extract numeric percent value from keys like samp_50 or subj_75."""
    text = str(percent_key)
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", text)
    if not match:
        return None
    return float(match.group(1))


def _parse_split_values_arg(values: Optional[str], key_prefix: str) -> Optional[List[str]]:
    """Parse comma-separated percent values into normalized split keys.

    Example: "50,62,75" with key_prefix="samp" -> ["samp_50", "samp_62", "samp_75"]
    """
    if values is None:
        return None

    parsed_keys: List[str] = []
    for raw_piece in values.split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        try:
            numeric_value = float(piece)
        except ValueError as exc:
            raise ValueError(f"Invalid split value '{piece}' for {key_prefix}. Expected numeric values.") from exc
        parsed_keys.append(f"{key_prefix}_{numeric_value:g}")

    # Preserve order while dropping duplicates.
    deduped_keys = list(dict.fromkeys(parsed_keys))
    if not deduped_keys:
        raise ValueError(f"No valid split values provided for {key_prefix}.")
    return deduped_keys


def _parse_hyperparams_from_run_name(run_name: str, known_keys: Sequence[str]) -> Dict[str, str]:
    """Parse key/value hyperparameters encoded in run_name."""
    text = str(run_name)
    parsed: Dict[str, str] = {}
    if not text:
        return parsed

    escaped_keys = [re.escape(k) for k in known_keys]
    key_union = "|".join(escaped_keys)

    for key in known_keys:
        pattern = rf"{re.escape(key)}_(.*?)(?=_(?:{key_union})_|$)"
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value != "":
                parsed[key] = value

    return parsed


def _write_best_hyperparameter_summary(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    output_dir: str,
    model_name_map: Dict[str, str],
) -> str:
    """Write one CSV with best-by-val and best-by-test hyperparameters per model."""

    def _extract_model_id_from_row(row: pd.Series) -> Optional[str]:
        run_name = str(row.get("run_name", ""))
        m = re.search(r"model_index_(\d+)", run_name)
        if m:
            return f"model_{m.group(1)}"
        return None

    # Build model -> metric columns map from standard column names.
    model_metric_columns: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"val": [], "test": []})
    for col in dendrite_columns:
        m = re.match(r"(model_\d+)_dendrite_\d+_max_(val|test)$", col)
        if m:
            model_metric_columns[m.group(1)][m.group(2)].append(col)

    # Build model groups from rows.
    model_to_row_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        model_id = _extract_model_id_from_row(row)
        if model_id:
            model_to_row_indices[model_id].append(idx)

    if not model_to_row_indices:
        raise ValueError("Could not identify model_index in run_name for best-hyperparameter summary.")

    # Determine potential hyperparameter keys from run_name strings.
    common_hparam_keys = [
        "model_index",
        "dataset",
        "data_percent",
        "lr",
        "weight_decay",
        "label_smoothing",
        "scheduler_mode",
        "improvement_threshold",
        "pai_forward_function",
        "batch_size",
        "epochs",
        "lr_warmup_epochs",
    ]

    rows_output: List[Dict[str, Any]] = []
    all_hparam_keys_seen = set()

    for model_id in sorted(model_to_row_indices.keys(), key=lambda m: int(m.split("_")[1])):
        model_rows = df.loc[model_to_row_indices[model_id]]
        metric_cols_for_model = model_metric_columns.get(model_id, {"val": [], "test": []})

        for metric_key in ("val", "test"):
            metric_cols = metric_cols_for_model.get(metric_key, [])
            if not metric_cols:
                continue

            # Per-row best score for this metric across dendrite counts for this model.
            per_row_best = model_rows[metric_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
            per_row_best = per_row_best.dropna()
            if per_row_best.empty:
                continue

            best_idx = per_row_best.idxmax()
            best_row = df.loc[best_idx]
            best_score = float(per_row_best.loc[best_idx])

            # Also capture opposite metric score for the same selected run (for reference).
            other_metric_key = "test" if metric_key == "val" else "val"
            other_metric_cols = metric_cols_for_model.get(other_metric_key, [])
            other_score = None
            if other_metric_cols:
                other_score_series = pd.to_numeric(best_row[other_metric_cols], errors="coerce")
                if not other_score_series.dropna().empty:
                    other_score = float(other_score_series.max())

            parsed_hparams = _parse_hyperparams_from_run_name(str(best_row.get("run_name", "")), common_hparam_keys)
            all_hparam_keys_seen.update(parsed_hparams.keys())

            row_out: Dict[str, Any] = {
                "model_id": model_id,
                "model_name": model_name_map.get(model_id, model_id),
                "selected_for": f"best_{metric_key}",
                "run_id": str(best_row.get("run_id", "")),
                "run_name": str(best_row.get("run_name", "")),
                "final_val_score": "",
                "final_test_score": "",
                "selected_score": round(best_score, 6),
                "opposite_metric_score": round(other_score, 6) if other_score is not None else "",
            }

            if metric_key == "val":
                row_out["final_val_score"] = round(best_score, 6)
            else:
                row_out["final_test_score"] = round(best_score, 6)

            for key, value in parsed_hparams.items():
                row_out[key] = value

            rows_output.append(row_out)

    if not rows_output:
        raise ValueError("No best-hyperparameter rows could be produced from input data.")

    ordered_hparams = [k for k in common_hparam_keys if k in all_hparam_keys_seen]
    extra_hparams = sorted(k for k in all_hparam_keys_seen if k not in ordered_hparams)
    ordered_hparams.extend(extra_hparams)

    fixed_columns = [
        "model_id",
        "model_name",
        "selected_for",
        "run_id",
        "run_name",
    ]
    score_columns = ["final_val_score", "final_test_score", "selected_score", "opposite_metric_score"]
    output_columns = fixed_columns + ordered_hparams + score_columns

    out_path = os.path.join(output_dir, "best_hyperparameters_by_model_val_test.csv")
    pd.DataFrame(rows_output, columns=output_columns).to_csv(out_path, index=False)
    return out_path


def _build_column_color_map(dendrite_columns: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    """Build per-column colors: rainbow hue by model, brightness by dendrite within each model."""
    model_to_columns: Dict[str, List[Tuple[int, str]]] = {}
    unknown_columns: List[str] = []

    for col in dendrite_columns:
        model_id, dendrite_idx = _parse_model_and_dendrite(col)
        if model_id is None or dendrite_idx is None:
            unknown_columns.append(col)
            continue
        model_to_columns.setdefault(model_id, []).append((dendrite_idx, col))

    def _model_sort_key(model_id: str) -> Tuple[int, int, str]:
        m = re.match(r"model_(\d+)$", model_id)
        if m:
            return (0, int(m.group(1)), model_id)
        return (1, 0, model_id)

    model_ids = sorted(model_to_columns.keys(), key=_model_sort_key)
    n_models = max(1, len(model_ids))
    color_map: Dict[str, Tuple[float, float, float]] = {}

    for i, model_id in enumerate(model_ids):
        hue = i / n_models
        columns_for_model = sorted(model_to_columns[model_id], key=lambda x: x[0])
        n_dendrites = max(1, len(columns_for_model))

        for j, (_, col) in enumerate(columns_for_model):
            if n_dendrites == 1:
                value = 1.0
            else:
                # Keep all shades clearly colorful (avoid near-black).
                value = 1.0 - (0.35 * (j / (n_dendrites - 1)))
            rgb = colorsys.hsv_to_rgb(hue, 0.8, value)
            color_map[col] = rgb

    for col in unknown_columns:
        color_map[col] = (0.4, 0.4, 0.4)

    return color_map


def _read_by_dendrite_separate_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Read CSV with top metadata rows and return data + dendrite column metadata."""
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 3:
        raise ValueError("CSV does not have the expected metadata + header + data layout.")

    metadata_label_row = rows[0]
    metadata_value_row = rows[1]
    header_row = rows[2]
    data_rows = rows[3:]

    # Pad rows to header width so index-based mapping is safe.
    header_len = len(header_row)
    if len(metadata_label_row) < header_len:
        metadata_label_row += [""] * (header_len - len(metadata_label_row))
    if len(metadata_value_row) < header_len:
        metadata_value_row += [""] * (header_len - len(metadata_value_row))

    normalized_data_rows = []
    for row in data_rows:
        if len(row) < header_len:
            row = row + [""] * (header_len - len(row))
        elif len(row) > header_len:
            row = row[:header_len]
        normalized_data_rows.append(row)

    df = pd.DataFrame(normalized_data_rows, columns=header_row)

    dendrite_columns = [
        col for col in df.columns
        if (col.endswith("_max_val") or col.endswith("_max_test")) and "dendrite" in col
    ]

    if not dendrite_columns:
        raise ValueError(
            "No dendrite metric columns found. Expected columns like "
            "model_0_dendrite_2_max_val or model_0_dendrite_2_max_test."
        )

    dendrite_columns = sorted(dendrite_columns, key=_dendrite_sort_key)

    # Extract param_count values from metadata rows using column position.
    param_counts_by_column: Dict[str, float] = {}
    header_index = {name: idx for idx, name in enumerate(header_row)}

    for col in dendrite_columns:
        idx = header_index[col]
        label = metadata_label_row[idx].strip()
        value = metadata_value_row[idx].strip()

        # Prefer explicit metadata labeling, but still allow direct value parsing.
        parsed_value = _safe_float(value)
        if label.startswith("param_count") and parsed_value is not None:
            param_counts_by_column[col] = parsed_value
        elif parsed_value is not None:
            param_counts_by_column[col] = parsed_value

    # Convert metric columns to numeric.
    for col in dendrite_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, dendrite_columns, param_counts_by_column


def _build_box_stats(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    model_name_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, float]]:
    """Compute box/candlestick statistics per dendrite column."""
    stats: List[Dict[str, float]] = []

    for col in dendrite_columns:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        med = float(series.quantile(0.50))
        q3 = float(series.quantile(0.75))

        stats.append({
            "column": col,
            "label": _display_label(col, model_name_map),
            "whislo": float(series.min()),
            "q1": q1,
            "med": med,
            "q3": q3,
            "whishi": float(series.max()),
        })

    if not stats:
        raise ValueError("No non-empty dendrite metric data found to plot.")

    return stats


def _compute_within_model_stats(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    param_counts_by_column: Dict[str, float],
    model_name_map: Dict[str, str],
    initial_model_id: Optional[str],
    output_dir: str,
    x_break: Optional[Tuple[float, float]] = None,
    column_color_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    output_suffix: str = "",
    metric_label: str = "Val",
) -> List[str]:
    """Compute three within-model comparison metrics and save CSV + bar charts.

    Returns a list of output file paths created.
    """
    # --- build model -> [(dendrite_idx, col)] lookup ---
    model_dendrite_cols: Dict[str, List[Tuple[int, str]]] = {}
    for col in dendrite_columns:
        model_id, dendrite_idx = _parse_model_and_dendrite(col)
        if model_id is None or dendrite_idx is None:
            continue
        model_dendrite_cols.setdefault(model_id, []).append((dendrite_idx, col))
    for mid in model_dendrite_cols:
        model_dendrite_cols[mid].sort(key=lambda x: x[0])

    def _get_model_id(run_name: str) -> Optional[str]:
        m = re.search(r"model_index_(\d+)", str(run_name))
        return f"model_{m.group(1)}" if m else None

    def _model_order(model_id: str) -> int:
        m = re.match(r"model_(\d+)$", model_id)
        return int(m.group(1)) if m else 9999

    # --- collect per-run score progressions ---
    # run_scores[run_id] = {model_id: str, scores: {dendrite_idx: float}}
    run_scores: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        run_id = str(row.get("run_id", ""))
        run_name = str(row.get("run_name", ""))
        model_id = _get_model_id(run_name)
        if model_id is None:
            continue
        if run_id not in run_scores:
            run_scores[run_id] = {"model_id": model_id, "scores": {}}
        if model_id in model_dendrite_cols:
            for dendrite_idx, col in model_dendrite_cols[model_id]:
                val = pd.to_numeric(row.get(col, None), errors="coerce")
                if not pd.isna(val):
                    run_scores[run_id]["scores"][dendrite_idx] = float(val)

    # group runs by model
    model_runs: Dict[str, List[Dict]] = {}
    for data in run_scores.values():
        model_runs.setdefault(data["model_id"], []).append(data)

    if column_color_map is None:
        column_color_map = {}

    # --- Metric 1 & 2: per-model, per-run comparisons ---
    # metric1: % runs where any higher-dendrite score beats lowest-dendrite score
    # metric2: avg error reduction from lowest-dendrite to each subsequent dendrite

    metric1_rows: List[Dict] = []
    metric2_rows: List[Dict] = []

    for model_id in sorted(model_runs.keys(), key=_model_order):
        model_label = model_name_map.get(model_id, model_id)
        improved = 0
        total_multi = 0
        reductions_by_dendrite: Dict[int, List[float]] = {}

        for data in model_runs[model_id]:
            scores = data["scores"]
            if len(scores) < 2:
                continue
            sorted_d = sorted(scores.keys())
            base_d = sorted_d[0]
            base_score = scores[base_d]
            base_error = 100.0 - base_score
            total_multi += 1
            any_improved = False

            for d in sorted_d[1:]:
                s = scores[d]
                if s > base_score:
                    any_improved = True
                if base_error > 0:
                    reduction = (base_error - (100.0 - s)) / base_error
                    reductions_by_dendrite.setdefault(d, []).append(reduction)

            if any_improved:
                improved += 1

        pct = (improved / total_multi * 100.0) if total_multi > 0 else None
        metric1_rows.append({
            "model_id": model_id,
            "model_name": model_label,
            "runs_with_multiple_dendrites": total_multi,
            "runs_improved": improved,
            "pct_improved": round(pct, 2) if pct is not None else "",
        })

        for d, reds in sorted(reductions_by_dendrite.items()):
            avg_red = sum(reds) / len(reds) * 100.0
            metric2_rows.append({
                "model_id": model_id,
                "model_name": model_label,
                "dendrite_count": d,
                "n_runs": len(reds),
                "avg_error_reduction_pct": round(avg_red, 4),
            })

    # --- Metric 3: per-model error reduction per 1M params vs each model baseline dendrite ---
    metric3_rows: List[Dict] = []

    # Average score per (model, dendrite) across all runs
    avg_scores: Dict[str, Dict[int, float]] = {}
    for model_id, runs in model_runs.items():
        dendrite_values: Dict[int, List[float]] = {}
        for data in runs:
            for d, s in data["scores"].items():
                dendrite_values.setdefault(d, []).append(s)
        for d, vals in dendrite_values.items():
            if vals:
                avg_scores.setdefault(model_id, {})[d] = sum(vals) / len(vals)

    for model_id in sorted(model_runs.keys(), key=_model_order):
        model_label = model_name_map.get(model_id, model_id)
        if model_id not in model_dendrite_cols:
            continue
        # Metric 3 only applies to model types that actually add dendrites.
        if len(model_dendrite_cols[model_id]) <= 1:
            continue

        baseline_d, baseline_col = model_dendrite_cols[model_id][0]
        baseline_score = avg_scores.get(model_id, {}).get(baseline_d)
        baseline_params = param_counts_by_column.get(baseline_col)
        if baseline_score is None or baseline_params is None:
            continue

        baseline_error = 100.0 - baseline_score
        if baseline_error <= 0:
            continue

        for d_idx, col in model_dendrite_cols[model_id][1:]:
            avg_s = avg_scores.get(model_id, {}).get(d_idx)
            params = param_counts_by_column.get(col)
            if avg_s is None or params is None:
                continue
            extra_params = params - baseline_params
            if extra_params <= 0:
                continue

            error_reduction = (baseline_error - (100.0 - avg_s)) / baseline_error * 100.0
            metric3_rows.append({
                "model_id": model_id,
                "model_name": model_label,
                "baseline_dendrite_count": baseline_d,
                "dendrite_count": d_idx,
                "baseline_param_count": int(baseline_params),
                "param_count": int(params),
                "extra_params_vs_baseline": int(extra_params),
                "baseline_avg_score": round(baseline_score, 4),
                "avg_score": round(avg_s, 4),
                "error_reduction_pct": round(error_reduction, 4),
                "error_reduction_per_param": round(error_reduction / extra_params, 8),
            })

    # --- Metric 4: baseline top-percentile thresholds and beat rates by model/dendrite combo ---
    metric4_rows: List[Dict] = []

    if initial_model_id and initial_model_id in model_dendrite_cols:
        baseline_d, _baseline_col = model_dendrite_cols[initial_model_id][0]

        baseline_scores = [
            data["scores"][baseline_d]
            for data in model_runs.get(initial_model_id, [])
            if baseline_d in data["scores"]
        ]

        if baseline_scores:
            baseline_scores_sorted = sorted(baseline_scores)

            def _threshold_for_top_percent(sorted_scores: List[float], top_percent: float) -> Tuple[float, int, int]:
                n = len(sorted_scores)
                # Use ceil with a minimum of 1 so small-n experiments still have a top bucket.
                top_n = max(1, int(math.ceil(n * top_percent)))
                # Threshold is the best score just below the top bucket.
                threshold_rank_1based = max(1, n - top_n)
                threshold = sorted_scores[threshold_rank_1based - 1]
                return threshold, threshold_rank_1based, top_n

            percentile_specs = [
                (0.01, "top_1pct"),
                (0.05, "top_5pct"),
            ]

            for pct, label in percentile_specs:
                threshold, rank_1based, target_top_n = _threshold_for_top_percent(baseline_scores_sorted, pct)
                baseline_n_above = sum(1 for s in baseline_scores_sorted if s > threshold)
                baseline_n_at_or_above = sum(1 for s in baseline_scores_sorted if s >= threshold)

                n_scores = len(baseline_scores_sorted)
                top_x_plus_one_count = min(n_scores, target_top_n + 1)
                top_x_plus_one_scores = sorted(baseline_scores_sorted, reverse=True)[:top_x_plus_one_count]
                print(
                    f"Baseline threshold details for {label}: "
                    f"n_runs={n_scores}, top_n={target_top_n}, "
                    f"threshold_rank_1based={rank_1based}, threshold_score={threshold:.6f}"
                )
                print(
                    f"  Top {top_x_plus_one_count} baseline scores (desc, top_n+1 view): "
                    + ", ".join(f"{s:.6f}" for s in top_x_plus_one_scores)
                )
                print(
                    f"  Baseline counts: > threshold = {baseline_n_above}, "
                    f">= threshold = {baseline_n_at_or_above}"
                )

                for model_id in sorted(model_runs.keys(), key=_model_order):
                    for d_idx, col in model_dendrite_cols.get(model_id, []):
                        if model_id == initial_model_id and d_idx == baseline_d:
                            continue

                        combo_scores = [
                            data["scores"][d_idx]
                            for data in model_runs.get(model_id, [])
                            if d_idx in data["scores"]
                        ]
                        if not combo_scores:
                            continue

                        beats = sum(1 for s in combo_scores if s > threshold)
                        pct_beats = beats / len(combo_scores) * 100.0

                        metric4_rows.append({
                            "percentile_label": label,
                            "percentile": pct,
                            "baseline_model_id": initial_model_id,
                            "baseline_model_name": model_name_map.get(initial_model_id, initial_model_id),
                            "baseline_dendrite_count": baseline_d,
                            "baseline_n_runs": len(baseline_scores_sorted),
                            "baseline_target_top_n": target_top_n,
                            "baseline_threshold_rank_1based": rank_1based,
                            "baseline_threshold_score": round(float(threshold), 6),
                            "baseline_n_above_threshold": baseline_n_above,
                            "baseline_n_at_or_above_threshold": baseline_n_at_or_above,
                            "baseline_param_count": int(param_counts_by_column.get(model_dendrite_cols[initial_model_id][0][1], 0)),
                            "model_id": model_id,
                            "model_name": model_name_map.get(model_id, model_id),
                            "dendrite_count": d_idx,
                            "combo_column": col,
                            "combo_param_count": int(param_counts_by_column.get(col, 0)),
                            "combo_n_runs": len(combo_scores),
                            "combo_n_beating_threshold": beats,
                            "combo_pct_beating_threshold": round(pct_beats, 4),
                        })

    # --- save CSVs ---
    created: List[str] = []

    csv1_path = os.path.join(output_dir, f"stats_pct_improved{output_suffix}.csv")
    pd.DataFrame(metric1_rows).to_csv(csv1_path, index=False)
    created.append(csv1_path)

    csv2_path = os.path.join(output_dir, f"stats_error_reduction{output_suffix}.csv")
    pd.DataFrame(metric2_rows).to_csv(csv2_path, index=False)
    created.append(csv2_path)

    csv3_path = os.path.join(output_dir, f"stats_error_reduction_per_param{output_suffix}.csv")
    pd.DataFrame(metric3_rows).to_csv(csv3_path, index=False)
    created.append(csv3_path)

    csv4_columns = [
        "percentile_label",
        "percentile",
        "baseline_model_id",
        "baseline_model_name",
        "baseline_dendrite_count",
        "baseline_n_runs",
        "baseline_target_top_n",
        "baseline_threshold_rank_1based",
        "baseline_threshold_score",
        "baseline_n_above_threshold",
        "baseline_n_at_or_above_threshold",
        "baseline_param_count",
        "model_id",
        "model_name",
        "dendrite_count",
        "combo_column",
        "combo_param_count",
        "combo_n_runs",
        "combo_n_beating_threshold",
        "combo_pct_beating_threshold",
    ]
    csv4_path = os.path.join(output_dir, f"stats_top_percentile_vs_baseline{output_suffix}.csv")
    pd.DataFrame(metric4_rows, columns=csv4_columns).to_csv(csv4_path, index=False)
    created.append(csv4_path)

    # --- Chart 1: % improved per model ---
    if metric1_rows:
        labels1 = [r["model_name"] for r in metric1_rows]
        values1 = [float(r["pct_improved"]) if r["pct_improved"] != "" else 0.0 for r in metric1_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels1) * 1.2), 5))
        bars = ax.bar(labels1, values1)
        ax.set_ylim(0, 110)
        ax.set_title(f"% of Runs Improved by Adding Dendrites ({metric_label})")
        ax.set_xlabel("Model")
        ax.set_ylabel("% Runs Improved")
        for bar, val in zip(bars, values1):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        chart1_path = os.path.join(output_dir, f"stats_chart_pct_improved{output_suffix}.png")
        fig.savefig(chart1_path, dpi=150)
        plt.close(fig)
        created.append(chart1_path)

    # --- Chart 2: avg error reduction by model + dendrite ---
    if metric2_rows:
        df2 = pd.DataFrame(metric2_rows)
        models_ordered = sorted(df2["model_id"].unique(), key=_model_order)
        dendrites_ordered = sorted(df2["dendrite_count"].unique())
        x = range(len(models_ordered))
        width = 0.8 / max(1, len(dendrites_ordered))
        fig, ax = plt.subplots(figsize=(max(7, len(models_ordered) * 1.5), 5))
        for i, d in enumerate(dendrites_ordered):
            subset = df2[df2["dendrite_count"] == d]
            vals = []
            for mid in models_ordered:
                row = subset[subset["model_id"] == mid]
                vals.append(float(row["avg_error_reduction_pct"].iloc[0]) if not row.empty else 0.0)
            offset = (i - (len(dendrites_ordered) - 1) / 2.0) * width
            positions = [xi + offset for xi in x]
            ax.bar(positions, vals, width=width * 0.9, label=f"dendrite {d}")
        ax.set_xticks(list(x))
        ax.set_xticklabels([model_name_map.get(mid, mid) for mid in models_ordered], rotation=20, ha="right")
        ax.set_title(f"Avg Error Reduction (%) by Adding Dendrites ({metric_label})")
        ax.set_xlabel("Model")
        ax.set_ylabel("Avg Error Reduction %")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        chart2_path = os.path.join(output_dir, f"stats_chart_error_reduction{output_suffix}.png")
        fig.savefig(chart2_path, dpi=150)
        plt.close(fig)
        created.append(chart2_path)

    # --- Chart 3: per-model error reduction per parameter vs baseline dendrite ---
    if metric3_rows:
        df3 = pd.DataFrame(metric3_rows)
        models_ordered = sorted(df3["model_id"].unique(), key=_model_order)
        dendrites_ordered = sorted(df3["dendrite_count"].unique())
        x = range(len(models_ordered))
        width = 0.8 / max(1, len(dendrites_ordered))
        fig, ax = plt.subplots(figsize=(max(7, len(models_ordered) * 1.5), 5))

        for i, d in enumerate(dendrites_ordered):
            subset = df3[df3["dendrite_count"] == d]
            vals = []
            for mid in models_ordered:
                row = subset[subset["model_id"] == mid]
                vals.append(float(row["error_reduction_per_param"].iloc[0]) if not row.empty else 0.0)
            offset = (i - (len(dendrites_ordered) - 1) / 2.0) * width
            positions = [xi + offset for xi in x]
            ax.bar(positions, vals, width=width * 0.9, label=f"dendrite {d}")

        ax.set_xticks(list(x))
        ax.set_xticklabels([model_name_map.get(mid, mid) for mid in models_ordered], rotation=20, ha="right")
        ax.set_title(f"Error Reduction per Parameter vs Model Baseline Dendrite ({metric_label})")
        ax.set_xlabel("Model")
        ax.set_ylabel("Error Reduction % per Parameter")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        chart3_path = os.path.join(output_dir, f"stats_chart_error_reduction_per_param{output_suffix}.png")
        fig.savefig(chart3_path, dpi=150)
        plt.close(fig)
        created.append(chart3_path)

    # --- Chart 4/5: % of scores beating baseline top-1% and top-5% thresholds ---
    if metric4_rows:
        df4 = pd.DataFrame(metric4_rows)

        def _plot_metric4_param_scatter(
            subset_df: pd.DataFrame,
            title_suffix: str,
            out_name: str,
            baseline_y: float,
        ) -> Optional[str]:
            if subset_df.empty:
                return None

            rows = subset_df.to_dict("records")
            x_values = [float(r["combo_param_count"]) for r in rows if float(r["combo_param_count"]) > 0]
            if not x_values:
                return None

            baseline_x = float(subset_df["baseline_param_count"].iloc[0])
            threshold_score = float(subset_df["baseline_threshold_score"].iloc[0])
            baseline_name = str(subset_df["baseline_model_name"].iloc[0])
            baseline_d = int(subset_df["baseline_dendrite_count"].iloc[0])
            baseline_n_runs = int(subset_df["baseline_n_runs"].iloc[0])
            baseline_target_top_n = int(subset_df["baseline_target_top_n"].iloc[0])
            baseline_n_above = int(subset_df["baseline_n_above_threshold"].iloc[0])
            baseline_n_at_or_above = int(subset_df["baseline_n_at_or_above_threshold"].iloc[0])

            x_min = min(x_values + [baseline_x])
            x_max = max(x_values + [baseline_x])
            x_span = x_max - x_min

            def _add_scatter_legend(ax, data_rows: List[Dict]) -> None:
                handles: List[Line2D] = []
                labels: List[str] = []
                seen_cols = set()

                for r in data_rows:
                    col = str(r.get("combo_column", ""))
                    if not col or col in seen_cols:
                        continue
                    seen_cols.add(col)

                    model_name = str(r.get("model_name", ""))
                    dendrite_count = int(r.get("dendrite_count", 0))
                    label = f"{model_name} / d{dendrite_count}"
                    color = column_color_map.get(col, (0.2, 0.4, 0.8))

                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="none",
                            markersize=5,
                            markerfacecolor=color,
                            markeredgecolor=color,
                        )
                    )
                    labels.append(label)

                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="*",
                        linestyle="none",
                        markersize=8,
                        markerfacecolor="black",
                        markeredgecolor="black",
                    )
                )
                labels.append(f"Baseline ({baseline_name} / d{baseline_d})")

                ax.legend(handles, labels, loc="lower right", fontsize=7, framealpha=0.85)

            def _scatter_points(ax, data_rows: List[Dict]) -> None:
                for r in data_rows:
                    x = float(r["combo_param_count"])
                    y = float(r["combo_pct_beating_threshold"])
                    col = str(r.get("combo_column", ""))
                    color = column_color_map.get(col, (0.2, 0.4, 0.8))
                    ax.scatter([x], [y], s=30, color=color, alpha=0.85)

            if x_break is None:
                fig, ax = plt.subplots(figsize=(12, 6))
                x_pad = max(1.0, x_span * 0.08)
                ax.set_xlim(x_min - x_pad, x_max + x_pad)

                _scatter_points(ax, rows)
                ax.scatter([baseline_x], [baseline_y], s=80, marker="*", color="black")

                ax.set_title(
                    f"By-Parameter % Above {baseline_name} d{baseline_d} {title_suffix} Threshold ({threshold_score:.4f})"
                )
                ax.set_xlabel("Parameter Count")
                ax.set_ylabel("% Scores Above Baseline Threshold")
                ax.grid(axis="y", alpha=0.25)
                ax.set_ylim(0, 100)
                _add_scatter_legend(ax, rows)
                fig.tight_layout()
                out_path = os.path.join(output_dir, out_name)
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                return out_path

            break_start, break_end = x_break
            if break_start >= break_end:
                return None

            left_rows = [r for r in rows if float(r["combo_param_count"]) <= break_start]
            right_rows = [r for r in rows if float(r["combo_param_count"]) >= break_end]

            # Include baseline reference point in appropriate side.
            baseline_on_left = baseline_x <= break_start
            baseline_on_right = baseline_x >= break_end

            if (not left_rows and not baseline_on_left) or (not right_rows and not baseline_on_right):
                return None

            left_x_vals = [float(r["combo_param_count"]) for r in left_rows]
            right_x_vals = [float(r["combo_param_count"]) for r in right_rows]
            if baseline_on_left:
                left_x_vals.append(baseline_x)
            if baseline_on_right:
                right_x_vals.append(baseline_x)

            shared_pad = break_start - max(left_x_vals)
            if shared_pad <= 0:
                shared_pad = max(1.0, x_span * 0.01)

            left_xlim_min = min(left_x_vals) - shared_pad
            left_xlim_max = break_start
            right_xlim_min = break_end
            right_xlim_max = max(right_x_vals) + shared_pad

            left_span = max(1.0, left_xlim_max - left_xlim_min)
            right_span = max(1.0, right_xlim_max - right_xlim_min)

            fig, (ax_left, ax_right) = plt.subplots(
                1,
                2,
                sharey=True,
                figsize=(14, 6),
                gridspec_kw={"width_ratios": [left_span, right_span]},
            )

            ax_left.set_xlim(left_xlim_min, left_xlim_max)
            ax_right.set_xlim(right_xlim_min, right_xlim_max)

            _scatter_points(ax_left, left_rows)
            _scatter_points(ax_right, right_rows)

            if baseline_on_left:
                ax_left.scatter([baseline_x], [baseline_y], s=80, marker="*", color="black")
            if baseline_on_right:
                ax_right.scatter([baseline_x], [baseline_y], s=80, marker="*", color="black")

            ax_left.set_title(
                f"By-Parameter % Above {baseline_name} d{baseline_d} {title_suffix} Threshold ({threshold_score:.4f})"
            )
            ax_left.set_xlabel("Parameter Count")
            ax_right.set_xlabel("Parameter Count")
            ax_left.set_ylabel("% Scores Above Baseline Threshold")
            ax_left.grid(axis="y", alpha=0.25)
            ax_right.grid(axis="y", alpha=0.25)
            ax_left.set_ylim(0, 100)
            _add_scatter_legend(ax_right, rows)

            ax_left.spines["right"].set_visible(False)
            ax_right.spines["left"].set_visible(False)
            ax_right.yaxis.tick_right()
            ax_right.tick_params(labelright=False)

            marker_kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=8, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
            ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **marker_kwargs)
            ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **marker_kwargs)

            fig.tight_layout()
            out_path = os.path.join(output_dir, out_name)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path

        for label, title_suffix, out_name in [
            ("top_1pct", "Top 1%", f"stats_chart_pct_beating_baseline_top1{output_suffix}.png"),
            ("top_5pct", "Top 5%", f"stats_chart_pct_beating_baseline_top5{output_suffix}.png"),
        ]:
            subset = df4[df4["percentile_label"] == label]
            if subset.empty:
                continue

            subset = subset.copy()
            subset["combo_label"] = subset.apply(
                lambda r: f"{r['model_name']} / d{int(r['dendrite_count'])}",
                axis=1,
            )
            subset = subset.sort_values(["model_id", "dendrite_count"])

            labels = subset["combo_label"].tolist()
            vals = subset["combo_pct_beating_threshold"].astype(float).tolist()

            fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5))
            bars = ax.bar(labels, vals)
            ax.set_ylim(0, 100)
            baseline_name = str(subset["baseline_model_name"].iloc[0])
            baseline_d = int(subset["baseline_dendrite_count"].iloc[0])
            threshold_score = float(subset["baseline_threshold_score"].iloc[0])
            ax.set_title(
                f"% Beating {baseline_name} d{baseline_d} {title_suffix} Threshold ({threshold_score:.4f}) [{metric_label}]"
            )
            ax.set_xlabel("Model / Dendrite Combo")
            ax.set_ylabel("% Scores Above Baseline Threshold")
            ax.grid(axis="y", alpha=0.25)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

            fig.tight_layout()
            out_path = os.path.join(output_dir, out_name)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            created.append(out_path)

        scatter_top1 = _plot_metric4_param_scatter(
            df4[df4["percentile_label"] == "top_1pct"],
            "Top 1%",
            f"stats_scatter_pct_beating_baseline_top1_by_param{output_suffix}.png",
            baseline_y=1.0,
        )
        if scatter_top1:
            created.append(scatter_top1)

        scatter_top5 = _plot_metric4_param_scatter(
            df4[df4["percentile_label"] == "top_5pct"],
            "Top 5%",
            f"stats_scatter_pct_beating_baseline_top5_by_param{output_suffix}.png",
            baseline_y=5.0,
        )
        if scatter_top5:
            created.append(scatter_top5)

    return created


def _create_categorical_plot(
    stats: Sequence[Dict[str, float]],
    output_path: str,
    column_color_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    metric_label: str = "Val",
) -> None:
    """Create candlestick-style plot with categorical x-axis labels."""
    fig, ax = plt.subplots(figsize=(max(10, len(stats) * 0.45), 6))

    bxp_stats = [
        {
            "label": item["label"],
            "whislo": item["whislo"],
            "q1": item["q1"],
            "med": item["med"],
            "q3": item["q3"],
            "whishi": item["whishi"],
        }
        for item in stats
    ]

    artists = ax.bxp(bxp_stats, showfliers=False, patch_artist=True)

    if column_color_map is None:
        column_color_map = {}

    for i, item in enumerate(stats):
        color = column_color_map.get(item["column"], (0.2, 0.4, 0.8))
        artists["boxes"][i].set_facecolor((*color, 0.35))
        artists["boxes"][i].set_edgecolor(color)
        artists["boxes"][i].set_linewidth(1.2)

        artists["medians"][i].set_color(color)
        artists["medians"][i].set_linewidth(1.8)

        artists["whiskers"][2 * i].set_color(color)
        artists["whiskers"][2 * i + 1].set_color(color)
        artists["caps"][2 * i].set_color(color)
        artists["caps"][2 * i + 1].set_color(color)
    ax.set_title(f"Dendrite {metric_label} Distribution by Model/Dendrite Pair")
    ax.set_xlabel("Model / Dendrite Pair")
    ax.set_ylabel(f"Max {metric_label}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_param_count_plot(
    stats: Sequence[Dict[str, float]],
    param_counts_by_column: Dict[str, float],
    output_path: str,
    x_break: Optional[Tuple[float, float]] = None,
    connect_extremes: bool = False,
    column_color_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    metric_label: str = "Val",
) -> None:
    """Create candlestick-style plot positioned by param_count on x-axis."""
    if column_color_map is None:
        column_color_map = {}

    def _style_bxp_artists(artists: Dict[str, List], items: Sequence[Dict[str, float]]) -> None:
        for i, item in enumerate(items):
            color = column_color_map.get(item["column"], (0.2, 0.4, 0.8))
            artists["boxes"][i].set_facecolor((*color, 0.35))
            artists["boxes"][i].set_edgecolor(color)
            artists["boxes"][i].set_linewidth(1.2)

            artists["medians"][i].set_color(color)
            artists["medians"][i].set_linewidth(1.8)

            artists["whiskers"][2 * i].set_color(color)
            artists["whiskers"][2 * i + 1].set_color(color)
            artists["caps"][2 * i].set_color(color)
            artists["caps"][2 * i + 1].set_color(color)

    def _add_inside_legend(ax, items: Sequence[Dict[str, float]]) -> None:
        handles = []
        labels = []
        for item in items:
            color = column_color_map.get(item["column"], (0.2, 0.4, 0.8))
            handles.append(Line2D([0], [0], marker="s", linestyle="none", markersize=6, markerfacecolor=color, markeredgecolor=color))
            labels.append(item["label"])
        if handles:
            ax.legend(handles, labels, loc="lower right", fontsize=10.5, framealpha=0.85)

    stats_with_counts = [
        item for item in stats
        if item["column"] in param_counts_by_column
    ]

    if not stats_with_counts:
        raise ValueError("No parameter-count metadata found for dendrite columns.")

    base_positions = [param_counts_by_column[item["column"]] for item in stats_with_counts]

    # If multiple columns share the same param_count, add tiny deterministic offsets.
    grouped_indices: Dict[float, List[int]] = defaultdict(list)
    for i, pos in enumerate(base_positions):
        grouped_indices[pos].append(i)

    if base_positions:
        x_min = min(base_positions)
        x_max = max(base_positions)
        x_span = x_max - x_min
    else:
        x_span = 0.0

    offset_step = max(1.0, x_span * 0.002)
    adjusted_positions = base_positions[:]

    for _, idxs in grouped_indices.items():
        if len(idxs) <= 1:
            continue
        center = (len(idxs) - 1) / 2.0
        for k, idx in enumerate(idxs):
            adjusted_positions[idx] = base_positions[idx] + (k - center) * offset_step

    def _positive_gaps(values: Sequence[float]) -> List[float]:
        unique_vals = sorted(set(values))
        gaps: List[float] = []
        for i in range(1, len(unique_vals)):
            gap = unique_vals[i] - unique_vals[i - 1]
            if gap <= 0:
                continue
            gaps.append(gap)
        return gaps

    def _reference_gap(values: Sequence[float], quantile: float = 0.25) -> Optional[float]:
        gaps = _positive_gaps(values)
        if not gaps:
            return None
        return float(pd.Series(gaps).quantile(quantile))

    def _compute_width_px(default_axis_width_px: float, min_gap_px: Optional[float]) -> float:
        default_width_px = max(1.0, default_axis_width_px * 0.015)
        if min_gap_px is None:
            return default_width_px
        # overlap_fraction = (width_px - gap_px) / width_px; require overlap_fraction < 0.5
        # => width_px < gap_px / 0.5. Use 0.51 for strict "left side at 51 when width=100" behavior.
        cap_width_px = min_gap_px / 0.51
        return max(1.0, min(default_width_px, cap_width_px))

    def _enforce_min_center_gap(positions: Sequence[float], min_gap_data: float) -> List[float]:
        adjusted = list(positions)
        sorted_indices = sorted(range(len(adjusted)), key=lambda i: adjusted[i])
        prev = None
        for idx in sorted_indices:
            current = adjusted[idx]
            if prev is None:
                prev = current
                continue
            if current - prev < min_gap_data:
                current = prev + min_gap_data
                adjusted[idx] = current
            prev = current
        return adjusted

    def _to_bxp_stats(items: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
        return [
            {
                "label": item["label"],
                "whislo": item["whislo"],
                "q1": item["q1"],
                "med": item["med"],
                "q3": item["q3"],
                "whishi": item["whishi"],
            }
            for item in items
        ]

    def _draw_extreme_connectors_single(
        ax,
        items: Sequence[Dict[str, float]],
        positions: Sequence[float],
    ) -> None:
        if len(items) < 2:
            return
        left_idx = min(range(len(positions)), key=lambda i: positions[i])
        right_idx = max(range(len(positions)), key=lambda i: positions[i])
        x_left = positions[left_idx]
        x_right = positions[right_idx]
        y_keys = ["whislo", "med", "whishi"]
        for y_key in y_keys:
            y_left = items[left_idx][y_key]
            y_right = items[right_idx][y_key]
            ax.plot([x_left, x_right], [y_left, y_right], color="tab:orange", linewidth=1.5, alpha=0.9)

    def _interpolate_y(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
        if x2 == x1:
            return y1
        return y1 + (y2 - y1) * ((x - x1) / (x2 - x1))

    def _draw_extreme_connectors_broken(
        ax_left,
        ax_right,
        left_items: Sequence[Dict[str, float]],
        left_positions: Sequence[float],
        right_items: Sequence[Dict[str, float]],
        right_positions: Sequence[float],
        break_start_val: float,
        break_end_val: float,
    ) -> None:
        if not left_items or not right_items:
            return

        left_idx = min(range(len(left_positions)), key=lambda i: left_positions[i])
        right_idx = max(range(len(right_positions)), key=lambda i: right_positions[i])
        x_left = left_positions[left_idx]
        x_right = right_positions[right_idx]

        y_keys = ["whislo", "med", "whishi"]
        for y_key in y_keys:
            y_left = left_items[left_idx][y_key]
            y_right = right_items[right_idx][y_key]

            y_at_break_start = _interpolate_y(x_left, y_left, x_right, y_right, break_start_val)
            y_at_break_end = _interpolate_y(x_left, y_left, x_right, y_right, break_end_val)

            ax_left.plot([x_left, break_start_val], [y_left, y_at_break_start], color="tab:orange", linewidth=1.5, alpha=0.9)
            ax_right.plot([break_end_val, x_right], [y_at_break_end, y_right], color="tab:orange", linewidth=1.5, alpha=0.9)

    if x_break is None:
        fig, ax = plt.subplots(figsize=(12, 6))

        x_pad = max(1.0, x_span * 0.08)
        xlim_min = min(adjusted_positions) - x_pad
        xlim_max = max(adjusted_positions) + x_pad
        ax.set_xlim(xlim_min, xlim_max)

        fig.canvas.draw()
        axis_width_px = ax.get_window_extent().width
        axis_span = xlim_max - xlim_min
        px_per_data = axis_width_px / axis_span if axis_span > 0 else 0.0

        ref_gap_data = _reference_gap(base_positions, quantile=0.25)
        ref_gap_px = (ref_gap_data * px_per_data) if (ref_gap_data is not None and px_per_data > 0) else None
        width_px = _compute_width_px(axis_width_px, ref_gap_px)
        width = width_px / px_per_data if px_per_data > 0 else max(1.0, x_span * 0.015)

        if px_per_data > 0:
            required_gap_data = (width_px * 0.51) / px_per_data
            plot_positions = _enforce_min_center_gap(adjusted_positions, required_gap_data)
        else:
            plot_positions = adjusted_positions

        artists = ax.bxp(
            _to_bxp_stats(stats_with_counts),
            positions=plot_positions,
            widths=width,
            showfliers=False,
            manage_ticks=False,
            patch_artist=True,
        )
        _style_bxp_artists(artists, stats_with_counts)

        ax.set_title(f"Dendrite {metric_label} Distribution by Parameter Count")
        ax.set_xlabel("Parameter Count")
        ax.set_ylabel(f"Max {metric_label}")
        ax.grid(axis="y", alpha=0.25)

        _add_inside_legend(ax, stats_with_counts)

        if connect_extremes:
            _draw_extreme_connectors_single(ax, stats_with_counts, plot_positions)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    break_start, break_end = x_break
    if break_start >= break_end:
        raise ValueError("x-break must have start < end.")

    left_items: List[Dict[str, float]] = []
    left_positions: List[float] = []
    right_items: List[Dict[str, float]] = []
    right_positions: List[float] = []

    for item, base_pos, adjusted_pos in zip(stats_with_counts, base_positions, adjusted_positions):
        if base_pos <= break_start:
            left_items.append(item)
            left_positions.append(adjusted_pos)
        elif base_pos >= break_end:
            right_items.append(item)
            right_positions.append(adjusted_pos)

    if not left_items or not right_items:
        data_min = min(base_positions)
        data_max = max(base_positions)
        distinct_positions = sorted(set(base_positions))
        formatted_positions = ", ".join(f"{int(v):,}" if float(v).is_integer() else f"{v:,.3f}" for v in distinct_positions)
        raise ValueError(
            "x-break range removes one side of the chart. "
            f"Data param_count range is [{data_min:,.0f}, {data_max:,.0f}] with values: {formatted_positions}. "
            "Choose START/END so there are points <= START and >= END."
        )

    left_base_positions = [
        param_counts_by_column[item["column"]]
        for item in left_items
    ]
    right_base_positions = [
        param_counts_by_column[item["column"]]
        for item in right_items
    ]

    # Shared tight padding from nearest left value to break start.
    shared_pad = break_start - max(left_base_positions)
    if shared_pad <= 0:
        shared_pad = max(1.0, x_span * 0.01)

    left_xlim_min = min(left_base_positions) - shared_pad
    left_xlim_max = break_start
    right_xlim_min = break_end
    right_xlim_max = max(right_base_positions) + shared_pad

    left_span = left_xlim_max - left_xlim_min
    right_span = right_xlim_max - right_xlim_min
    left_span = left_span if left_span > 0 else 1.0
    right_span = right_span if right_span > 0 else 1.0

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [left_span, right_span]},
    )

    ax_left.set_xlim(left_xlim_min, left_xlim_max)
    ax_right.set_xlim(right_xlim_min, right_xlim_max)

    fig.canvas.draw()
    left_axis_width_px = ax_left.get_window_extent().width
    right_axis_width_px = ax_right.get_window_extent().width
    left_axis_span = left_xlim_max - left_xlim_min
    right_axis_span = right_xlim_max - right_xlim_min

    left_px_per_data = left_axis_width_px / left_axis_span if left_axis_span > 0 else 0.0
    right_px_per_data = right_axis_width_px / right_axis_span if right_axis_span > 0 else 0.0

    left_ref_gap_data = _reference_gap(left_base_positions, quantile=0.25)
    right_ref_gap_data = _reference_gap(right_base_positions, quantile=0.25)

    gap_candidates_px: List[float] = []
    if left_ref_gap_data is not None and left_px_per_data > 0:
        gap_candidates_px.append(left_ref_gap_data * left_px_per_data)
    if right_ref_gap_data is not None and right_px_per_data > 0:
        gap_candidates_px.append(right_ref_gap_data * right_px_per_data)
    ref_gap_px = min(gap_candidates_px) if gap_candidates_px else None

    total_axis_width_px = left_axis_width_px + right_axis_width_px
    global_width_px = _compute_width_px(total_axis_width_px, ref_gap_px)
    left_width = global_width_px / left_px_per_data if left_px_per_data > 0 else 1.0
    right_width = global_width_px / right_px_per_data if right_px_per_data > 0 else 1.0

    if left_px_per_data > 0:
        left_required_gap_data = (global_width_px * 0.51) / left_px_per_data
        left_plot_positions = _enforce_min_center_gap(left_positions, left_required_gap_data)
    else:
        left_plot_positions = left_positions

    if right_px_per_data > 0:
        right_required_gap_data = (global_width_px * 0.51) / right_px_per_data
        right_plot_positions = _enforce_min_center_gap(right_positions, right_required_gap_data)
    else:
        right_plot_positions = right_positions

    left_artists = ax_left.bxp(
        _to_bxp_stats(left_items),
        positions=left_plot_positions,
        widths=left_width,
        showfliers=False,
        manage_ticks=False,
        patch_artist=True,
    )
    right_artists = ax_right.bxp(
        _to_bxp_stats(right_items),
        positions=right_plot_positions,
        widths=right_width,
        showfliers=False,
        manage_ticks=False,
        patch_artist=True,
    )
    _style_bxp_artists(left_artists, left_items)
    _style_bxp_artists(right_artists, right_items)

    ax_left.set_title(f"Dendrite {metric_label} Distribution by Parameter Count (Broken X-Axis)")
    ax_left.set_xlabel("Parameter Count")
    ax_right.set_xlabel("Parameter Count")
    ax_left.set_ylabel(f"Max {metric_label}")
    ax_left.grid(axis="y", alpha=0.25)
    ax_right.grid(axis="y", alpha=0.25)

    _add_inside_legend(ax_right, stats_with_counts)

    if connect_extremes:
        _draw_extreme_connectors_broken(
            ax_left,
            ax_right,
            left_items,
            left_plot_positions,
            right_items,
            right_plot_positions,
            break_start,
            break_end,
        )

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(labelright=False)

    marker_kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=8, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **marker_kwargs)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **marker_kwargs)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_param_count_scatter_plot(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    param_counts_by_column: Dict[str, float],
    output_path: str,
    x_break: Optional[Tuple[float, float]] = None,
    model_name_map: Optional[Dict[str, str]] = None,
    column_color_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    metric_label: str = "Val",
) -> None:
    """Create per-column colored scatter plot of all values against param_count."""
    column_points: List[Dict[str, object]] = []
    for col in dendrite_columns:
        if col not in param_counts_by_column:
            continue
        y_vals = df[col].dropna().tolist()
        if not y_vals:
            continue
        column_points.append(
            {
                "column": col,
                "label": _display_label(col, model_name_map),
                "x": float(param_counts_by_column[col]),
                "y": y_vals,
            }
        )

    if not column_points:
        raise ValueError("No non-empty dendrite metric data found to scatter.")

    if column_color_map is None:
        column_color_map = {}

    x_values = [item["x"] for item in column_points]
    x_min = min(x_values)
    x_max = max(x_values)
    x_span = x_max - x_min

    if x_break is None:
        fig, ax = plt.subplots(figsize=(12, 6))

        x_pad = max(1.0, x_span * 0.08)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

        for item in column_points:
            color = column_color_map.get(item["column"], (0.2, 0.4, 0.8))
            x = item["x"]
            y_vals = item["y"]
            ax.scatter([x] * len(y_vals), y_vals, s=12, color=color, alpha=0.75, label=item["label"])

        ax.set_title(f"Dendrite {metric_label} Scatter by Parameter Count")
        ax.set_xlabel("Parameter Count")
        ax.set_ylabel(f"Max {metric_label}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=7)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    break_start, break_end = x_break
    if break_start >= break_end:
        raise ValueError("x-break must have start < end.")

    left_points = [item for item in column_points if item["x"] <= break_start]
    right_points = [item for item in column_points if item["x"] >= break_end]

    if not left_points or not right_points:
        distinct_positions = sorted(set(x_values))
        formatted_positions = ", ".join(f"{int(v):,}" if float(v).is_integer() else f"{v:,.3f}" for v in distinct_positions)
        raise ValueError(
            "x-break range removes one side of the scatter chart. "
            f"Data param_count values: {formatted_positions}. "
            "Choose START/END so there are points <= START and >= END."
        )

    left_x_vals = [item["x"] for item in left_points]
    right_x_vals = [item["x"] for item in right_points]

    shared_pad = break_start - max(left_x_vals)
    if shared_pad <= 0:
        shared_pad = max(1.0, x_span * 0.01)

    left_xlim_min = min(left_x_vals) - shared_pad
    left_xlim_max = break_start
    right_xlim_min = break_end
    right_xlim_max = max(right_x_vals) + shared_pad

    left_span = left_xlim_max - left_xlim_min
    right_span = right_xlim_max - right_xlim_min
    left_span = left_span if left_span > 0 else 1.0
    right_span = right_span if right_span > 0 else 1.0

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [left_span, right_span]},
    )

    ax_left.set_xlim(left_xlim_min, left_xlim_max)
    ax_right.set_xlim(right_xlim_min, right_xlim_max)

    for item in column_points:
        color = column_color_map.get(item["column"], (0.2, 0.4, 0.8))
        x = item["x"]
        y_vals = item["y"]
        if x <= break_start:
            ax_left.scatter([x] * len(y_vals), y_vals, s=12, color=color, alpha=0.75, label=item["label"])
        elif x >= break_end:
            ax_right.scatter([x] * len(y_vals), y_vals, s=12, color=color, alpha=0.75, label=item["label"])

    ax_left.set_title(f"Dendrite {metric_label} Scatter by Parameter Count (Broken X-Axis)")
    ax_left.set_xlabel("Parameter Count")
    ax_right.set_xlabel("Parameter Count")
    ax_left.set_ylabel(f"Max {metric_label}")
    ax_left.grid(axis="y", alpha=0.25)
    ax_right.grid(axis="y", alpha=0.25)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(labelright=False)

    marker_kwargs = dict(marker=[(-1, -1), (1, 1)], markersize=8, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **marker_kwargs)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **marker_kwargs)

    handles, labels = ax_left.get_legend_handles_labels()
    if handles:
        ax_left.legend(handles, labels, loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_data_percent_line_plot(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    param_counts_by_column: Dict[str, float],
    output_path: str,
    model_name_map: Optional[Dict[str, str]] = None,
    metric_label: str = "Val",
    x_break: Optional[Tuple[float, float]] = None,
    split_filter: Optional[str] = None,
    require_complete_repeat_dendrites: bool = False,
    expected_repeat_count: int = 5,
    dendrite_percent_to_graph: Optional[float] = None,
    sample_split_keys: Optional[Sequence[str]] = None,
    subject_split_keys: Optional[Sequence[str]] = None,
    subject_filter_key: Optional[str] = None,
    draw_boxplots: bool = False,
    average_repeats: bool = True,
) -> None:
    """Create a single line graph with one connected line per run.

    X-axis: parameter count.
    Y-axis: score.
    Color: data percent.
    Marker shape: unique per run.

    Each run contributes one point per dendrite count (ordered by dendrite index),
    connected by a line.

    If repeated runs exist for the same split/model/data-percent group, they are
    averaged before plotting (sweep_number in metadata/name is one way to signal
    this). Dendrite points are averaged independently, so missing higher
    dendrites in some runs do not block averaging for the available points.

    When require_complete_repeat_dendrites is enabled, a dendrite point is kept
    only if it appears in all repeats for that group and the repeat count meets
    expected_repeat_count.

    When dendrite_percent_to_graph is set (for example 0.6), a dendrite point
    is kept if it appears in at least that fraction of runs within the group.
    """
    model_dendrite_cols: Dict[str, List[Tuple[int, str]]] = {}
    for col in dendrite_columns:
        model_id, dendrite_idx = _parse_model_and_dendrite(col)
        if model_id is None or dendrite_idx is None:
            continue
        model_dendrite_cols.setdefault(model_id, []).append((dendrite_idx, col))
    for model_id in model_dendrite_cols:
        model_dendrite_cols[model_id].sort(key=lambda x: x[0])

    run_data: Dict[str, Dict[str, Any]] = {}
    has_sweep_number = False
    has_config_sweep_number = "config_sweep_number" in df.columns
    for _, row in df.iterrows():
        run_id = str(row.get("run_id", "")).strip()
        run_name = str(row.get("run_name", "")).strip()
        split_label = _extract_split_label(run_name)

        if split_filter is not None and split_label != split_filter:
            continue

        data_percent, data_percent_label = _extract_data_percent_from_run_name(run_name)
        if data_percent is None:
            continue

        subject_key, subject_label, sample_key, sample_label = _extract_subject_sample_from_run_name(run_name)
        if subject_key is None or sample_key is None:
            continue

        if subject_filter_key is not None and str(subject_key) != str(subject_filter_key):
            continue

        if sample_split_keys is not None and str(sample_key) not in sample_split_keys:
            continue
        if subject_split_keys is not None and str(subject_key) not in subject_split_keys:
            continue

        if re.search(r"sweep_number_([^_]+)", run_name):
            has_sweep_number = True
        elif has_config_sweep_number:
            config_sweep_value = row.get("config_sweep_number", None)
            if pd.notna(config_sweep_value):
                has_sweep_number = True

        model_match = re.search(r"model_index_(\d+)", run_name)
        model_id = f"model_{model_match.group(1)}" if model_match else None

        # Fallback for run names without model_index token.
        if model_id is None:
            populated_models: List[str] = []
            for col in dendrite_columns:
                model_from_col, _ = _parse_model_and_dendrite(col)
                if model_from_col is None:
                    continue
                value = pd.to_numeric(row.get(col, None), errors="coerce")
                if not pd.isna(value):
                    populated_models.append(model_from_col)
            unique_models = sorted(set(populated_models), key=_model_sort_key)
            if len(unique_models) == 1:
                model_id = unique_models[0]

        if model_id is None:
            continue

        if run_id not in run_data:
            run_data[run_id] = {
                "split_label": split_label,
                "model_id": model_id,
                "data_percent": str(data_percent),
                "data_percent_label": data_percent_label,
                "subject_key": subject_key,
                "subject_label": subject_label,
                "sample_key": sample_key,
                "sample_label": sample_label,
                "scores": {},
            }

        for dendrite_idx, col in model_dendrite_cols.get(model_id, []):
            if col not in param_counts_by_column:
                continue
            value = pd.to_numeric(row.get(col, None), errors="coerce")
            if not pd.isna(value):
                param_count = float(param_counts_by_column[col])
                run_data[run_id]["scores"][dendrite_idx] = (param_count, float(value))

    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for run_id, data in run_data.items():
        group_key = (str(data["split_label"]), str(data["model_id"]), str(data["data_percent"]))
        if group_key not in grouped:
            grouped[group_key] = {
                "split_label": data["split_label"],
                "model_id": data["model_id"],
                "data_percent": data["data_percent"],
                "data_percent_label": data["data_percent_label"],
                "subject_key": data["subject_key"],
                "subject_label": data["subject_label"],
                "sample_key": data["sample_key"],
                "sample_label": data["sample_label"],
                "scores_accum": defaultdict(list),
                "run_ids": [],
            }
        grouped[group_key]["run_ids"].append(run_id)
        for dendrite_idx, pair in data["scores"].items():
            grouped[group_key]["scores_accum"][dendrite_idx].append(pair)

    has_repeated_groups = any(len(grouped_entry["run_ids"]) > 1 for grouped_entry in grouped.values())

    if average_repeats and (has_sweep_number or has_repeated_groups) and run_data:

        averaged_run_data: Dict[str, Dict[str, Any]] = {}
        for i, group_key in enumerate(sorted(grouped.keys())):
            grouped_entry = grouped[group_key]
            averaged_scores: Dict[int, Tuple[float, float]] = {}
            group_run_count = len(grouped_entry["run_ids"])
            for dendrite_idx, pair_list in grouped_entry["scores_accum"].items():
                if not pair_list:
                    continue

                if require_complete_repeat_dendrites:
                    has_full_group_coverage = len(pair_list) == group_run_count
                    meets_expected_repeat_count = group_run_count >= expected_repeat_count
                    if not (has_full_group_coverage and meets_expected_repeat_count):
                        continue
                elif dendrite_percent_to_graph is not None:
                    if group_run_count <= 0:
                        continue
                    present_fraction = len(pair_list) / group_run_count
                    if present_fraction < dendrite_percent_to_graph:
                        continue

                avg_param_count = float(sum(p[0] for p in pair_list) / len(pair_list))
                avg_score = float(sum(p[1] for p in pair_list) / len(pair_list))
                averaged_scores[dendrite_idx] = (avg_param_count, avg_score)

            if not averaged_scores:
                continue

            averaged_run_data[f"avg_{i}"] = {
                "split_label": grouped_entry["split_label"],
                "model_id": grouped_entry["model_id"],
                "data_percent": grouped_entry["data_percent"],
                "data_percent_label": grouped_entry["data_percent_label"],
                "subject_key": grouped_entry["subject_key"],
                "subject_label": grouped_entry["subject_label"],
                "sample_key": grouped_entry["sample_key"],
                "sample_label": grouped_entry["sample_label"],
                "scores": averaged_scores,
                "avg_from_runs": len(grouped_entry["run_ids"]),
            }

        run_data = averaged_run_data

    # In non-averaged mode (used by boxplots), still enforce dendrite coverage
    # threshold at the repeat-group level so sparse dendrites are excluded.
    if (not average_repeats) and (dendrite_percent_to_graph is not None) and run_data:
        allowed_dendrites_by_group: Dict[Tuple[str, str, str], set] = {}
        for group_key, grouped_entry in grouped.items():
            group_run_count = len(grouped_entry["run_ids"])
            if group_run_count <= 0:
                continue

            allowed = {
                dendrite_idx
                for dendrite_idx, pair_list in grouped_entry["scores_accum"].items()
                if (len(pair_list) / group_run_count) >= dendrite_percent_to_graph
            }
            allowed_dendrites_by_group[group_key] = allowed

        filtered_run_data: Dict[str, Dict[str, Any]] = {}
        for run_id, data in run_data.items():
            group_key = (str(data["split_label"]), str(data["model_id"]), str(data["data_percent"]))
            allowed = allowed_dendrites_by_group.get(group_key, set())
            filtered_scores = {
                dendrite_idx: pair
                for dendrite_idx, pair in data["scores"].items()
                if dendrite_idx in allowed
            }
            if not filtered_scores:
                continue

            new_data = dict(data)
            new_data["scores"] = filtered_scores
            filtered_run_data[run_id] = new_data

        run_data = filtered_run_data

    if not run_data:
        split_text = f" for split '{split_filter}'" if split_filter is not None else ""
        raise ValueError(f"No run data could be extracted for the data_percent line plot{split_text}.")

    observed_sample_keys = {str(d["sample_key"]) for d in run_data.values()}
    if sample_split_keys is not None:
        all_sample_keys = sorted(
            [key for key in sample_split_keys if key in observed_sample_keys],
            key=_percent_key_sort_key,
            reverse=True,
        )
    else:
        all_sample_keys = sorted(observed_sample_keys, key=_percent_key_sort_key, reverse=True)

    if not all_sample_keys:
        split_text = f" for split '{split_filter}'" if split_filter is not None else ""
        raise ValueError(f"No matching sample splits found in run data{split_text}.")

    sample_cmap = plt.get_cmap("viridis")
    sample_colors = {
        sample_key: sample_cmap(i / max(1, len(all_sample_keys) - 1))
        for i, sample_key in enumerate(all_sample_keys)
    }

    sample_labels = {
        str(d["sample_key"]): str(d["sample_label"])
        for d in run_data.values()
    }

    observed_subject_keys = {str(d["subject_key"]) for d in run_data.values()}
    if subject_split_keys is not None:
        all_subject_keys = sorted(
            [key for key in subject_split_keys if key in observed_subject_keys],
            key=_percent_key_sort_key,
            reverse=True,
        )
    else:
        all_subject_keys = sorted(observed_subject_keys, key=_percent_key_sort_key, reverse=True)

    if not all_subject_keys:
        split_text = f" for split '{split_filter}'" if split_filter is not None else ""
        raise ValueError(f"No matching subject splits found in run data{split_text}.")

    base_markers = [
        "o", "s", "^", "D", "v", "P", "*", "X", "<", ">",
        "h", "H", "d", "p", "8", "1", "2", "3", "4", "+", "x",
    ]
    subject_markers: Dict[str, str] = {}
    for i, subject_key in enumerate(all_subject_keys):
        if i < len(base_markers):
            subject_markers[subject_key] = base_markers[i]
        else:
            subject_markers[subject_key] = f"${i + 1}$"

    subject_labels = {
        str(d["subject_key"]): str(d["subject_label"])
        for d in run_data.values()
    }

    def _build_legend(ax) -> None:
        handles: List[Line2D] = []
        labels: List[str] = []

        for sample_key in all_sample_keys:
            handles.append(Line2D([0], [0], color=sample_colors[sample_key], linewidth=2))
            labels.append(sample_labels.get(sample_key, sample_key))

        if all_sample_keys and all_subject_keys:
            handles.append(Line2D([0], [0], linestyle="none", color="none"))
            labels.append("")

        for subject_key in all_subject_keys:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=subject_markers[subject_key],
                    linestyle="none",
                    markersize=7,
                    color="black",
                    markerfacecolor="black",
                )
            )
            labels.append(subject_labels.get(subject_key, subject_key))

        ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.85)

    def _plot_runs(ax, x_min_filter=None, x_max_filter=None) -> None:
        for run_id in sorted(run_data.keys()):
            data = run_data[run_id]
            scores = data["scores"]
            if not scores:
                continue

            points = [scores[dendrite_idx] for dendrite_idx in sorted(scores.keys())]

            if x_min_filter is not None:
                points = [(x, y) for x, y in points if x >= x_min_filter]
            if x_max_filter is not None:
                points = [(x, y) for x, y in points if x <= x_max_filter]
            if not points:
                continue

            x_vals, y_vals = zip(*points)
            ax.plot(
                x_vals,
                y_vals,
                marker=subject_markers[str(data["subject_key"])],
                color=sample_colors[str(data["sample_key"])],
                linewidth=1.5,
                markersize=5,
                alpha=0.8,
            )

    def _plot_boxplots(ax, x_min_filter=None, x_max_filter=None) -> None:
        grouped_values: Dict[Tuple[float, str, str], List[float]] = defaultdict(list)

        for run_id in sorted(run_data.keys()):
            data = run_data[run_id]
            sample_key = str(data["sample_key"])
            subject_key = str(data["subject_key"])
            scores = data["scores"]
            if not scores:
                continue

            for dendrite_idx in sorted(scores.keys()):
                x, y = scores[dendrite_idx]
                if x_min_filter is not None and x < x_min_filter:
                    continue
                if x_max_filter is not None and x > x_max_filter:
                    continue
                grouped_values[(float(x), sample_key, subject_key)].append(float(y))

        if not grouped_values:
            return

        sorted_group_keys = sorted(
            grouped_values.keys(),
            key=lambda key: (
                key[0],
                _percent_key_sort_key(key[1]),
                _percent_key_sort_key(key[2]),
            ),
        )

        x_values_local = sorted({key[0] for key in sorted_group_keys})
        x_span_local = x_values_local[-1] - x_values_local[0] if len(x_values_local) > 1 else 0.0
        offset_step = max(1.0, x_span_local * 0.003)
        box_width = max(1.0, x_span_local * 0.006)

        x_to_group_keys: Dict[float, List[Tuple[float, str, str]]] = defaultdict(list)
        for group_key in sorted_group_keys:
            x_to_group_keys[group_key[0]].append(group_key)

        for x_value in x_values_local:
            same_x_groups = x_to_group_keys[x_value]
            center = (len(same_x_groups) - 1) / 2.0
            for i, group_key in enumerate(same_x_groups):
                _, sample_key, _subject_key = group_key
                y_values = grouped_values[group_key]
                if len(y_values) < 2:
                    continue
                position = x_value + (i - center) * offset_step

                artists = ax.boxplot(
                    [y_values],
                    positions=[position],
                    widths=box_width,
                    showfliers=False,
                    patch_artist=True,
                    manage_ticks=False,
                )

                color = sample_colors.get(sample_key, (0.2, 0.4, 0.8, 1.0))
                artists["boxes"][0].set_facecolor((*color[:3], 0.35) if len(color) == 4 else (*color, 0.35))
                artists["boxes"][0].set_edgecolor(color)
                artists["boxes"][0].set_linewidth(1.2)
                artists["medians"][0].set_color(color)
                artists["medians"][0].set_linewidth(1.8)
                for whisker in artists["whiskers"]:
                    whisker.set_color(color)
                for cap in artists["caps"]:
                    cap.set_color(color)

    all_x = [scores_pair[0] for data in run_data.values() for scores_pair in data["scores"].values()]
    x_min = min(all_x)
    x_max = max(all_x)
    x_span = x_max - x_min

    if x_break is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pad = max(1.0, x_span * 0.05)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        if draw_boxplots:
            _plot_boxplots(ax)
        else:
            _plot_runs(ax)
        _build_legend(ax)
        title_parts = [f"Score by Parameter Count ({metric_label})"]
        if split_filter is not None:
            title_parts.append(str(split_filter))
        if subject_filter_key is not None:
            title_parts.append(str(subject_labels.get(subject_filter_key, subject_filter_key)))
        box_title_suffix = " (Boxplot)" if draw_boxplots else ""
        ax.set_title(" - ".join(title_parts) + box_title_suffix)
        ax.set_xlabel("Parameter Count")
        ax.set_ylabel(f"Max {metric_label} Score")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    break_start, break_end = x_break
    left_x = [x for x in all_x if x <= break_start]
    right_x = [x for x in all_x if x >= break_end]
    if not left_x or not right_x:
        raise ValueError(
            "x-break range removes one side of the line plot. "
            "Choose START/END so there are points <= START and >= END."
        )

    shared_pad = break_start - max(left_x)
    if shared_pad <= 0:
        shared_pad = max(1.0, x_span * 0.01)

    left_xlim_min = min(left_x) - shared_pad
    left_xlim_max = break_start
    right_xlim_min = break_end
    right_xlim_max = max(right_x) + shared_pad
    left_span = max(1.0, left_xlim_max - left_xlim_min)
    right_span = max(1.0, right_xlim_max - right_xlim_min)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [left_span, right_span]},
    )
    ax_left.set_xlim(left_xlim_min, left_xlim_max)
    ax_right.set_xlim(right_xlim_min, right_xlim_max)

    if draw_boxplots:
        _plot_boxplots(ax_left, x_max_filter=break_start)
        _plot_boxplots(ax_right, x_min_filter=break_end)
    else:
        _plot_runs(ax_left, x_max_filter=break_start)
        _plot_runs(ax_right, x_min_filter=break_end)

    _build_legend(ax_right)
    title_parts = [f"Score by Parameter Count ({metric_label})"]
    if split_filter is not None:
        title_parts.append(str(split_filter))
    if subject_filter_key is not None:
        title_parts.append(str(subject_labels.get(subject_filter_key, subject_filter_key)))
    box_title_suffix = " (Boxplot)" if draw_boxplots else ""
    ax_left.set_title(" - ".join(title_parts) + box_title_suffix)
    ax_left.set_xlabel("Parameter Count")
    ax_right.set_xlabel("Parameter Count")
    ax_left.set_ylabel(f"Max {metric_label} Score")
    ax_left.grid(axis="y", alpha=0.25)
    ax_right.grid(axis="y", alpha=0.25)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(labelright=False)

    marker_kwargs = dict(
        marker=[(-1, -1), (1, 1)],
        markersize=8,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **marker_kwargs)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **marker_kwargs)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_sample_split_average_plot(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    output_path: str,
    metric_label: str = "Val",
    split_filter: str = "sample-split",
    sample_split_keys: Optional[Sequence[str]] = None,
    subject_split_keys: Optional[Sequence[str]] = None,
    subject_filter_key: Optional[str] = None,
    dendrite_percent_to_graph: Optional[float] = None,
    flip_axes: bool = False,
) -> None:
    """Create sample-split average plot: d0 average vs best dendrite average.

    X-axis: sample split percent.
    Y-axis: score.
    """
    sample_data: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        run_name = str(row.get("run_name", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if not run_name or not run_id:
            continue

        if _extract_split_label(run_name) != split_filter:
            continue

        subject_key, _subject_label, sample_key, _sample_label = _extract_subject_sample_from_run_name(run_name)
        if subject_key is None or sample_key is None:
            continue

        if sample_split_keys is not None and str(sample_key) not in sample_split_keys:
            continue
        if subject_split_keys is not None and str(subject_key) not in subject_split_keys:
            continue
        if subject_filter_key is not None and str(subject_key) != str(subject_filter_key):
            continue

        sample_entry = sample_data.setdefault(
            str(sample_key),
            {
                "run_ids": set(),
                "scores": defaultdict(list),
                "run_ids_by_dendrite": defaultdict(set),
            },
        )
        sample_entry["run_ids"].add(run_id)

        for col in dendrite_columns:
            model_id, dendrite_idx = _parse_model_and_dendrite(col)
            if model_id is None or dendrite_idx is None:
                continue
            value = pd.to_numeric(row.get(col, None), errors="coerce")
            if pd.isna(value):
                continue
            sample_entry["scores"][dendrite_idx].append(float(value))
            sample_entry["run_ids_by_dendrite"][dendrite_idx].add(run_id)

    if not sample_data:
        raise ValueError("No sample-split run data found for average sample-split plot.")

    x_vals: List[float] = []
    d0_vals: List[float] = []
    best_d_vals: List[float] = []

    for sample_key in sorted(sample_data.keys(), key=_percent_key_sort_key):
        sample_pct = _percent_key_to_float(sample_key)
        if sample_pct is None:
            continue

        entry = sample_data[sample_key]
        total_runs = len(entry["run_ids"])
        if total_runs <= 0:
            continue

        allowed_dendrites: List[int] = []
        for dendrite_idx in sorted(entry["scores"].keys()):
            if dendrite_percent_to_graph is None:
                allowed_dendrites.append(dendrite_idx)
                continue
            present_runs = len(entry["run_ids_by_dendrite"].get(dendrite_idx, set()))
            if (present_runs / total_runs) >= dendrite_percent_to_graph:
                allowed_dendrites.append(dendrite_idx)

        if 0 not in allowed_dendrites:
            continue

        d0_scores = entry["scores"].get(0, [])
        if not d0_scores:
            continue

        dendrite_candidates = [d for d in allowed_dendrites if d > 0 and entry["scores"].get(d)]
        if not dendrite_candidates:
            continue

        best_avg = max(sum(entry["scores"][d]) / len(entry["scores"][d]) for d in dendrite_candidates)
        d0_avg = sum(d0_scores) / len(d0_scores)

        x_vals.append(sample_pct)
        d0_vals.append(d0_avg)
        best_d_vals.append(best_avg)

    if not x_vals:
        raise ValueError("No sample-split points passed filters for average sample-split plot.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if flip_axes:
        ax.plot(d0_vals, x_vals, marker="o", linewidth=2, markersize=5, label="Baseline Model")
        ax.plot(best_d_vals, x_vals, marker="s", linewidth=2, markersize=5, label="Best Dendrite Average")
    else:
        ax.plot(x_vals, d0_vals, marker="o", linewidth=2, markersize=5, label="Baseline Model")
        ax.plot(x_vals, best_d_vals, marker="s", linewidth=2, markersize=5, label="Best Dendrite Average")

    title_parts = ["Sample-Split Averages", str(metric_label)]
    if subject_filter_key is not None:
        title_parts.append(str(subject_filter_key))
    ax.set_title(" - ".join(title_parts))
    if flip_axes:
        ax.set_xlabel(f"Max {metric_label} Score")
        ax.set_ylabel("Sample Split (%)")
    else:
        ax.set_xlabel("Sample Split (%)")
        ax.set_ylabel(f"Max {metric_label} Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_split_pct", "baseline_model", "best_dendrite_average"])
        for x, d0, bd in zip(x_vals, d0_vals, best_d_vals):
            writer.writerow([x, d0, bd])


def _catmull_rom_smooth_series(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    samples_per_segment: int = 40,
) -> Tuple[List[float], List[float]]:
    """Create a smooth curve through points using Catmull-Rom interpolation."""
    points = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
    if len(points) < 3:
        return [p[0] for p in points], [p[1] for p in points]

    def _interp(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    smooth_x: List[float] = []
    smooth_y: List[float] = []
    n = len(points)

    for i in range(n - 1):
        x0, y0 = points[i - 1] if i > 0 else points[i]
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        x3, y3 = points[i + 2] if (i + 2) < n else points[i + 1]

        for j in range(samples_per_segment):
            t = j / float(samples_per_segment)
            smooth_x.append(_interp(x0, x1, x2, x3, t))
            smooth_y.append(_interp(y0, y1, y2, y3, t))

    smooth_x.append(points[-1][0])
    smooth_y.append(points[-1][1])
    return smooth_x, smooth_y


def _find_left_intersection_x(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    target_y: float,
    start_x: float,
) -> Optional[float]:
    """Find the nearest intersection on/left of start_x for y(x)=target_y."""
    if len(x_vals) < 2:
        return None

    best_x: Optional[float] = None
    for i in range(len(x_vals) - 1):
        x1, y1 = x_vals[i], y_vals[i]
        x2, y2 = x_vals[i + 1], y_vals[i + 1]

        seg_min_x = min(x1, x2)
        seg_max_x = max(x1, x2)
        if seg_min_x > start_x:
            continue

        if y1 == y2:
            if y1 != target_y:
                continue
            candidate_x = min(start_x, seg_max_x)
            if candidate_x >= seg_min_x and (best_x is None or candidate_x > best_x):
                best_x = candidate_x
            continue

        if (target_y - y1) * (target_y - y2) > 0:
            continue

        t = (target_y - y1) / (y2 - y1)
        if t < 0.0 or t > 1.0:
            continue
        candidate_x = x1 + t * (x2 - x1)
        if candidate_x <= start_x and (best_x is None or candidate_x > best_x):
            best_x = candidate_x

    return best_x


def _create_sample_split_average_plot_smoothed(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    output_path: str,
    metric_label: str = "Val",
    split_filter: str = "sample-split",
    sample_split_keys: Optional[Sequence[str]] = None,
    subject_split_keys: Optional[Sequence[str]] = None,
    subject_filter_key: Optional[str] = None,
    dendrite_percent_to_graph: Optional[float] = None,
) -> None:
    """Create smoothed 2-line sample-split average chart without markers.

    Includes a horizontal guide line from the top-right Baseline Model point
    to where it intersects the Best Dendrite Average curve.
    """
    sample_data: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        run_name = str(row.get("run_name", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if not run_name or not run_id:
            continue

        if _extract_split_label(run_name) != split_filter:
            continue

        subject_key, _subject_label, sample_key, _sample_label = _extract_subject_sample_from_run_name(run_name)
        if subject_key is None or sample_key is None:
            continue

        if sample_split_keys is not None and str(sample_key) not in sample_split_keys:
            continue
        if subject_split_keys is not None and str(subject_key) not in subject_split_keys:
            continue
        if subject_filter_key is not None and str(subject_key) != str(subject_filter_key):
            continue

        sample_entry = sample_data.setdefault(
            str(sample_key),
            {
                "run_ids": set(),
                "scores": defaultdict(list),
                "run_ids_by_dendrite": defaultdict(set),
            },
        )
        sample_entry["run_ids"].add(run_id)

        for col in dendrite_columns:
            model_id, dendrite_idx = _parse_model_and_dendrite(col)
            if model_id is None or dendrite_idx is None:
                continue
            value = pd.to_numeric(row.get(col, None), errors="coerce")
            if pd.isna(value):
                continue
            sample_entry["scores"][dendrite_idx].append(float(value))
            sample_entry["run_ids_by_dendrite"][dendrite_idx].add(run_id)

    if not sample_data:
        raise ValueError("No sample-split run data found for smoothed average sample-split plot.")

    x_vals: List[float] = []
    d0_vals: List[float] = []
    best_d_vals: List[float] = []

    for sample_key in sorted(sample_data.keys(), key=_percent_key_sort_key):
        sample_pct = _percent_key_to_float(sample_key)
        if sample_pct is None:
            continue

        entry = sample_data[sample_key]
        total_runs = len(entry["run_ids"])
        if total_runs <= 0:
            continue

        allowed_dendrites: List[int] = []
        for dendrite_idx in sorted(entry["scores"].keys()):
            if dendrite_percent_to_graph is None:
                allowed_dendrites.append(dendrite_idx)
                continue
            present_runs = len(entry["run_ids_by_dendrite"].get(dendrite_idx, set()))
            if (present_runs / total_runs) >= dendrite_percent_to_graph:
                allowed_dendrites.append(dendrite_idx)

        if 0 not in allowed_dendrites:
            continue

        d0_scores = entry["scores"].get(0, [])
        if not d0_scores:
            continue

        dendrite_candidates = [d for d in allowed_dendrites if d > 0 and entry["scores"].get(d)]
        if not dendrite_candidates:
            continue

        best_avg = max(sum(entry["scores"][d]) / len(entry["scores"][d]) for d in dendrite_candidates)
        d0_avg = sum(d0_scores) / len(d0_scores)

        x_vals.append(sample_pct)
        d0_vals.append(d0_avg)
        best_d_vals.append(best_avg)

    if not x_vals:
        raise ValueError("No sample-split points passed filters for smoothed average sample-split plot.")

    smooth_x_d0, smooth_y_d0 = _catmull_rom_smooth_series(x_vals, d0_vals)
    smooth_x_best, smooth_y_best = _catmull_rom_smooth_series(x_vals, best_d_vals)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(smooth_x_d0, smooth_y_d0, linewidth=2.2, label="Baseline Model")
    ax.plot(smooth_x_best, smooth_y_best, linewidth=2.2, label="Best Dendrite Average")

    top_right_idx = max(range(len(x_vals)), key=lambda i: x_vals[i])
    top_right_x = x_vals[top_right_idx]
    top_right_y = d0_vals[top_right_idx]
    intersection_x = _find_left_intersection_x(smooth_x_best, smooth_y_best, top_right_y, top_right_x)
    if intersection_x is not None and intersection_x < top_right_x:
        ax.plot(
            [intersection_x, top_right_x],
            [top_right_y, top_right_y],
            linestyle="--",
            linewidth=1.5,
            color="gray",
            alpha=0.8,
        )

    title_parts = ["Sample-Split Averages", str(metric_label)]
    if subject_filter_key is not None:
        title_parts.append(str(subject_filter_key))
    ax.set_title(" - ".join(title_parts))
    ax.set_xlabel("Sample Split (%)")
    ax.set_ylabel(f"Max {metric_label} Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_subject_split_lines_by_sample_plot(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    output_path: str,
    metric_label: str = "Val",
    split_filter: str = "sample-split",
    sample_split_keys: Optional[Sequence[str]] = None,
    subject_split_keys: Optional[Sequence[str]] = None,
    dendrite_percent_to_graph: Optional[float] = None,
    score_mode: str = "best_dendrite",
) -> None:
    """Create subject-on-x plots with one line per sample split.

    score_mode:
        - "d0": uses dendrite 0 average only
        - "best_dendrite": uses best average across allowed dendrite > 0
    """
    if score_mode not in ("d0", "best_dendrite"):
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _, row in df.iterrows():
        run_name = str(row.get("run_name", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if not run_name or not run_id:
            continue

        if _extract_split_label(run_name) != split_filter:
            continue

        subject_key, _subject_label, sample_key, _sample_label = _extract_subject_sample_from_run_name(run_name)
        if subject_key is None or sample_key is None:
            continue

        subject_key = str(subject_key)
        sample_key = str(sample_key)
        if sample_split_keys is not None and sample_key not in sample_split_keys:
            continue
        if subject_split_keys is not None and subject_key not in subject_split_keys:
            continue

        key = (subject_key, sample_key)
        entry = grouped.setdefault(
            key,
            {
                "run_ids": set(),
                "scores": defaultdict(list),
                "run_ids_by_dendrite": defaultdict(set),
            },
        )
        entry["run_ids"].add(run_id)

        for col in dendrite_columns:
            model_id, dendrite_idx = _parse_model_and_dendrite(col)
            if model_id is None or dendrite_idx is None:
                continue
            value = pd.to_numeric(row.get(col, None), errors="coerce")
            if pd.isna(value):
                continue
            entry["scores"][dendrite_idx].append(float(value))
            entry["run_ids_by_dendrite"][dendrite_idx].add(run_id)

    if not grouped:
        raise ValueError("No sample-split run data found for subject-split-by-sample plot.")

    subjects_sorted = sorted({k[0] for k in grouped.keys()}, key=_percent_key_sort_key)
    samples_sorted = sorted({k[1] for k in grouped.keys()}, key=_percent_key_sort_key)
    if not subjects_sorted or not samples_sorted:
        raise ValueError("No subject/sample keys available for subject-split-by-sample plot.")

    x_subjects: List[float] = []
    for subject_key in subjects_sorted:
        subject_pct = _percent_key_to_float(subject_key)
        if subject_pct is None:
            continue
        x_subjects.append(subject_pct)

    if not x_subjects:
        raise ValueError("Could not parse subject split percentages for x-axis.")

    cmap = plt.get_cmap("viridis")
    sample_colors = {
        sample_key: cmap(i / max(1, len(samples_sorted) - 1))
        for i, sample_key in enumerate(samples_sorted)
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    plotted_any = False
    for sample_key in samples_sorted:
        x_vals: List[float] = []
        y_vals: List[float] = []

        for subject_key in subjects_sorted:
            key = (subject_key, sample_key)
            if key not in grouped:
                continue

            entry = grouped[key]
            total_runs = len(entry["run_ids"])
            if total_runs <= 0:
                continue

            allowed_dendrites: List[int] = []
            for dendrite_idx in sorted(entry["scores"].keys()):
                if dendrite_percent_to_graph is None:
                    allowed_dendrites.append(dendrite_idx)
                    continue
                present_runs = len(entry["run_ids_by_dendrite"].get(dendrite_idx, set()))
                if (present_runs / total_runs) >= dendrite_percent_to_graph:
                    allowed_dendrites.append(dendrite_idx)

            y_value: Optional[float] = None
            if score_mode == "d0":
                if 0 in allowed_dendrites and entry["scores"].get(0):
                    vals = entry["scores"][0]
                    y_value = sum(vals) / len(vals)
            else:
                candidates = [d for d in allowed_dendrites if d > 0 and entry["scores"].get(d)]
                if candidates:
                    y_value = max(sum(entry["scores"][d]) / len(entry["scores"][d]) for d in candidates)

            if y_value is None:
                continue

            subject_pct = _percent_key_to_float(subject_key)
            if subject_pct is None:
                continue
            x_vals.append(subject_pct)
            y_vals.append(y_value)

        if x_vals and y_vals:
            label = f"samp={_percent_key_to_float(sample_key):g}%" if _percent_key_to_float(sample_key) is not None else sample_key
            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                linewidth=2,
                markersize=5,
                color=sample_colors[sample_key],
                label=label,
            )
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        raise ValueError("No points passed filters for subject-split-by-sample plot.")

    mode_label = "Baseline Model" if score_mode == "d0" else "Best Dendrite Average"
    ax.set_title(f"Subject-Split Lines by Sample ({metric_label}) - {mode_label}")
    ax.set_xlabel("Subject Split (%)")
    ax.set_ylabel(f"Max {metric_label} Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_subject_split_two_line_plot_for_sample(
    df: pd.DataFrame,
    dendrite_columns: Sequence[str],
    output_path: str,
    sample_filter_key: str,
    metric_label: str = "Val",
    split_filter: str = "sample-split",
    subject_split_keys: Optional[Sequence[str]] = None,
    dendrite_percent_to_graph: Optional[float] = None,
) -> None:
    """Create one subject-axis chart for a specific sample split with 2 lines.

    Lines:
      - d0 average
      - best dendrite average (best non-zero dendrite among allowed dendrites)
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        run_name = str(row.get("run_name", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        if not run_name or not run_id:
            continue

        if _extract_split_label(run_name) != split_filter:
            continue

        subject_key, _subject_label, sample_key, _sample_label = _extract_subject_sample_from_run_name(run_name)
        if subject_key is None or sample_key is None:
            continue

        subject_key = str(subject_key)
        sample_key = str(sample_key)
        if sample_key != str(sample_filter_key):
            continue
        if subject_split_keys is not None and subject_key not in subject_split_keys:
            continue

        entry = grouped.setdefault(
            subject_key,
            {
                "run_ids": set(),
                "scores": defaultdict(list),
                "run_ids_by_dendrite": defaultdict(set),
            },
        )
        entry["run_ids"].add(run_id)

        for col in dendrite_columns:
            model_id, dendrite_idx = _parse_model_and_dendrite(col)
            if model_id is None or dendrite_idx is None:
                continue
            value = pd.to_numeric(row.get(col, None), errors="coerce")
            if pd.isna(value):
                continue
            entry["scores"][dendrite_idx].append(float(value))
            entry["run_ids_by_dendrite"][dendrite_idx].add(run_id)

    if not grouped:
        raise ValueError(
            f"No sample-split run data found for subject-axis two-line plot ({sample_filter_key})."
        )

    subjects_sorted = sorted(grouped.keys(), key=_percent_key_sort_key)

    x_vals: List[float] = []
    d0_vals: List[float] = []
    best_d_vals: List[float] = []

    for subject_key in subjects_sorted:
        subject_pct = _percent_key_to_float(subject_key)
        if subject_pct is None:
            continue

        entry = grouped[subject_key]
        total_runs = len(entry["run_ids"])
        if total_runs <= 0:
            continue

        allowed_dendrites: List[int] = []
        for dendrite_idx in sorted(entry["scores"].keys()):
            if dendrite_percent_to_graph is None:
                allowed_dendrites.append(dendrite_idx)
                continue
            present_runs = len(entry["run_ids_by_dendrite"].get(dendrite_idx, set()))
            if (present_runs / total_runs) >= dendrite_percent_to_graph:
                allowed_dendrites.append(dendrite_idx)

        if 0 not in allowed_dendrites or not entry["scores"].get(0):
            continue

        candidates = [d for d in allowed_dendrites if d > 0 and entry["scores"].get(d)]
        if not candidates:
            continue

        d0_avg = sum(entry["scores"][0]) / len(entry["scores"][0])
        best_avg = max(sum(entry["scores"][d]) / len(entry["scores"][d]) for d in candidates)

        x_vals.append(subject_pct)
        d0_vals.append(d0_avg)
        best_d_vals.append(best_avg)

    if not x_vals:
        raise ValueError(
            f"No subject points passed filters for sample {sample_filter_key} in subject-axis two-line plot."
        )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_vals, d0_vals, marker="o", linewidth=2, markersize=5, label="Baseline Model")
    ax.plot(x_vals, best_d_vals, marker="s", linewidth=2, markersize=5, label="Best Dendrite Average")

    sample_pct = _percent_key_to_float(str(sample_filter_key))
    sample_label = f"samp={sample_pct:g}%" if sample_pct is not None else str(sample_filter_key)
    ax.set_title(f"Subject-Split Two-Line Averages ({metric_label}) [{sample_label}]")
    ax.set_xlabel("Subject Split (%)")
    ax.set_ylabel(f"Max {metric_label} Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a by-dendrite-separate CSV and generate candlestick summary graphs.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to by-dendrite-separate CSV file",
    )
    parser.add_argument(
        "--output",
        help="Output directory path. If omitted, uses a folder named after the input CSV stem.",
    )
    parser.add_argument(
        "--x-break",
        help="Optional x-axis break range for param-count plot, format: START,END (e.g., 13000000,22000000)",
    )
    parser.add_argument(
        "--require-complete-repeat-dendrites",
        action="store_true",
        help="When averaging repeats, only keep dendrite points that exist in all repeats for that group.",
    )
    parser.add_argument(
        "--expected-repeat-count",
        type=int,
        default=5,
        help="Expected number of repeats per group used with --require-complete-repeat-dendrites (default: 5).",
    )
    parser.add_argument(
        "--dendrite-percent-to-graph",
        type=float,
        help=(
            "When averaging repeats, keep and graph a dendrite point only if it is present in at least "
            "this fraction of runs in the group (e.g., 0.6 means >=60%% of runs)."
        ),
    )
    parser.add_argument(
        "--sample-splits",
        help=(
            "Optional comma-separated sample split percentages to include and order explicitly "
            "(e.g., 50,62,75,87,100)."
        ),
    )
    parser.add_argument(
        "--subject-splits",
        help=(
            "Optional comma-separated subject split percentages to include and order explicitly "
            "(e.g., 50,75,100)."
        ),
    )
    args = parser.parse_args()

    x_break: Optional[Tuple[float, float]] = None
    if args.x_break:
        parts = [p.strip() for p in args.x_break.split(",")]
        if len(parts) != 2:
            print("Error: --x-break must be in format START,END", file=sys.stderr)
            sys.exit(1)
        try:
            break_start = float(parts[0])
            break_end = float(parts[1])
        except ValueError:
            print("Error: --x-break values must be numeric", file=sys.stderr)
            sys.exit(1)
        if break_start >= break_end:
            print("Error: --x-break requires START < END", file=sys.stderr)
            sys.exit(1)
        x_break = (break_start, break_end)

    if args.expected_repeat_count < 1:
        print("Error: --expected-repeat-count must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.dendrite_percent_to_graph is not None:
        if args.dendrite_percent_to_graph < 0.0 or args.dendrite_percent_to_graph > 1.0:
            print("Error: --dendrite-percent-to-graph must be between 0.0 and 1.0", file=sys.stderr)
            sys.exit(1)

    try:
        sample_split_keys = _parse_split_values_arg(args.sample_splits, "samp")
        subject_split_keys = _parse_split_values_arg(args.subject_splits, "subj")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(csv_path))
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = args.output if args.output else os.path.join(base_dir, csv_stem)
    os.makedirs(output_dir, exist_ok=True)

    try:
        model_name_map, initial_model_id = _load_model_info(base_dir)
        df, dendrite_columns, param_counts_by_column = _read_by_dendrite_separate_csv(csv_path)
        val_columns = [col for col in dendrite_columns if col.endswith("_max_val")]
        test_columns = [col for col in dendrite_columns if col.endswith("_max_test")]

        metric_runs: List[Tuple[str, str, List[str]]] = []
        if val_columns:
            metric_runs.append(("val", "Val", val_columns))
        if test_columns:
            metric_runs.append(("test", "Test", test_columns))

        if not metric_runs:
            raise ValueError(
                "No val/test dendrite columns found. Expected columns like "
                "model_0_dendrite_2_max_val or model_0_dendrite_2_max_test."
            )

        created_files: List[str] = []
        split_modes = sorted({_extract_split_label(str(name)) for name in df["run_name"].dropna().astype(str)})
        sample_only_split_mode = next((s for s in split_modes if s == "sample-split"), None)
        if sample_only_split_mode is None:
            sample_only_split_mode = next((s for s in split_modes if "sample" in s), None)

        for metric_key, metric_label, metric_columns in metric_runs:
            line_plot_path = os.path.join(output_dir, f"data_percent_line_plot_{metric_key}.png")
            _create_data_percent_line_plot(
                df,
                metric_columns,
                param_counts_by_column,
                line_plot_path,
                model_name_map=model_name_map,
                metric_label=metric_label,
                x_break=x_break,
                require_complete_repeat_dendrites=args.require_complete_repeat_dendrites,
                expected_repeat_count=args.expected_repeat_count,
                dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                sample_split_keys=sample_split_keys,
                subject_split_keys=subject_split_keys,
            )
            created_files.append(line_plot_path)

            line_plot_box_path = os.path.join(output_dir, f"data_percent_line_plot_{metric_key}_boxplot.png")
            _create_data_percent_line_plot(
                df,
                metric_columns,
                param_counts_by_column,
                line_plot_box_path,
                model_name_map=model_name_map,
                metric_label=metric_label,
                x_break=x_break,
                require_complete_repeat_dendrites=args.require_complete_repeat_dendrites,
                expected_repeat_count=args.expected_repeat_count,
                dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                sample_split_keys=sample_split_keys,
                subject_split_keys=subject_split_keys,
                draw_boxplots=True,
                average_repeats=False,
            )
            created_files.append(line_plot_box_path)

            for split_mode in split_modes:
                split_safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", split_mode)
                split_plot_path = os.path.join(
                    output_dir,
                    f"data_percent_line_plot_{metric_key}_{split_safe}.png",
                )
                _create_data_percent_line_plot(
                    df,
                    metric_columns,
                    param_counts_by_column,
                    split_plot_path,
                    model_name_map=model_name_map,
                    metric_label=metric_label,
                    x_break=x_break,
                    split_filter=split_mode,
                    require_complete_repeat_dendrites=args.require_complete_repeat_dendrites,
                    expected_repeat_count=args.expected_repeat_count,
                    dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                    sample_split_keys=sample_split_keys,
                    subject_split_keys=subject_split_keys,
                )
                created_files.append(split_plot_path)

                split_plot_box_path = os.path.join(
                    output_dir,
                    f"data_percent_line_plot_{metric_key}_{split_safe}_boxplot.png",
                )
                _create_data_percent_line_plot(
                    df,
                    metric_columns,
                    param_counts_by_column,
                    split_plot_box_path,
                    model_name_map=model_name_map,
                    metric_label=metric_label,
                    x_break=x_break,
                    split_filter=split_mode,
                    require_complete_repeat_dendrites=args.require_complete_repeat_dendrites,
                    expected_repeat_count=args.expected_repeat_count,
                    dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                    sample_split_keys=sample_split_keys,
                    subject_split_keys=subject_split_keys,
                    draw_boxplots=True,
                    average_repeats=False,
                )
                created_files.append(split_plot_box_path)

            if sample_only_split_mode is not None:
                sample_split_subject_keys = sorted(
                    {
                        str(subject_key)
                        for run_name in df["run_name"].dropna().astype(str)
                        if _extract_split_label(run_name) == sample_only_split_mode
                        for subject_key in [_extract_subject_sample_from_run_name(run_name)[0]]
                        if subject_key is not None
                    },
                    key=_percent_key_sort_key,
                    reverse=True,
                )

                if subject_split_keys is not None:
                    sample_split_subject_keys = [
                        key for key in sample_split_subject_keys if key in set(subject_split_keys)
                    ]

                for subject_key in sample_split_subject_keys:
                    subject_safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", subject_key)
                    split_safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", sample_only_split_mode)
                    split_subject_plot_path = os.path.join(
                        output_dir,
                        f"data_percent_line_plot_{metric_key}_{split_safe}_{subject_safe}.png",
                    )
                    _create_data_percent_line_plot(
                        df,
                        metric_columns,
                        param_counts_by_column,
                        split_subject_plot_path,
                        model_name_map=model_name_map,
                        metric_label=metric_label,
                        x_break=x_break,
                        split_filter=sample_only_split_mode,
                        require_complete_repeat_dendrites=args.require_complete_repeat_dendrites,
                        expected_repeat_count=args.expected_repeat_count,
                        dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                        sample_split_keys=sample_split_keys,
                        subject_split_keys=subject_split_keys,
                        subject_filter_key=subject_key,
                    )
                    created_files.append(split_subject_plot_path)

                    split_subject_box_plot_path = os.path.join(
                        output_dir,
                        f"data_percent_line_plot_{metric_key}_{split_safe}_{subject_safe}_boxplot.png",
                    )
                    _create_data_percent_line_plot(
                        df,
                        metric_columns,
                        param_counts_by_column,
                        split_subject_box_plot_path,
                        model_name_map=model_name_map,
                        metric_label=metric_label,
                        x_break=x_break,
                        split_filter=sample_only_split_mode,
                        require_complete_repeat_dendrites=args.require_complete_repeat_dendrites,
                        expected_repeat_count=args.expected_repeat_count,
                        dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                        sample_split_keys=sample_split_keys,
                        subject_split_keys=subject_split_keys,
                        subject_filter_key=subject_key,
                        draw_boxplots=True,
                        average_repeats=False,
                    )
                    created_files.append(split_subject_box_plot_path)

                for subject_key in sample_split_subject_keys:
                    subject_safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", subject_key)
                    sample_average_plot_path = os.path.join(
                        output_dir,
                        f"sample_split_average_lines_{metric_key}_{subject_safe}.png",
                    )
                    _create_sample_split_average_plot(
                        df,
                        metric_columns,
                        sample_average_plot_path,
                        metric_label=metric_label,
                        split_filter=sample_only_split_mode,
                        sample_split_keys=sample_split_keys,
                        subject_split_keys=subject_split_keys,
                        subject_filter_key=subject_key,
                        dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                    )
                    created_files.append(sample_average_plot_path)

                    if metric_key == "test" and str(subject_key) == "subj_100":
                        sample_average_flipped_plot_path = os.path.join(
                            output_dir,
                            f"sample_split_average_lines_{metric_key}_{subject_safe}_flipped.png",
                        )
                        _create_sample_split_average_plot(
                            df,
                            metric_columns,
                            sample_average_flipped_plot_path,
                            metric_label=metric_label,
                            split_filter=sample_only_split_mode,
                            sample_split_keys=sample_split_keys,
                            subject_split_keys=subject_split_keys,
                            subject_filter_key=subject_key,
                            dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                            flip_axes=True,
                        )
                        created_files.append(sample_average_flipped_plot_path)

                        sample_average_smoothed_plot_path = os.path.join(
                            output_dir,
                            f"sample_split_average_lines_{metric_key}_{subject_safe}_smoothed.png",
                        )
                        _create_sample_split_average_plot_smoothed(
                            df,
                            metric_columns,
                            sample_average_smoothed_plot_path,
                            metric_label=metric_label,
                            split_filter=sample_only_split_mode,
                            sample_split_keys=sample_split_keys,
                            subject_split_keys=subject_split_keys,
                            subject_filter_key=subject_key,
                            dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                        )
                        created_files.append(sample_average_smoothed_plot_path)

                sample_split_keys_for_subject_plots = sorted(
                    {
                        str(sample_key)
                        for run_name in df["run_name"].dropna().astype(str)
                        if _extract_split_label(run_name) == sample_only_split_mode
                        for sample_key in [_extract_subject_sample_from_run_name(run_name)[2]]
                        if sample_key is not None
                    },
                    key=_percent_key_sort_key,
                    reverse=True,
                )

                if sample_split_keys is not None:
                    sample_split_keys_for_subject_plots = [
                        key for key in sample_split_keys_for_subject_plots if key in set(sample_split_keys)
                    ]

                split_safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", sample_only_split_mode)
                for sample_key in sample_split_keys_for_subject_plots:
                    sample_safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", sample_key)
                    subject_two_line_path = os.path.join(
                        output_dir,
                        f"subject_split_two_line_{metric_key}_{split_safe}_{sample_safe}.png",
                    )
                    _create_subject_split_two_line_plot_for_sample(
                        df,
                        metric_columns,
                        subject_two_line_path,
                        sample_filter_key=sample_key,
                        metric_label=metric_label,
                        split_filter=sample_only_split_mode,
                        subject_split_keys=subject_split_keys,
                        dendrite_percent_to_graph=args.dendrite_percent_to_graph,
                    )
                    created_files.append(subject_two_line_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Output directory: {output_dir}")
    for f in created_files:
        print(f"Created: {os.path.basename(f)}")


if __name__ == "__main__":
    main()
