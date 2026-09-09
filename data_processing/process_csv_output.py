################################################################################
# Process a by-dendrite-separate CSV and generate candlestick summary plots.   #
################################################################################

#
"""
Imports
"""
import os
import re
import sys
import csv
import math
import argparse
import pai_style

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines  import Line2D
from matplotlib.colors import to_rgba
from collections       import defaultdict
from typing            import Any, Dict, List, Optional, Sequence, Tuple

#
"""
Config
"""
pai_style.apply_rc_style()

# Raw val and test score lists keyed by metric name
ScoreLists = Dict[str, List[float]]

#
"""
Functions
"""
def safe_float(value: str) -> Optional[float]:
    '''
    Read a float out of a cell, or None if blank

    Signature:
        value (str):
            - Cell text from the CSV, may also be None
    '''
    if value is None:
        return None
    text = str(value).strip()
    if text == '':
        return None
    try:
        return float(text)
    except ValueError:
        return None

def dendrite_sort_key(column_name: str) -> Tuple[int, int, int, str]:
    '''
    Sort dendrite columns by metric, then by dendrite count

    Notes:
        - Both column formats sort the same way
            -> model_<n>_dendrite_<d>_max_<val|test>
            -> dendrite_<d>_max_<val|test>
        - Anything that matches neither format goes to the end

    Signature:
        column_name (str):
            - Column such as model_0_dendrite_2_max_val
    '''
    match = re.match(
        r'(?:model_([^_]+)_)?dendrite_(\d+)_max_(val|test)$',
        column_name,
    )
    if match:
        model_tag = match.group(1) if match.group(1) is not None else '0'
        try:
            model_idx = int(model_tag)
        except ValueError:
            model_idx = 0
        dendrite_idx = int(match.group(2))
        metric       = match.group(3)
        metric_order = 0 if metric == 'val' else 1
        return (0, metric_order, dendrite_idx, f'{model_idx:06d}_{model_tag}')
    return (1, 2, 999999, column_name)

def display_label(
    column_name    : str,
    model_name_map : Optional[Dict[str, str]] = None,
) -> str:
    '''
    Shorten a dendrite column name to a plot label like d2

    Signature:
        column_name (str):
            - Column such as model_0_dendrite_2_max_val
        model_name_map (Optional[Dict[str, str]]):
            - Model id to display name, accepted but not used yet
    '''
    generic = re.match(r'dendrite_(\d+)_max_(?:val|test)$', column_name)
    if generic:
        return f'd{generic.group(1)}'

    match = re.match(
        r'model_([^_]+)_dendrite_(\d+)_max_(?:val|test)$',
        column_name,
    )
    if match:
        return f'd{match.group(2)}'
    return column_name.replace('_max_val', '').replace('_max_test', '')

def extract_dendrite_idx(column_name: str) -> Optional[int]:
    '''
    Read the dendrite index out of a column name

    Notes:
        - None if the name matches neither column format

    Signature:
        column_name (str):
            - Column such as model_0_dendrite_2_max_val
    '''
    match = re.match(
        r'(?:model_[^_]+_)?dendrite_(\d+)_max_(?:val|test)$',
        str(column_name),
    )
    if not match:
        return None
    return int(match.group(1))

def load_model_info(base_dir: str) -> Tuple[Dict[str, str], Optional[str]]:
    '''
    Read model names and the baseline model out of model_info.csv

    Notes:
        - The file needs model_id and model_name columns
        - A row with is_initial_model=true marks the baseline model
        - If the file is missing or will not parse we return an empty
          map and no baseline, so callers can carry on without names

    Signature:
        base_dir (str):
            - Folder the sweep CSV lives in, where model_info.csv should
              be
    '''
    model_info_path = os.path.join(base_dir, 'model_info.csv')
    if not os.path.exists(model_info_path):
        return {}, None

    try:
        model_info_df = pd.read_csv(model_info_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        return {}, None

    if not {'model_id', 'model_name'}.issubset(set(model_info_df.columns)):
        return {}, None

    model_name_map   : Dict[str, str] = {}
    initial_model_id : Optional[str]  = None

    for _, row in model_info_df.iterrows():
        model_id   = str(row['model_id']).strip()
        model_name = str(row['model_name']).strip()
        if not model_id or model_id.lower() == 'nan':
            continue
        if model_name and model_name.lower() != 'nan':
            model_name_map[model_id] = model_name
        if 'is_initial_model' in model_info_df.columns:
            flag = str(row['is_initial_model']).strip().lower()
            if flag in ('true', '1', 'yes'):
                initial_model_id = model_id

    return model_name_map, initial_model_id

def load_model_name_map(base_dir: str) -> Dict[str, str]:
    '''
    Read just the model name map out of model_info.csv

    Signature:
        base_dir (str):
            - Folder the sweep CSV lives in, where model_info.csv should
              be
    '''
    # Only the name map is needed here, the baseline flag is dropped
    name_map, _ = load_model_info(base_dir)
    return name_map

def parse_model_and_dendrite(
    column_name : str,
) -> Tuple[Optional[str], Optional[int]]:
    '''
    Split a column name into its model_id and dendrite index

    Notes:
        - (None, None) if the name is not model_<n>_dendrite_<d>_max_*

    Signature:
        column_name (str):
            - Column such as model_0_dendrite_2_max_val
    '''
    match = re.match(
        r'(model_\d+)_dendrite_(\d+)_max_(?:val|test)$',
        column_name,
    )
    if not match:
        return None, None
    return match.group(1), int(match.group(2))

def parse_hyperparams_from_run_name(
    run_name   : str,
    known_keys : Sequence[str],
) -> Dict[str, str]:
    '''
    Pull hyperparameter values out of a run name

    Notes:
        - A value runs from after its key up to the next known key, so
          values can hold underscores of their own

    Signature:
        run_name (str):
            - Run name made of <key>_<value> segments
        known_keys (Sequence[str]):
            - Keys to look for, such as lr or batch_size
    '''
    text                    = str(run_name)
    parsed : Dict[str, str] = {}
    if not text:
        return parsed

    escaped_keys = [re.escape(k) for k in known_keys]
    key_union    = '|'.join(escaped_keys)

    for key in known_keys:
        pattern = rf'{re.escape(key)}_(.*?)(?=_(?:{key_union})_|$)'
        match   = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value != '':
                parsed[key] = value

    return parsed

def write_companion_csv_for_png(
    png_path   : str,
    data       : Any,
    empty_note : str = 'No data available for this chart.',
) -> str:
    '''
    Save a chart's numbers as a CSV beside its PNG

    Notes:
        - The CSV is the PNG path with the extension swapped to .csv
        - If there is nothing to write we still write one row holding
          empty_note, so every PNG has a companion file

    Signature:
        png_path (str):
            - Path of the PNG the numbers belong to
        data (Any):
            - The chart data as a DataFrame, a list of dicts, or one dict
        empty_note (str):
            - Text for the single row we write when data is empty
    '''
    csv_path = os.path.splitext(png_path)[0] + '.csv'

    if isinstance(data, pd.DataFrame):
        out_df = data.copy()
    elif isinstance(data, list):
        out_df = pd.DataFrame(data)
    elif isinstance(data, dict):
        out_df = pd.DataFrame([data])
    else:
        out_df = pd.DataFrame([{'note': empty_note}])

    if out_df.empty:
        out_df = pd.DataFrame([{'note': empty_note}])

    out_df.to_csv(csv_path, index=False)
    return csv_path

def extract_model_id_from_run_name(run_name: str) -> Optional[str]:
    '''
    Read model_index_<n> out of a run name and return model_<n>

    Notes:
        - None if the run name has no model_index segment

    Signature:
        run_name (str):
            - Run name from the run_name column
    '''
    match = re.search(r'model_index_(\d+)', str(run_name))
    if not match:
        return None
    return f'model_{match.group(1)}'

def collect_per_run_best_scores_by_model(
    df               : pd.DataFrame,
    dendrite_columns : Sequence[str],
    model_name_map   : Dict[str, str],
) -> pd.DataFrame:
    '''
    Find each run's best val and test score for its model

    Notes:
        - A run's best score is the max over every dendrite count of its
          model, so one row comes out for each run
        - Rows with no model_index in run_name are skipped

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        dendrite_columns (Sequence[str]):
            - Every model_<n>_dendrite_<d>_max_<val|test> column
        model_name_map (Dict[str, str]):
            - Model id to display name, from model_info.csv
    '''
    model_metric_columns : Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: {'val': [], 'test': []}
    )
    for col in dendrite_columns:
        match = re.match(r'(model_\d+)_dendrite_\d+_max_(val|test)$', col)
        if not match:
            continue
        model_metric_columns[match.group(1)][match.group(2)].append(col)

    rows : List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        model_id = extract_model_id_from_run_name(row.get('run_name', ''))
        if model_id is None:
            continue

        model_cols = model_metric_columns.get(
            model_id, {'val': [], 'test': []}
        )
        if not model_cols['val'] and not model_cols['test']:
            continue

        best_val  = np.nan
        best_test = np.nan
        if model_cols['val']:
            best_val = pd.to_numeric(
                row[model_cols['val']], errors='coerce'
            ).max()
        if model_cols['test']:
            best_test = pd.to_numeric(
                row[model_cols['test']], errors='coerce'
            ).max()

        if pd.isna(best_val) and pd.isna(best_test):
            continue

        rows.append({
            'model_id'       : model_id,
            'model_name'     : model_name_map.get(model_id, model_id),
            'run_id'         : str(row.get('run_id', '')),
            'run_name'       : str(row.get('run_name', '')),
            'best_val_score' : (
                float(best_val) if not pd.isna(best_val) else np.nan
            ),
            'best_test_score': (
                float(best_test) if not pd.isna(best_test) else np.nan
            ),
        })

    if not rows:
        raise ValueError(
            'Could not compute per-run best model scores from CSV data.'
        )

    return pd.DataFrame(rows)

def create_bell_curve_best_scores_plot(
    run_best_scores_df     : pd.DataFrame,
    output_path            : str,
    base_model_zero_scores : Optional[ScoreLists] = None,
    base_model_zero_label  : str                  = 'Base model / d0',
) -> None:
    '''
    Plot the spread of each model's best scores as bell curves

    Notes:
        - The left panel is best val, the right panel is best test
        - Each model gets a density histogram with a normal curve fitted
          on top
        - If base_model_zero_scores is passed we also draw the base
          model's dendrite-0 scores as a hatched histogram with a dashed
          curve, so the gain over the baseline is easy to see

    Signature:
        run_best_scores_df (pd.DataFrame):
            - Output of collect_per_run_best_scores_by_model
        output_path (str):
            - Where the PNG goes
        base_model_zero_scores (Optional[ScoreLists]):
            - Val and test score lists for the base model at dendrite 0
        base_model_zero_label (str):
            - Legend text for the base model dendrite-0 series
    '''
    model_ids = sorted(
        run_best_scores_df['model_id'].dropna().unique(),
        key = lambda m: (
            int(m.split('_')[1]) if re.match(r'model_\d+$', str(m)) else 9999
        ),
    )
    if not model_ids:
        raise ValueError('No model ids found for bell-curve plot.')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    metric_specs = [
        ('best_val_score', 'Best Val Score'),
        ('best_test_score', 'Best Test Score'),
    ]

    for ax, (metric_col, metric_title) in zip(axes, metric_specs):
        any_drawn = False

        metric_values_all = (
            run_best_scores_df[metric_col].dropna().astype(float).tolist()
        )
        global_bins = None
        if metric_values_all:
            base_bins = max(
                5, min(15, int(math.sqrt(len(metric_values_all))) + 2)
            )
            doubled_bins = max(10, min(40, base_bins * 2))
            global_bins  = np.histogram_bin_edges(
                metric_values_all, bins=doubled_bins
            )

        for idx, model_id in enumerate(model_ids):
            subset = run_best_scores_df[
                run_best_scores_df['model_id'] == model_id
            ]
            values = subset[metric_col].dropna().astype(float).tolist()
            if not values:
                continue

            any_drawn = True
            color     = pai_style.stream_color(idx, len(model_ids))
            label     = str(subset['model_name'].iloc[0])

            if global_bins is not None:
                ax.hist(
                    values, bins=global_bins, density=True, alpha=0.2,
                    color = color,
                )
            else:
                ax.hist(values, bins=10, density=True, alpha=0.2, color=color)

            if len(values) >= 2:
                mean_v = float(np.mean(values))
                std_v  = float(np.std(values))
                if std_v > 1e-12:
                    x_min = min(values)
                    x_max = max(values)
                    if x_max == x_min:
                        x_min -= 1.0
                        x_max += 1.0
                    x_vals = np.linspace(x_min, x_max, 200)
                    y_vals = (
                        1.0 / (std_v * math.sqrt(2.0 * math.pi))
                    ) * np.exp(
                        -((x_vals - mean_v) ** 2) / (2.0 * (std_v ** 2))
                    )
                    ax.plot(
                        x_vals, y_vals, color=color, linewidth=2, label=label
                    )
                else:
                    ax.axvline(values[0], color=color, linewidth=2, label=label)
            else:
                ax.axvline(values[0], color=color, linewidth=2, label=label)

        # Draw the base model dendrite-0 histogram as a separate series
        if base_model_zero_scores is not None:
            metric_key = 'val' if metric_col == 'best_val_score' else 'test'
            base_values = [
                float(v)
                for v in base_model_zero_scores.get(metric_key, [])
                if not pd.isna(v)
            ]
            if base_values:
                any_drawn = True
                if global_bins is not None:
                    ax.hist(
                        base_values,
                        bins      = global_bins,
                        density   = True,
                        alpha     = 0.35,
                        color     = pai_style.neutral_ink,
                        edgecolor = pai_style.neutral_ink,
                        hatch     = '//',
                        label     = base_model_zero_label,
                    )
                else:
                    ax.hist(
                        base_values,
                        bins      = 10,
                        density   = True,
                        alpha     = 0.35,
                        color     = pai_style.neutral_ink,
                        edgecolor = pai_style.neutral_ink,
                        hatch     = '//',
                        label     = base_model_zero_label,
                    )

                # Draw the fitted bell curve for d0 to match the other series
                if len(base_values) >= 2:
                    base_mean = float(np.mean(base_values))
                    base_std  = float(np.std(base_values))
                    if base_std > 1e-12:
                        x_min = min(base_values)
                        x_max = max(base_values)
                        if x_max == x_min:
                            x_min -= 1.0
                            x_max += 1.0
                        x_vals = np.linspace(x_min, x_max, 200)
                        y_vals = (
                            1.0 / (base_std * math.sqrt(2.0 * math.pi))
                        ) * np.exp(
                            -((x_vals - base_mean) ** 2)
                            / (2.0 * (base_std ** 2))
                        )
                        ax.plot(
                            x_vals,
                            y_vals,
                            color     = pai_style.neutral_ink,
                            linewidth = 2.2,
                            linestyle = '--',
                        )
                    else:
                        ax.axvline(
                            base_values[0],
                            color     = pai_style.neutral_ink,
                            linewidth = 2.2,
                            linestyle = '--',
                        )
                else:
                    ax.axvline(
                        base_values[0],
                        color     = pai_style.neutral_ink,
                        linewidth = 2.2,
                        linestyle = '--',
                    )

        ax.set_title(metric_title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        pai_style.apply_axes_style(ax)
        if any_drawn:
            ax.legend(**pai_style.legend_kwargs)

    fig.suptitle('Bell-Curve Style Comparison of Per-Run Best Scores by Model')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pai_style.save_figure(fig, output_path)

def collect_base_model_zero_scores(
    df            : pd.DataFrame,
    base_model_id : str,
) -> ScoreLists:
    '''
    Gather the base model's dendrite-0 val and test scores

    Notes:
        - A missing column just gives an empty list for that metric

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        base_model_id (str):
            - Id of the baseline model, for example model_0
    '''
    out      : ScoreLists = {'val': [], 'test': []}
    val_col               = f'{base_model_id}_dendrite_0_max_val'
    test_col              = f'{base_model_id}_dendrite_0_max_test'

    if val_col in df.columns:
        out['val'] = (
            pd.to_numeric(df[val_col], errors='coerce')
            .dropna().astype(float).tolist()
        )
    if test_col in df.columns:
        out['test'] = (
            pd.to_numeric(df[test_col], errors='coerce')
            .dropna().astype(float).tolist()
        )

    return out

def create_base_model_d0_vs_final_histogram(
    df             : pd.DataFrame,
    base_model_id  : str,
    output_path    : str,
    model_name_map : Optional[Dict[str, str]] = None,
) -> Tuple[List[str], pd.DataFrame]:
    '''
    Plot base model dendrite-0 scores against each run's final score

    Notes:
        - We write one PNG for each metric, with _val or _test inserted
          before the .png of output_path
        - Rows are grouped by run_id, so each run contributes one
          dendrite-0 score and one final score for each metric
        - The final score is the one at the highest dendrite index that
          has a value in that run

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        base_model_id (str):
            - Id of the baseline model, for example model_0
        output_path (str):
            - PNG path we derive the two metric file names from
        model_name_map (Optional[Dict[str, str]]):
            - Model id to display name, used for the base model label
    '''
    model_pattern = re.compile(
        rf'{re.escape(base_model_id)}_dendrite_(\d+)_max_(val|test)$'
    )
    generic_pattern = re.compile(r'dendrite_(\d+)_max_(val|test)$')
    metric_to_cols  : Dict[str, List[Tuple[int, str]]] = {
        'val': [], 'test': []
    }

    for col in df.columns:
        match = model_pattern.match(str(col))
        if not match:
            match = generic_pattern.match(str(col))
        if not match:
            continue
        dendrite_idx = int(match.group(1))
        metric       = match.group(2)
        metric_to_cols[metric].append((dendrite_idx, col))

    if (not metric_to_cols['val']) and (not metric_to_cols['test']):
        created_paths : List[str] = []
        for metric in ('val', 'test'):
            metric_output_path = output_path.replace('.png', f'_{metric}.png')
            fig, ax            = plt.subplots(figsize=(10, 4))
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                f'No dendrite columns found for base model: {base_model_id}',
                ha       = 'center',
                va       = 'center',
                fontsize = 11,
            )
            fig.tight_layout()
            pai_style.save_figure(fig, metric_output_path)
            created_paths.append(metric_output_path)
        note = f'No dendrite columns found for base model: {base_model_id}'
        return created_paths, pd.DataFrame([{'note': note}])

    base_label = base_model_id
    if model_name_map:
        base_label = model_name_map.get(base_model_id, base_model_id)

    # Limit to runs of the selected base model when run_name is available
    base_rows_df = df
    if 'run_name' in df.columns:
        model_ids = df['run_name'].apply(extract_model_id_from_run_name)
        matched   = df[model_ids == base_model_id]
        if not matched.empty:
            base_rows_df = matched

    # Group by run so one run does not contribute several dendrite rows
    if 'run_id' in base_rows_df.columns:
        run_groups = list(base_rows_df.groupby('run_id', dropna=False))
    else:
        run_groups = [
            (idx, row.to_frame().T) for idx, row in base_rows_df.iterrows()
        ]

    hist_rows     : List[Dict[str, Any]] = []
    created_paths : List[str]            = []
    metric_specs  = [
        ('val', 'Validation'),
        ('test', 'Test'),
    ]

    for metric, metric_title in metric_specs:
        fig, ax = plt.subplots(figsize=(10, 5), sharey=False)
        cols    = sorted(metric_to_cols[metric], key=lambda x: x[0])
        if not cols:
            ax.set_title(f'{metric_title} (not available)')
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            pai_style.apply_axes_style(ax)
            metric_output_path = output_path.replace('.png', f'_{metric}.png')
            fig.tight_layout()
            pai_style.save_figure(fig, metric_output_path)
            created_paths.append(metric_output_path)
            continue

        d0_cols = [c for c in cols if c[0] == 0]
        if not d0_cols:
            ax.set_title(f'{metric_title} (d0 not available)')
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            pai_style.apply_axes_style(ax)
            metric_output_path = output_path.replace('.png', f'_{metric}.png')
            fig.tight_layout()
            pai_style.save_figure(fig, metric_output_path)
            created_paths.append(metric_output_path)
            continue

        d0_col       = d0_cols[0][1]
        gt_zero_cols = [(d_idx, c_name) for d_idx, c_name in cols if d_idx > 0]

        d0_values            : List[float] = []
        final_gt_zero_values : List[float] = []

        gt_zero_cols_sorted = sorted(gt_zero_cols, key=lambda x: x[0])

        for _, group_df in run_groups:
            run_key = ''
            if 'run_id' in group_df.columns:
                run_key = str(group_df['run_id'].iloc[0])
            # d0 score per run
            d0_series = pd.to_numeric(group_df[d0_col], errors='coerce')
            d0_series = d0_series.dropna()
            if not d0_series.empty:
                score = float(d0_series.max())
                d0_values.append(score)
                hist_rows.append({
                    'model_id'  : base_model_id,
                    'model_name': base_label,
                    'metric'    : metric,
                    'bucket'    : 'd0',
                    'run_id'    : run_key,
                    'score'     : score,
                })

            # Final d>0 per run is the highest dendrite index with a score
            chosen_final : Optional[float] = None
            for _, col_name in reversed(gt_zero_cols_sorted):
                series = pd.to_numeric(group_df[col_name], errors='coerce')
                series = series.dropna()
                if not series.empty:
                    chosen_final = float(series.max())
                    break
            if chosen_final is not None:
                final_gt_zero_values.append(chosen_final)
                hist_rows.append({
                    'model_id'  : base_model_id,
                    'model_name': base_label,
                    'metric'    : metric,
                    'bucket'    : 'final_d_gt_0',
                    'run_id'    : run_key,
                    'score'     : chosen_final,
                })

        if (not d0_values) and (not final_gt_zero_values):
            ax.set_title(f'{metric_title} (no data)')
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            pai_style.apply_axes_style(ax)
            metric_output_path = output_path.replace('.png', f'_{metric}.png')
            fig.tight_layout()
            pai_style.save_figure(fig, metric_output_path)
            created_paths.append(metric_output_path)
            continue

        combined : List[float] = d0_values + final_gt_zero_values
        if combined:
            n_bins = max(10, min(50, (int(math.sqrt(len(combined))) + 2) * 2))
            bins   = np.histogram_bin_edges(combined, bins=n_bins)
        else:
            bins = 10

        if d0_values:
            ax.hist(
                d0_values,
                bins      = bins,
                alpha     = 0.45,
                color     = pai_style.neutral_ink,
                edgecolor = pai_style.neutral_ink,
                label     = 'd0',
            )
        if final_gt_zero_values:
            ax.hist(
                final_gt_zero_values,
                bins      = bins,
                alpha     = 0.45,
                color     = pai_style.stream_palette[2],
                edgecolor = pai_style.stream_palette[2],
                label     = 'final d>0 per run',
            )

        ax.set_title(f'{metric_title}: {base_label} d0 vs final d>0')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        pai_style.apply_axes_style(ax)
        ax.legend(**pai_style.legend_kwargs)

        metric_output_path = output_path.replace('.png', f'_{metric}.png')
        fig.tight_layout()
        pai_style.save_figure(fig, metric_output_path)
        created_paths.append(metric_output_path)

    return created_paths, pd.DataFrame(hist_rows)

def write_best_run_per_model_csv(
    run_best_scores_df : pd.DataFrame,
    output_dir         : str,
    select_metric      : str          = 'val',
) -> str:
    '''
    Write a CSV of each model's best run

    Notes:
        - If a model has no score for select_metric we pick by the other
          metric instead, so every model still gets a row

    Signature:
        run_best_scores_df (pd.DataFrame):
            - Output of collect_per_run_best_scores_by_model
        output_dir (str):
            - Folder the CSV goes in
        select_metric (str):
            - val or test, which score to rank runs by
    '''
    rows_out : List[Dict[str, Any]] = []

    if select_metric not in ('val', 'test'):
        raise ValueError("select_metric must be 'val' or 'test'.")

    model_ids = sorted(
        run_best_scores_df['model_id'].dropna().unique(),
        key = lambda m: (
            int(m.split('_')[1]) if re.match(r'model_\d+$', str(m)) else 9999
        ),
    )

    for model_id in model_ids:
        subset = run_best_scores_df[
            run_best_scores_df['model_id'] == model_id
        ]
        subset = subset.copy()
        if subset.empty:
            continue

        # Select by the requested metric and fall back to the other one
        if select_metric == 'val':
            primary_col  = 'best_val_score'
            fallback_col = 'best_test_score'
        else:
            primary_col  = 'best_test_score'
            fallback_col = 'best_val_score'

        if subset[primary_col].notna().any():
            best_idx = subset[primary_col].idxmax()
        elif subset[fallback_col].notna().any():
            best_idx = subset[fallback_col].idxmax()
        else:
            continue

        best_row   = subset.loc[best_idx]
        val_score  = ''
        test_score = ''
        if not pd.isna(best_row['best_val_score']):
            val_score = round(float(best_row['best_val_score']), 6)
        if not pd.isna(best_row['best_test_score']):
            test_score = round(float(best_row['best_test_score']), 6)
        rows_out.append({
            'model_name': str(best_row.get('model_name', model_id)),
            'run_name'  : str(best_row.get('run_name', '')),
            'val_score' : val_score,
            'test_score': test_score,
        })

    if not rows_out:
        raise ValueError('No best-run-per-model rows could be generated.')

    if select_metric == 'val':
        output_name = 'best_run_scores_by_model_best_val.csv'
    else:
        output_name = 'best_run_scores_by_model_best_test.csv'
    output_path = os.path.join(output_dir, output_name)
    pd.DataFrame(
        rows_out,
        columns = ['model_name', 'run_name', 'val_score', 'test_score'],
    ).to_csv(output_path, index=False)
    return output_path

def write_best_hyperparameter_summary(
    df               : pd.DataFrame,
    dendrite_columns : Sequence[str],
    output_dir       : str,
    model_name_map   : Dict[str, str],
) -> str:
    '''
    Write each model's best hyperparameters by val and by test

    Notes:
        - We parse hyperparameters out of run_name with a fixed list of
          common keys such as lr and batch_size
        - Each row also keeps the other metric's score for the chosen
          run, so you can see whether val and test agree

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        dendrite_columns (Sequence[str]):
            - Every model_<n>_dendrite_<d>_max_<val|test> column
        output_dir (str):
            - Folder the CSV goes in
        model_name_map (Dict[str, str]):
            - Model id to display name, from model_info.csv
    '''

    def extract_model_id_from_row(row: pd.Series) -> Optional[str]:
        '''
        Read model_<n> out of a row's run_name field

        Signature:
            row (pd.Series):
                - One data row from df
        '''
        run_name = str(row.get('run_name', ''))
        m        = re.search(r'model_index_(\d+)', run_name)
        if m:
            return f'model_{m.group(1)}'
        return None

    # Build model -> metric columns map from standard column names
    model_metric_columns : Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: {'val': [], 'test': []}
    )
    for col in dendrite_columns:
        m = re.match(r'(model_\d+)_dendrite_\d+_max_(val|test)$', col)
        if m:
            model_metric_columns[m.group(1)][m.group(2)].append(col)

    # Build model groups from rows
    model_to_row_indices : Dict[str, List[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        model_id = extract_model_id_from_row(row)
        if model_id:
            model_to_row_indices[model_id].append(idx)

    if not model_to_row_indices:
        raise ValueError(
            'Could not identify model_index in run_name for '
            'best-hyperparameter summary.'
        )

    # Hyperparameter keys that may be encoded in run_name strings
    common_hparam_keys = [
        'model_index',
        'dataset',
        'data_percent',
        'lr',
        'weight_decay',
        'label_smoothing',
        'scheduler_mode',
        'improvement_threshold',
        'pai_forward_function',
        'batch_size',
        'epochs',
        'lr_warmup_epochs',
    ]

    rows_output          : List[Dict[str, Any]] = []
    all_hparam_keys_seen                        = set()

    model_ids = sorted(
        model_to_row_indices.keys(), key=lambda m: int(m.split('_')[1])
    )
    for model_id in model_ids:
        model_rows = df.loc[model_to_row_indices[model_id]]
        metric_cols_for_model = model_metric_columns.get(
            model_id, {'val': [], 'test': []}
        )

        for metric_key in ('val', 'test'):
            metric_cols = metric_cols_for_model.get(metric_key, [])
            if not metric_cols:
                continue

            # Per-row best score across dendrite counts for this model
            per_row_best = model_rows[metric_cols].apply(
                pd.to_numeric, errors='coerce'
            ).max(axis=1)
            per_row_best = per_row_best.dropna()
            if per_row_best.empty:
                continue

            best_idx   = per_row_best.idxmax()
            best_row   = df.loc[best_idx]
            best_score = float(per_row_best.loc[best_idx])

            # Keep the opposite metric score of the same run for reference
            other_metric_key  = 'test' if metric_key == 'val' else 'val'
            other_metric_cols = metric_cols_for_model.get(other_metric_key, [])
            other_score       = None
            if other_metric_cols:
                other_score_series = pd.to_numeric(
                    best_row[other_metric_cols], errors='coerce'
                )
                if not other_score_series.dropna().empty:
                    other_score = float(other_score_series.max())

            parsed_hparams = parse_hyperparams_from_run_name(
                str(best_row.get('run_name', '')), common_hparam_keys
            )
            all_hparam_keys_seen.update(parsed_hparams.keys())

            opposite_score = ''
            if other_score is not None:
                opposite_score = round(other_score, 6)
            row_out : Dict[str, Any] = {
                'model_id'             : model_id,
                'model_name'           : model_name_map.get(model_id, model_id),
                'selected_for'         : f'best_{metric_key}',
                'run_id'               : str(best_row.get('run_id', '')),
                'run_name'             : str(best_row.get('run_name', '')),
                'final_val_score'      : '',
                'final_test_score'     : '',
                'selected_score'       : round(best_score, 6),
                'opposite_metric_score': opposite_score,
            }

            if metric_key == 'val':
                row_out['final_val_score'] = round(best_score, 6)
            else:
                row_out['final_test_score'] = round(best_score, 6)

            for key, value in parsed_hparams.items():
                row_out[key] = value

            rows_output.append(row_out)

    if not rows_output:
        raise ValueError(
            'No best-hyperparameter rows could be produced from input data.'
        )

    ordered_hparams = [
        k for k in common_hparam_keys if k in all_hparam_keys_seen
    ]
    extra_hparams   = sorted(
        k for k in all_hparam_keys_seen if k not in ordered_hparams
    )
    ordered_hparams.extend(extra_hparams)

    fixed_columns = [
        'model_id',
        'model_name',
        'selected_for',
        'run_id',
        'run_name',
    ]
    score_columns = [
        'final_val_score',
        'final_test_score',
        'selected_score',
        'opposite_metric_score',
    ]
    output_columns = fixed_columns + ordered_hparams + score_columns

    out_path = os.path.join(
        output_dir, 'best_hyperparameters_by_model_val_test.csv'
    )
    pd.DataFrame(rows_output, columns=output_columns).to_csv(
        out_path, index=False
    )
    return out_path

def build_column_color_map(dendrite_columns: Sequence[str]) -> Dict[str, str]:
    '''
    Give each model a hue and shade it by dendrite count

    Notes:
        - Columns that do not look like model_<n>_dendrite_<d> get gray,
          so they still draw without taking a model hue

    Signature:
        dendrite_columns (Sequence[str]):
            - Every metric column that will be drawn
    '''
    model_to_columns : Dict[str, List[Tuple[int, str]]] = {}
    unknown_columns  : List[str]                        = []

    for col in dendrite_columns:
        model_id, dendrite_idx = parse_model_and_dendrite(col)
        if model_id is None or dendrite_idx is None:
            unknown_columns.append(col)
            continue
        model_to_columns.setdefault(model_id, []).append((dendrite_idx, col))

    def model_sort_key(model_id: str) -> Tuple[int, int, str]:
        '''
        Sort model ids by their number, ids we cannot parse go last

        Signature:
            model_id (str):
                - Model id such as model_3
        '''
        m = re.match(r'model_(\d+)$', model_id)
        if m:
            return (0, int(m.group(1)), model_id)
        return (1, 0, model_id)

    model_ids = sorted(model_to_columns.keys(), key=model_sort_key)
    n_models  = max(1, len(model_ids))

    color_map : Dict[str, str] = {}

    for i, model_id in enumerate(model_ids):
        base = pai_style.stream_color(i, n_models)
        columns_for_model = sorted(
            model_to_columns[model_id], key=lambda x: x[0]
        )
        n_dendrites = max(1, len(columns_for_model))

        for j, (_, col) in enumerate(columns_for_model):
            fraction       = 0.0 if n_dendrites == 1 else j / (n_dendrites - 1)
            color_map[col] = pai_style.shade_color(base, fraction)

    for col in unknown_columns:
        color_map[col] = '#999999'

    return color_map

def read_by_dendrite_separate_csv(
    csv_path : str,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    '''
    Read the sweep CSV into a DataFrame plus its dendrite column metadata

    Notes:
        - We pad or trim every row to the header width, so a column
          index means the same thing in the metadata rows and the data
        - param_count for each column comes from the two metadata rows,
          looked up by column position
        - Returns the DataFrame, the sorted dendrite columns, and the
          param_count map

    Signature:
        csv_path (str):
            - Path of the by-dendrite-separate CSV
    '''
    with open(csv_path, 'r', newline='') as f:
        rows = list(csv.reader(f))

    if len(rows) < 3:
        raise ValueError(
            'CSV does not have the expected metadata + header + data layout.'
        )

    metadata_label_row = rows[0]
    metadata_value_row = rows[1]
    header_row         = rows[2]
    data_rows          = rows[3:]

    # Pad rows to header width so index-based mapping is safe
    header_len = len(header_row)
    if len(metadata_label_row) < header_len:
        metadata_label_row += [''] * (header_len - len(metadata_label_row))
    if len(metadata_value_row) < header_len:
        metadata_value_row += [''] * (header_len - len(metadata_value_row))

    normalized_data_rows = []
    for row in data_rows:
        if len(row) < header_len:
            row = row + [''] * (header_len - len(row))
        elif len(row) > header_len:
            row = row[:header_len]
        normalized_data_rows.append(row)

    df = pd.DataFrame(normalized_data_rows, columns=header_row)

    dendrite_columns = [
        col for col in df.columns
        if (col.endswith('_max_val') or col.endswith('_max_test'))
        and 'dendrite' in col
    ]

    if not dendrite_columns:
        raise ValueError(
            'No dendrite metric columns found. Expected columns like '
            'model_0_dendrite_2_max_val or model_0_dendrite_2_max_test.'
        )

    dendrite_columns = sorted(dendrite_columns, key=dendrite_sort_key)

    # Extract param_count values from metadata rows using column position
    param_counts_by_column : Dict[str, float] = {}

    header_index = {name: idx for idx, name in enumerate(header_row)}

    for col in dendrite_columns:
        idx   = header_index[col]
        label = metadata_label_row[idx].strip()
        value = metadata_value_row[idx].strip()

        # Prefer explicit metadata labeling but still allow a bare value
        parsed_value = safe_float(value)
        if label.startswith('param_count') and parsed_value is not None:
            param_counts_by_column[col] = parsed_value
        elif parsed_value is not None:
            param_counts_by_column[col] = parsed_value

    # Convert metric columns to numeric
    for col in dendrite_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, dendrite_columns, param_counts_by_column

def build_box_stats(
    df               : pd.DataFrame,
    dendrite_columns : Sequence[str],
    model_name_map   : Optional[Dict[str, str]] = None,
) -> List[Dict[str, float]]:
    '''
    Work out box plot statistics for each dendrite column

    Notes:
        - The top_ keys describe only the values at or above the median,
          which build_top_half_stats uses for the top-half plots
        - Columns with no values are skipped

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        dendrite_columns (Sequence[str]):
            - Columns to compute statistics for
        model_name_map (Optional[Dict[str, str]]):
            - Model id to display name, passed on to display_label
    '''
    stats : List[Dict[str, float]] = []

    for col in dendrite_columns:
        series = df[col].dropna()
        if series.empty:
            continue

        q1         = float(series.quantile(0.25))
        med        = float(series.quantile(0.50))
        q3         = float(series.quantile(0.75))
        top_series = series[series >= med]
        if top_series.empty:
            top_series = series
        top_q1  = float(top_series.quantile(0.25))
        top_med = float(top_series.quantile(0.50))
        top_q3  = float(top_series.quantile(0.75))

        stats.append({
            'column'    : col,
            'label'     : display_label(col, model_name_map),
            'whislo'    : float(series.min()),
            'q1'        : q1,
            'med'       : med,
            'q3'        : q3,
            'whishi'    : float(series.max()),
            'mean'      : float(series.mean()),
            'max'       : float(series.max()),
            'top_whislo': med,
            'top_q1'    : top_q1,
            'top_med'   : top_med,
            'top_q3'    : top_q3,
            'top_whishi': float(top_series.max()),
            'top_mean'  : float(top_series.mean()),
            'top_max'   : float(top_series.max()),
        })

    if not stats:
        raise ValueError('No non-empty dendrite metric data found to plot.')

    return stats

def build_box_stats_by_dendrite(
    df               : pd.DataFrame,
    dendrite_columns : Sequence[str],
) -> List[Dict[str, float]]:
    '''
    Work out box plot statistics pooled across models by dendrite count

    Notes:
        - Every model's values at the same dendrite count go into one
          pool, so the boxes describe the sweep rather than one model

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        dendrite_columns (Sequence[str]):
            - Columns to pool, grouped by their dendrite index
    '''
    dendrite_to_columns : Dict[int, List[str]] = defaultdict(list)
    for col in dendrite_columns:
        d_idx = extract_dendrite_idx(col)
        if d_idx is None:
            continue
        dendrite_to_columns[d_idx].append(col)

    stats : List[Dict[str, float]] = []
    for d_idx in sorted(dendrite_to_columns.keys()):
        pooled = pd.to_numeric(
            df[dendrite_to_columns[d_idx]].stack(), errors='coerce'
        ).dropna()
        if pooled.empty:
            continue

        q1         = float(pooled.quantile(0.25))
        med        = float(pooled.quantile(0.50))
        q3         = float(pooled.quantile(0.75))
        top_series = pooled[pooled >= med]
        if top_series.empty:
            top_series = pooled
        top_q1  = float(top_series.quantile(0.25))
        top_med = float(top_series.quantile(0.50))
        top_q3  = float(top_series.quantile(0.75))

        stats.append({
            'column'    : f'dendrite_{d_idx}',
            'label'     : f'd{d_idx}',
            'whislo'    : float(pooled.min()),
            'q1'        : q1,
            'med'       : med,
            'q3'        : q3,
            'whishi'    : float(pooled.max()),
            'mean'      : float(pooled.mean()),
            'max'       : float(pooled.max()),
            'top_whislo': med,
            'top_q1'    : top_q1,
            'top_med'   : top_med,
            'top_q3'    : top_q3,
            'top_whishi': float(top_series.max()),
            'top_mean'  : float(top_series.mean()),
            'top_max'   : float(top_series.max()),
        })

    if not stats:
        raise ValueError(
            'No non-empty dendrite metric data found to plot after grouping '
            'by dendrite count.'
        )

    return stats

def aggregate_param_counts_by_dendrite(
    dendrite_columns       : Sequence[str],
    param_counts_by_column : Dict[str, float],
) -> Dict[str, float]:
    '''
    Collapse param_count to one median value for each dendrite count

    Notes:
        - The result is keyed dendrite_<d>, matching the pooled columns
          build_box_stats_by_dendrite makes

    Signature:
        dendrite_columns (Sequence[str]):
            - Columns whose param counts we group by dendrite index
        param_counts_by_column (Dict[str, float]):
            - Column name to parameter count, from the CSV metadata rows
    '''
    grouped : Dict[int, List[float]] = defaultdict(list)
    for col in dendrite_columns:
        d_idx = extract_dendrite_idx(col)
        if d_idx is None:
            continue
        val = param_counts_by_column.get(col)
        if val is not None:
            grouped[d_idx].append(float(val))

    out : Dict[str, float] = {}
    for d_idx, values in grouped.items():
        if values:
            out[f'dendrite_{d_idx}'] = float(np.median(values))
    return out

def extract_group_series(df: pd.DataFrame, group_key: str) -> pd.Series:
    '''
    Get each row's group value from a column or run_name

    Notes:
        - If group_key is a CSV column we use that column as is
        - Otherwise we search run_name for the key and its aliases, so
          data_percent also matches pct or samp

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        group_key (str):
            - CSV column name or a hyperparameter key such as data_percent
    '''
    key = str(group_key).strip()
    if key == '':
        raise ValueError('Grouping key cannot be empty.')

    if key in df.columns:
        return df[key].fillna('').astype(str).str.strip()

    if 'run_name' not in df.columns:
        raise ValueError(
            f"Cannot group by '{key}': key is not a CSV column and run_name "
            'is not present.'
        )

    alias_map : Dict[str, List[str]] = {
        'data_percent'  : [
            'data_percent', 'data_pct', 'percent', 'pct', 'samp',
            'sample_percent',
        ],
        'sample_percent': [
            'sample_percent', 'data_percent', 'samp', 'percent', 'pct',
        ],
        'samp'          : ['samp', 'sample', 'data_percent', 'percent', 'pct'],
    }
    candidate_keys = [key]
    candidate_keys.extend(alias_map.get(key.lower(), []))

    key_patterns = [
        re.compile(rf'(?:^|_){re.escape(k)}_([^_]+)(?:_|$)')
        for k in candidate_keys
        if str(k).strip() != ''
    ]

    def extract_from_run_name(run_name: str) -> str:
        '''
        Return the first candidate key's value in a run name

        Notes:
            - An empty string means none of the keys matched

        Signature:
            run_name (str):
                - Run name from the run_name column
        '''
        text = str(run_name)
        for pattern in key_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return ''

    return df['run_name'].apply(extract_from_run_name)

def parse_group_values(filter_value: str) -> Optional[List[str]]:
    '''
    Split the --filter-value text into the group values to keep

    Notes:
        - An empty string means no filter, so we return None

    Signature:
        filter_value (str):
            - Comma-separated values such as 12,25
    '''
    if not filter_value:
        return None
    values = [
        v.strip() for v in str(filter_value).split(',') if v.strip() != ''
    ]
    return values if values else None

def sort_group_values(values: Sequence[str]) -> List[str]:
    '''
    Sort group labels as numbers when they all parse, else as text

    Signature:
        values (Sequence[str]):
            - Group labels such as 12 or 25, duplicates allowed
    '''
    unique_vals = sorted(
        set(str(v).strip() for v in values if str(v).strip() != '')
    )

    def try_float(text: str) -> Optional[float]:
        '''
        Read a float out of text, or None if not numeric

        Signature:
            text (str):
                - Group label to try as a number
        '''
        try:
            return float(text)
        except ValueError:
            return None

    if unique_vals and all(try_float(v) is not None for v in unique_vals):
        return sorted(unique_vals, key=lambda v: float(v))
    return unique_vals

def build_grouped_box_stats_by_dendrite(
    df                     : pd.DataFrame,
    dendrite_columns       : Sequence[str],
    group_series           : pd.Series,
    param_counts_by_column : Dict[str, float],
    include_groups         : Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    '''
    Work out box statistics for each group and dendrite count

    Notes:
        - Rows whose group value came back empty are dropped, since we
          cannot place them in a group
        - The returned param counts map each virtual group column to the
          median param_count of its dendrite count, so the param-count
          plots can place it on the x axis

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        dendrite_columns (Sequence[str]):
            - Columns to pool, grouped by their dendrite index
        group_series (pd.Series):
            - Group value of each row, aligned with df
        param_counts_by_column (Dict[str, float]):
            - Column name to parameter count, from the CSV metadata rows
        include_groups (Optional[Sequence[str]]):
            - Group values to keep, or None to keep all of them
    '''
    if len(group_series) != len(df):
        raise ValueError(
            'Internal error: group series length does not match dataframe '
            'rows.'
        )

    tmp                    = df.copy()
    tmp['__group_value__'] = group_series.fillna('').astype(str).str.strip()

    if include_groups is not None:
        include_set = {
            str(v).strip() for v in include_groups if str(v).strip() != ''
        }
        if include_set:
            tmp = tmp[tmp['__group_value__'].isin(include_set)]
        if tmp.empty:
            raise ValueError(
                'No rows matched requested group values. '
                f'Requested values: {sorted(include_set)}'
            )

    # Ignore rows where the grouping key did not resolve to a value
    tmp = tmp[tmp['__group_value__'] != '']
    if tmp.empty:
        raise ValueError('Grouping key resolved to empty values for all rows.')

    dendrite_to_columns : Dict[int, List[str]] = defaultdict(list)
    for col in dendrite_columns:
        d_idx = extract_dendrite_idx(col)
        if d_idx is None:
            continue
        dendrite_to_columns[d_idx].append(col)

    base_param_counts = aggregate_param_counts_by_dendrite(
        dendrite_columns, param_counts_by_column
    )

    stats                : List[Dict[str, float]] = []
    grouped_param_counts : Dict[str, float]       = {}

    for group_value in sort_group_values(tmp['__group_value__'].tolist()):
        subset = tmp[tmp['__group_value__'] == group_value]
        if subset.empty:
            continue

        for d_idx in sorted(dendrite_to_columns.keys()):
            cols   = dendrite_to_columns[d_idx]
            pooled = pd.to_numeric(subset[cols].stack(), errors='coerce')
            pooled = pooled.dropna()
            if pooled.empty:
                continue

            q1         = float(pooled.quantile(0.25))
            med        = float(pooled.quantile(0.50))
            q3         = float(pooled.quantile(0.75))
            top_series = pooled[pooled >= med]
            if top_series.empty:
                top_series = pooled
            top_q1      = float(top_series.quantile(0.25))
            top_med     = float(top_series.quantile(0.50))
            top_q3      = float(top_series.quantile(0.75))
            virtual_col = f'group_{group_value}_dendrite_{d_idx}'

            stats.append({
                'column'        : virtual_col,
                'label'         : f'{group_value} / d{d_idx}',
                'group_value'   : group_value,
                'dendrite_count': d_idx,
                'whislo'        : float(pooled.min()),
                'q1'            : q1,
                'med'           : med,
                'q3'            : q3,
                'whishi'        : float(pooled.max()),
                'mean'          : float(pooled.mean()),
                'max'           : float(pooled.max()),
                'top_whislo'    : med,
                'top_q1'        : top_q1,
                'top_med'       : top_med,
                'top_q3'        : top_q3,
                'top_whishi'    : float(top_series.max()),
                'top_mean'      : float(top_series.mean()),
                'top_max'       : float(top_series.max()),
            })

            base_key = f'dendrite_{d_idx}'
            if base_key in base_param_counts:
                grouped_param_counts[virtual_col] = float(
                    base_param_counts[base_key]
                )

    if not stats:
        raise ValueError('No grouped dendrite data found to plot.')

    return stats, grouped_param_counts

def compute_within_model_stats(
    df                     : pd.DataFrame,
    dendrite_columns       : Sequence[str],
    param_counts_by_column : Dict[str, float],
    model_name_map         : Dict[str, str],
    initial_model_id       : Optional[str],
    output_dir             : str,
    x_break                : Optional[Tuple[float, float]] = None,
    column_color_map       : Optional[Dict[str, str]]      = None,
    output_suffix          : str                           = '',
    metric_label           : str                           = 'Val',
) -> List[str]:
    '''
    Compare dendrite counts within each model and save charts and CSVs

    Notes:
        - Metric 1 is the percent of runs where a higher dendrite count
          beats the lowest one
        - Metric 2 is the average error reduction from the lowest
          dendrite count to each later one
        - Metric 3 is that error reduction divided by the extra
          parameters, against each model's own baseline dendrite count
        - Metric 4 is the percent of runs beating the baseline model's
          top 1 percent and top 5 percent thresholds, scoring each run
          by its best result with at most d dendrites
        - Returns the paths of every file written

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        dendrite_columns (Sequence[str]):
            - Every model_<n>_dendrite_<d>_max_<val|test> column
        param_counts_by_column (Dict[str, float]):
            - Column name to parameter count, from the CSV metadata rows
        model_name_map (Dict[str, str]):
            - Model id to display name, from model_info.csv
        initial_model_id (Optional[str]):
            - Baseline model for metric 4, None skips that metric
        output_dir (str):
            - Folder every chart and CSV goes in
        x_break (Optional[Tuple[float, float]]):
            - (start, end) range to cut out of the x axis, or None
        column_color_map (Optional[Dict[str, str]]):
            - Column name to color, from build_column_color_map
        output_suffix (str):
            - Text added to every file name before its extension
        metric_label (str):
            - Val or Test, used in the chart titles
    '''
    # Build model -> [(dendrite_idx, col)] lookup
    model_dendrite_cols : Dict[str, List[Tuple[int, str]]] = {}
    for col in dendrite_columns:
        model_id, dendrite_idx = parse_model_and_dendrite(col)
        if model_id is None or dendrite_idx is None:
            continue
        model_dendrite_cols.setdefault(model_id, []).append(
            (dendrite_idx, col)
        )
    for mid in model_dendrite_cols:
        model_dendrite_cols[mid].sort(key=lambda x: x[0])

    def get_model_id(run_name: str) -> Optional[str]:
        '''
        Read model_index_<n> out of a run name and return model_<n>

        Signature:
            run_name (str):
                - Run name from the run_name column
        '''
        m = re.search(r'model_index_(\d+)', str(run_name))
        # None when the run name carries no model_index
        return f'model_{m.group(1)}' if m else None

    def model_order(model_id: str) -> int:
        '''
        Turn a model id into a number we can sort on

        Notes:
            - Ids that do not parse get 9999, so they sort last

        Signature:
            model_id (str):
                - Model id such as model_3
        '''
        m = re.match(r'model_(\d+)$', model_id)
        # Unparseable ids sort last
        return int(m.group(1)) if m else 9999

    # Collect per-run score progressions
    # run_scores[run_id] = {model_id: str, scores: {dendrite_idx: float}}
    run_scores : Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        run_id   = str(row.get('run_id', ''))
        run_name = str(row.get('run_name', ''))
        model_id = get_model_id(run_name)
        if model_id is None:
            continue
        if run_id not in run_scores:
            run_scores[run_id] = {'model_id': model_id, 'scores': {}}
        if model_id in model_dendrite_cols:
            for dendrite_idx, col in model_dendrite_cols[model_id]:
                val = pd.to_numeric(row.get(col, None), errors='coerce')
                if not pd.isna(val):
                    run_scores[run_id]['scores'][dendrite_idx] = float(val)

    # Group runs by model
    model_runs : Dict[str, List[Dict[str, Any]]] = {}
    for data in run_scores.values():
        model_runs.setdefault(data['model_id'], []).append(data)

    if column_color_map is None:
        column_color_map = {}

    # Metric 1 and 2, per-model and per-run comparisons
    metric1_rows : List[Dict[str, Any]] = []
    metric2_rows : List[Dict[str, Any]] = []

    for model_id in sorted(model_runs.keys(), key=model_order):
        model_label = model_name_map.get(model_id, model_id)
        improved    = 0
        total_multi = 0

        reductions_by_dendrite : Dict[int, List[float]] = {}

        for data in model_runs[model_id]:
            scores = data['scores']
            if len(scores) < 2:
                continue
            sorted_d   = sorted(scores.keys())
            base_d     = sorted_d[0]
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
            'model_id'                    : model_id,
            'model_name'                  : model_label,
            'runs_with_multiple_dendrites': total_multi,
            'runs_improved'               : improved,
            'pct_improved'                : (
                round(pct, 2) if pct is not None else ''
            ),
        })

        for d, reds in sorted(reductions_by_dendrite.items()):
            avg_red = sum(reds) / len(reds) * 100.0
            metric2_rows.append({
                'model_id'               : model_id,
                'model_name'             : model_label,
                'dendrite_count'         : d,
                'n_runs'                 : len(reds),
                'avg_error_reduction_pct': round(avg_red, 4),
            })

    # Metric 3, per-model error reduction per extra parameter
    metric3_rows : List[Dict[str, Any]] = []

    # Average score per (model, dendrite) across all runs
    avg_scores : Dict[str, Dict[int, float]] = {}
    for model_id, runs in model_runs.items():
        dendrite_values : Dict[int, List[float]] = {}
        for data in runs:
            for d, s in data['scores'].items():
                dendrite_values.setdefault(d, []).append(s)
        for d, vals in dendrite_values.items():
            if vals:
                avg_scores.setdefault(model_id, {})[d] = sum(vals) / len(vals)

    for model_id in sorted(model_runs.keys(), key=model_order):
        model_label = model_name_map.get(model_id, model_id)
        if model_id not in model_dendrite_cols:
            continue
        # Metric 3 only applies to model types that add dendrites
        if len(model_dendrite_cols[model_id]) <= 1:
            continue

        baseline_d, baseline_col = model_dendrite_cols[model_id][0]
        baseline_score           = avg_scores.get(model_id, {}).get(baseline_d)
        baseline_params          = param_counts_by_column.get(baseline_col)
        if baseline_score is None or baseline_params is None:
            continue

        baseline_error = 100.0 - baseline_score
        if baseline_error <= 0:
            continue

        for d_idx, col in model_dendrite_cols[model_id][1:]:
            avg_s  = avg_scores.get(model_id, {}).get(d_idx)
            params = param_counts_by_column.get(col)
            if avg_s is None or params is None:
                continue
            extra_params = params - baseline_params
            if extra_params <= 0:
                continue

            error_reduction = (
                (baseline_error - (100.0 - avg_s)) / baseline_error * 100.0
            )
            metric3_rows.append({
                'model_id'                 : model_id,
                'model_name'               : model_label,
                'baseline_dendrite_count'  : baseline_d,
                'dendrite_count'           : d_idx,
                'baseline_param_count'     : int(baseline_params),
                'param_count'              : int(params),
                'extra_params_vs_baseline' : int(extra_params),
                'baseline_avg_score'       : round(baseline_score, 4),
                'avg_score'                : round(avg_s, 4),
                'error_reduction_pct'      : round(error_reduction, 4),
                'error_reduction_per_param': round(
                    error_reduction / extra_params, 8
                ),
            })

    # Metric 4, baseline top-percentile thresholds and cumulative beat
    # rates by max allowed dendrites
    metric4_rows : List[Dict[str, Any]] = []

    if initial_model_id and initial_model_id in model_dendrite_cols:
        baseline_d, _ = model_dendrite_cols[initial_model_id][0]

        baseline_scores = [
            data['scores'][baseline_d]
            for data in model_runs.get(initial_model_id, [])
            if baseline_d in data['scores']
        ]

        if baseline_scores:
            baseline_scores_sorted = sorted(baseline_scores)

            def threshold_for_top_percent(
                sorted_scores : List[float],
                top_percent   : float,
            ) -> Tuple[float, int, int]:
                '''
                Find the baseline score just under the top top_percent bucket

                Notes:
                    - top_n is ceil(n * top_percent) with a floor of 1,
                      so even a small sweep has a top bucket
                    - Returns (threshold, its 1-based rank, top_n)

                Signature:
                    sorted_scores (List[float]):
                        - Baseline model scores sorted ascending
                    top_percent (float):
                        - Share of runs in the top bucket, 0.01 or 0.05
                '''
                n     = len(sorted_scores)
                top_n = max(1, int(math.ceil(n * top_percent)))
                # Threshold is the best score just below the top bucket
                threshold_rank_1based = max(1, n - top_n)
                threshold             = sorted_scores[threshold_rank_1based - 1]
                return threshold, threshold_rank_1based, top_n

            percentile_specs = [
                (0.01, 'top_1pct'),
                (0.05, 'top_5pct'),
            ]

            for pct, label in percentile_specs:
                threshold, rank_1based, target_top_n = (
                    threshold_for_top_percent(baseline_scores_sorted, pct)
                )
                baseline_n_above = sum(
                    1 for s in baseline_scores_sorted if s > threshold
                )
                baseline_n_at_or_above = sum(
                    1 for s in baseline_scores_sorted if s >= threshold
                )

                n_scores             = len(baseline_scores_sorted)
                top_x_plus_one_count = min(n_scores, target_top_n + 1)
                top_x_plus_one_scores = sorted(
                    baseline_scores_sorted, reverse=True
                )[:top_x_plus_one_count]
                print(
                    f'Baseline threshold details for {label}: '
                    f'n_runs={n_scores}, top_n={target_top_n}, '
                    f'threshold_rank_1based={rank_1based}, '
                    f'threshold_score={threshold:.6f}'
                )
                print(
                    f'  Top {top_x_plus_one_count} baseline scores '
                    '(desc, top_n+1 view): '
                    + ', '.join(f'{s:.6f}' for s in top_x_plus_one_scores)
                )
                print(
                    f'  Baseline counts: > threshold = {baseline_n_above}, '
                    f'>= threshold = {baseline_n_at_or_above}'
                )

                baseline_param_count = int(
                    param_counts_by_column.get(
                        model_dendrite_cols[initial_model_id][0][1], 0
                    )
                )
                baseline_model_name = model_name_map.get(
                    initial_model_id, initial_model_id
                )

                for model_id in sorted(model_runs.keys(), key=model_order):
                    for d_idx, col in model_dendrite_cols.get(model_id, []):
                        if model_id == initial_model_id and d_idx == baseline_d:
                            continue

                        # Cumulative semantics, for each run take the best
                        # score with at most d_idx dendrites
                        combo_scores : List[float] = []
                        for data in model_runs.get(model_id, []):
                            eligible_scores = [
                                s for d_cur, s in data['scores'].items()
                                if d_cur <= d_idx
                            ]
                            if eligible_scores:
                                combo_scores.append(float(max(eligible_scores)))

                        if not combo_scores:
                            continue

                        beats = sum(
                            1 for s in combo_scores if s > threshold
                        )
                        pct_beats = beats / len(combo_scores) * 100.0

                        metric4_rows.append({
                            'percentile_label': label,
                            'percentile': pct,
                            'comparison_mode': (
                                'cumulative_max_allowed_dendrites'
                            ),
                            'baseline_model_id': initial_model_id,
                            'baseline_model_name': baseline_model_name,
                            'baseline_dendrite_count': baseline_d,
                            'baseline_n_runs': len(baseline_scores_sorted),
                            'baseline_target_top_n': target_top_n,
                            'baseline_threshold_rank_1based': rank_1based,
                            'baseline_threshold_score': round(
                                float(threshold), 6
                            ),
                            'baseline_n_above_threshold': baseline_n_above,
                            'baseline_n_at_or_above_threshold': (
                                baseline_n_at_or_above
                            ),
                            'baseline_param_count': baseline_param_count,
                            'model_id': model_id,
                            'model_name': model_name_map.get(
                                model_id, model_id
                            ),
                            'dendrite_count': d_idx,
                            'combo_column': col,
                            'combo_label_mode': (
                                'best_score_with_max_dendrites_leq_d'
                            ),
                            'combo_param_count': int(
                                param_counts_by_column.get(col, 0)
                            ),
                            'combo_n_runs': len(combo_scores),
                            'combo_n_beating_threshold': beats,
                            'combo_pct_beating_threshold': round(
                                pct_beats, 4
                            ),
                        })

    # Save CSVs
    created : List[str] = []

    csv1_path = os.path.join(
        output_dir, f'stats_pct_improved{output_suffix}.csv'
    )
    pd.DataFrame(metric1_rows).to_csv(csv1_path, index=False)
    created.append(csv1_path)

    csv2_path = os.path.join(
        output_dir, f'stats_error_reduction{output_suffix}.csv'
    )
    pd.DataFrame(metric2_rows).to_csv(csv2_path, index=False)
    created.append(csv2_path)

    csv3_path = os.path.join(
        output_dir, f'stats_error_reduction_per_param{output_suffix}.csv'
    )
    pd.DataFrame(metric3_rows).to_csv(csv3_path, index=False)
    created.append(csv3_path)

    csv4_columns = [
        'percentile_label',
        'percentile',
        'comparison_mode',
        'baseline_model_id',
        'baseline_model_name',
        'baseline_dendrite_count',
        'baseline_n_runs',
        'baseline_target_top_n',
        'baseline_threshold_rank_1based',
        'baseline_threshold_score',
        'baseline_n_above_threshold',
        'baseline_n_at_or_above_threshold',
        'baseline_param_count',
        'model_id',
        'model_name',
        'dendrite_count',
        'combo_column',
        'combo_label_mode',
        'combo_param_count',
        'combo_n_runs',
        'combo_n_beating_threshold',
        'combo_pct_beating_threshold',
    ]
    csv4_path = os.path.join(
        output_dir, f'stats_top_percentile_vs_baseline{output_suffix}.csv'
    )
    pd.DataFrame(metric4_rows, columns=csv4_columns).to_csv(
        csv4_path, index=False
    )
    created.append(csv4_path)

    # Chart 1, percent improved per model
    if metric1_rows:
        labels1 = [r['model_name'] for r in metric1_rows]
        values1 = [
            float(r['pct_improved']) if r['pct_improved'] != '' else 0.0
            for r in metric1_rows
        ]
        fig, ax = plt.subplots(figsize=(max(6, len(labels1) * 1.2), 5))
        bars    = ax.bar(labels1, values1)
        ax.set_ylim(0, 110)
        ax.set_title(
            f'% of Runs Improved by Adding Dendrites ({metric_label})'
        )
        ax.set_xlabel('Model')
        ax.set_ylabel('% Runs Improved')
        for bar, val in zip(bars, values1):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:.1f}%',
                ha       = 'center',
                va       = 'bottom',
                fontsize = 8,
            )
        pai_style.apply_axes_style(ax)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        fig.tight_layout()
        chart1_path = os.path.join(
            output_dir, f'stats_chart_pct_improved{output_suffix}.png'
        )
        pai_style.save_figure(fig, chart1_path)
        created.append(chart1_path)
        created.append(
            write_companion_csv_for_png(chart1_path, pd.DataFrame(metric1_rows))
        )

    # Chart 2, average error reduction by model and dendrite
    if metric2_rows:
        df2               = pd.DataFrame(metric2_rows)
        models_ordered    = sorted(df2['model_id'].unique(), key=model_order)
        dendrites_ordered = sorted(df2['dendrite_count'].unique())
        x                 = range(len(models_ordered))
        width             = 0.8 / max(1, len(dendrites_ordered))
        fig, ax = plt.subplots(
            figsize = (max(7, len(models_ordered) * 1.5), 5)
        )
        for i, d in enumerate(dendrites_ordered):
            subset = df2[df2['dendrite_count'] == d]
            vals   = []
            for mid in models_ordered:
                row = subset[subset['model_id'] == mid]
                if row.empty:
                    vals.append(0.0)
                else:
                    vals.append(float(row['avg_error_reduction_pct'].iloc[0]))
            offset    = (i - (len(dendrites_ordered) - 1) / 2.0) * width
            positions = [xi + offset for xi in x]
            ax.bar(positions, vals, width=width * 0.9, label=f'dendrite {d}')
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [model_name_map.get(mid, mid) for mid in models_ordered],
            rotation = 20,
            ha       = 'right',
        )
        ax.set_title(
            f'Avg Error Reduction (%) by Adding Dendrites ({metric_label})'
        )
        ax.set_xlabel('Model')
        ax.set_ylabel('Avg Error Reduction %')
        ax.legend(**pai_style.legend_kwargs)
        pai_style.apply_axes_style(ax)
        fig.tight_layout()
        chart2_path = os.path.join(
            output_dir, f'stats_chart_error_reduction{output_suffix}.png'
        )
        pai_style.save_figure(fig, chart2_path)
        created.append(chart2_path)
        created.append(
            write_companion_csv_for_png(chart2_path, pd.DataFrame(metric2_rows))
        )

    # Chart 3, per-model error reduction per parameter vs baseline dendrite
    if metric3_rows:
        df3               = pd.DataFrame(metric3_rows)
        models_ordered    = sorted(df3['model_id'].unique(), key=model_order)
        dendrites_ordered = sorted(df3['dendrite_count'].unique())
        x                 = range(len(models_ordered))
        width             = 0.8 / max(1, len(dendrites_ordered))
        fig, ax = plt.subplots(
            figsize = (max(7, len(models_ordered) * 1.5), 5)
        )

        for i, d in enumerate(dendrites_ordered):
            subset = df3[df3['dendrite_count'] == d]
            vals   = []
            for mid in models_ordered:
                row = subset[subset['model_id'] == mid]
                if row.empty:
                    vals.append(0.0)
                else:
                    vals.append(float(row['error_reduction_per_param'].iloc[0]))
            offset    = (i - (len(dendrites_ordered) - 1) / 2.0) * width
            positions = [xi + offset for xi in x]
            ax.bar(positions, vals, width=width * 0.9, label=f'dendrite {d}')

        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [model_name_map.get(mid, mid) for mid in models_ordered],
            rotation = 20,
            ha       = 'right',
        )
        ax.set_title(
            'Error Reduction per Parameter vs Model Baseline Dendrite '
            f'({metric_label})'
        )
        ax.set_xlabel('Model')
        ax.set_ylabel('Error Reduction % per Parameter')
        ax.legend(**pai_style.legend_kwargs)
        pai_style.apply_axes_style(ax)
        fig.tight_layout()
        chart3_path = os.path.join(
            output_dir,
            f'stats_chart_error_reduction_per_param{output_suffix}.png',
        )
        pai_style.save_figure(fig, chart3_path)
        created.append(chart3_path)
        created.append(
            write_companion_csv_for_png(chart3_path, pd.DataFrame(metric3_rows))
        )

    # Chart 4 and 5, percent of runs beating baseline thresholds using
    # cumulative max-dendrite scoring
    if metric4_rows:
        df4 = pd.DataFrame(metric4_rows)

        def plot_metric4_param_scatter(
            subset_df    : pd.DataFrame,
            title_suffix : str,
            out_name     : str,
            baseline_y   : float,
        ) -> Optional[str]:
            '''
            Scatter percent of runs beating the threshold by param count

            Notes:
                - With x_break we draw two axes sharing y, so the gap in
                  parameter counts does not squash the points together
                - Returns the PNG path, or None when there is nothing to
                  draw

            Signature:
                subset_df (pd.DataFrame):
                    - Metric 4 rows for one percentile, top 1 or top 5
                title_suffix (str):
                    - Percentile wording for the title
                out_name (str):
                    - PNG file name, written inside output_dir
                baseline_y (float):
                    - Where on the y axis the baseline star marker sits
            '''
            if subset_df.empty:
                return None

            rows = subset_df.to_dict('records')
            x_values = [
                float(r['combo_param_count'])
                for r in rows
                if float(r['combo_param_count']) > 0
            ]
            if not x_values:
                return None

            baseline_x = float(subset_df['baseline_param_count'].iloc[0])
            threshold_score = float(
                subset_df['baseline_threshold_score'].iloc[0]
            )
            baseline_name = str(subset_df['baseline_model_name'].iloc[0])
            baseline_d    = int(subset_df['baseline_dendrite_count'].iloc[0])

            x_min  = min(x_values + [baseline_x])
            x_max  = max(x_values + [baseline_x])
            x_span = x_max - x_min

            def add_scatter_legend(
                ax        : plt.Axes,
                data_rows : List[Dict[str, Any]],
            ) -> None:
                '''
                Add one legend marker for each combo column plus the baseline

                Signature:
                    ax (plt.Axes):
                        - Axes the legend attaches to
                    data_rows (List[Dict[str, Any]]):
                        - Metric 4 rows drawn on the axes
                '''
                handles   : List[Line2D] = []
                labels    : List[str]    = []
                seen_cols                = set()

                for r in data_rows:
                    col = str(r.get('combo_column', ''))
                    if not col or col in seen_cols:
                        continue
                    seen_cols.add(col)

                    model_name     = str(r.get('model_name', ''))
                    dendrite_count = int(r.get('dendrite_count', 0))
                    label          = f'{model_name} / d{dendrite_count}'
                    color          = column_color_map.get(
                        col, pai_style.stream_palette[2]
                    )

                    handles.append(Line2D(
                        [0],
                        [0],
                        marker          = 'o',
                        linestyle       = 'none',
                        markersize      = 5,
                        markerfacecolor = color,
                        markeredgecolor = color,
                    ))
                    labels.append(label)

                handles.append(Line2D(
                    [0],
                    [0],
                    marker          = '*',
                    linestyle       = 'none',
                    markersize      = 8,
                    markerfacecolor = pai_style.neutral_ink,
                    markeredgecolor = pai_style.neutral_ink,
                ))
                labels.append(f'Baseline ({baseline_name} / d{baseline_d})')

                ax.legend(
                    handles, labels, loc='lower right',
                    **pai_style.legend_kwargs,
                )

            def scatter_points(
                ax        : plt.Axes,
                data_rows : List[Dict[str, Any]],
            ) -> None:
                '''
                Draw one point for each metric 4 row

                Signature:
                    ax (plt.Axes):
                        - Axes to draw into
                    data_rows (List[Dict[str, Any]]):
                        - Metric 4 rows to draw
                '''
                for r in data_rows:
                    x   = float(r['combo_param_count'])
                    y   = float(r['combo_pct_beating_threshold'])
                    col = str(r.get('combo_column', ''))
                    color = column_color_map.get(
                        col, pai_style.stream_palette[2]
                    )
                    ax.scatter([x], [y], s=30, color=color, alpha=0.85)

            title = (
                f'By-Parameter % Above {baseline_name} d{baseline_d} '
                f'{title_suffix} Threshold ({threshold_score:.4f})\n'
                '(Cumulative: best score with max allowed dendrites <= d)'
            )

            if x_break is None:
                fig, ax = plt.subplots(figsize=(12, 6))
                x_pad   = max(1.0, x_span * 0.08)
                ax.set_xlim(x_min - x_pad, x_max + x_pad)

                scatter_points(ax, rows)
                ax.scatter(
                    [baseline_x], [baseline_y], s=80, marker='*',
                    color = pai_style.neutral_ink,
                )

                ax.set_title(title)
                ax.set_xlabel('Parameter Count')
                ax.set_ylabel('% Runs Above Baseline Threshold')
                pai_style.apply_axes_style(ax)
                ax.set_ylim(0, 100)
                add_scatter_legend(ax, rows)
                fig.tight_layout()
                out_path = os.path.join(output_dir, out_name)
                pai_style.save_figure(fig, out_path)
                return out_path

            break_start, break_end = x_break
            if break_start >= break_end:
                return None

            left_rows = [
                r for r in rows if float(r['combo_param_count']) <= break_start
            ]
            right_rows = [
                r for r in rows if float(r['combo_param_count']) >= break_end
            ]

            # Include the baseline reference point on the matching side
            baseline_on_left  = baseline_x <= break_start
            baseline_on_right = baseline_x >= break_end

            left_empty  = not left_rows and not baseline_on_left
            right_empty = not right_rows and not baseline_on_right
            if left_empty or right_empty:
                return None

            left_x_vals  = [float(r['combo_param_count']) for r in left_rows]
            right_x_vals = [float(r['combo_param_count']) for r in right_rows]
            if baseline_on_left:
                left_x_vals.append(baseline_x)
            if baseline_on_right:
                right_x_vals.append(baseline_x)

            shared_pad = break_start - max(left_x_vals)
            if shared_pad <= 0:
                shared_pad = max(1.0, x_span * 0.01)

            left_xlim_min  = min(left_x_vals) - shared_pad
            left_xlim_max  = break_start
            right_xlim_min = break_end
            right_xlim_max = max(right_x_vals) + shared_pad

            left_span  = max(1.0, left_xlim_max - left_xlim_min)
            right_span = max(1.0, right_xlim_max - right_xlim_min)

            fig, (ax_left, ax_right) = plt.subplots(
                1,
                2,
                sharey      = True,
                figsize     = (14, 6),
                gridspec_kw = {'width_ratios': [left_span, right_span]},
            )

            ax_left.set_xlim(left_xlim_min, left_xlim_max)
            ax_right.set_xlim(right_xlim_min, right_xlim_max)

            scatter_points(ax_left, left_rows)
            scatter_points(ax_right, right_rows)

            if baseline_on_left:
                ax_left.scatter(
                    [baseline_x], [baseline_y], s=80, marker='*',
                    color = pai_style.neutral_ink,
                )
            if baseline_on_right:
                ax_right.scatter(
                    [baseline_x], [baseline_y], s=80, marker='*',
                    color = pai_style.neutral_ink,
                )

            ax_left.set_title(title)
            ax_left.set_xlabel('Parameter Count')
            ax_right.set_xlabel('Parameter Count')
            ax_left.set_ylabel('% Runs Above Baseline Threshold')
            pai_style.apply_axes_style(ax_left)
            pai_style.apply_axes_style(ax_right)
            ax_left.set_ylim(0, 100)
            add_scatter_legend(ax_right, rows)

            ax_left.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            ax_right.yaxis.tick_right()
            ax_right.tick_params(labelright=False)

            marker_kwargs = dict(
                marker     = [(-1, -1), (1, 1)],
                markersize = 8,
                linestyle  = 'none',
                color      = 'k',
                mec        = 'k',
                mew        = 1,
                clip_on    = False,
            )
            ax_left.plot(
                [1, 1], [0, 1], transform=ax_left.transAxes, **marker_kwargs
            )
            ax_right.plot(
                [0, 0], [0, 1], transform=ax_right.transAxes, **marker_kwargs
            )

            fig.tight_layout()
            out_path = os.path.join(output_dir, out_name)
            pai_style.save_figure(fig, out_path)
            return out_path

        bar_specs = [
            (
                'top_1pct',
                'Top 1%',
                f'stats_chart_pct_beating_baseline_top1{output_suffix}.png',
            ),
            (
                'top_5pct',
                'Top 5%',
                f'stats_chart_pct_beating_baseline_top5{output_suffix}.png',
            ),
        ]
        for label, title_suffix, out_name in bar_specs:
            subset = df4[df4['percentile_label'] == label]
            if subset.empty:
                continue

            subset = subset.copy()
            subset['combo_label'] = subset.apply(
                lambda r: (
                    f"{r['model_name']} / max d{int(r['dendrite_count'])}"
                ),
                axis = 1,
            )
            subset = subset.sort_values(['model_id', 'dendrite_count'])

            labels = subset['combo_label'].tolist()
            vals   = (
                subset['combo_pct_beating_threshold'].astype(float).tolist()
            )

            fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5))
            bars    = ax.bar(labels, vals)
            ax.set_ylim(0, 100)
            baseline_name   = str(subset['baseline_model_name'].iloc[0])
            baseline_d      = int(subset['baseline_dendrite_count'].iloc[0])
            threshold_score = float(subset['baseline_threshold_score'].iloc[0])
            ax.set_title(
                f'% Beating {baseline_name} d{baseline_d} {title_suffix} '
                f'Threshold ({threshold_score:.4f}) [{metric_label}]\n'
                '(Cumulative: best score with max allowed dendrites <= d)'
            )
            ax.set_xlabel('Model / Max Allowed Dendrites')
            ax.set_ylabel('% Runs Above Baseline Threshold')
            pai_style.apply_axes_style(ax)
            plt.setp(ax.get_xticklabels(), rotation=35, ha='right')

            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{val:.1f}%',
                    ha       = 'center',
                    va       = 'bottom',
                    fontsize = 7,
                )

            fig.tight_layout()
            out_path = os.path.join(output_dir, out_name)
            pai_style.save_figure(fig, out_path)
            created.append(out_path)
            created.append(write_companion_csv_for_png(out_path, subset))

        scatter_top1 = plot_metric4_param_scatter(
            df4[df4['percentile_label'] == 'top_1pct'],
            'Top 1%',
            f'stats_scatter_pct_beating_baseline_top1_by_param'
            f'{output_suffix}.png',
            baseline_y = 1.0,
        )
        if scatter_top1:
            created.append(scatter_top1)
            created.append(write_companion_csv_for_png(
                scatter_top1,
                df4[df4['percentile_label'] == 'top_1pct'],
            ))

        scatter_top5 = plot_metric4_param_scatter(
            df4[df4['percentile_label'] == 'top_5pct'],
            'Top 5%',
            f'stats_scatter_pct_beating_baseline_top5_by_param'
            f'{output_suffix}.png',
            baseline_y = 5.0,
        )
        if scatter_top5:
            created.append(scatter_top5)
            created.append(write_companion_csv_for_png(
                scatter_top5,
                df4[df4['percentile_label'] == 'top_5pct'],
            ))

    return created


def create_categorical_plot(
    stats            : Sequence[Dict[str, float]],
    output_path      : str,
    column_color_map : Optional[Dict[str, str]]   = None,
    metric_label     : str                        = 'Val',
) -> None:
    '''
    Draw a candlestick plot with one labeled slot for each column

    Signature:
        stats (Sequence[Dict[str, float]]):
            - Box statistics from build_box_stats, one entry for each
              column
        output_path (str):
            - Where the PNG goes
        column_color_map (Optional[Dict[str, str]]):
            - Column name to color, from build_column_color_map
        metric_label (str):
            - Val or Test, used in the title and y label
    '''
    fig, ax = plt.subplots(figsize=(max(10, len(stats) * 0.45), 6))

    bxp_stats = [
        {
            'label' : item['label'],
            'whislo': item['whislo'],
            'q1'    : item['q1'],
            'med'   : item['med'],
            'q3'    : item['q3'],
            'whishi': item['whishi'],
        }
        for item in stats
    ]

    artists = ax.bxp(bxp_stats, showfliers=False, patch_artist=True)

    if column_color_map is None:
        column_color_map = {}

    for i, item in enumerate(stats):
        color = column_color_map.get(
            item['column'], pai_style.stream_palette[2]
        )
        artists['boxes'][i].set_facecolor(to_rgba(color, alpha=0.35))
        artists['boxes'][i].set_edgecolor(color)
        artists['boxes'][i].set_linewidth(1.2)

        artists['medians'][i].set_color(color)
        artists['medians'][i].set_linewidth(1.8)

        artists['whiskers'][2 * i].set_color(color)
        artists['whiskers'][2 * i + 1].set_color(color)
        artists['caps'][2 * i].set_color(color)
        artists['caps'][2 * i + 1].set_color(color)
    ax.set_title(f'Dendrite {metric_label} Distribution by Dendrite Count')
    ax.set_xlabel('Dendrite Count')
    ax.set_ylabel(f'Max {metric_label}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    pai_style.apply_axes_style(ax)

    fig.tight_layout()
    pai_style.save_figure(fig, output_path)


def create_param_count_plot(
    stats                  : Sequence[Dict[str, float]],
    param_counts_by_column : Dict[str, float],
    output_path            : str,
    x_break                : Optional[Tuple[float, float]] = None,
    connect_extremes       : bool                          = False,
    column_color_map       : Optional[Dict[str, str]]      = None,
    metric_label           : str                           = 'Val',
) -> None:
    '''
    Draw a candlestick plot with boxes placed by parameter count

    Notes:
        - We size the boxes from the axis width in pixels and cap them so
          neighboring boxes overlap by less than half
        - Columns that share a param_count get small fixed offsets, so
          their boxes do not sit on top of each other
        - With x_break we split the plot into two axes sharing the y axis

    Signature:
        stats (Sequence[Dict[str, float]]):
            - Box statistics from build_box_stats, one entry for each
              column
        param_counts_by_column (Dict[str, float]):
            - Column name to parameter count, from the CSV metadata rows
        output_path (str):
            - Where the PNG goes
        x_break (Optional[Tuple[float, float]]):
            - (start, end) range to cut out of the x axis, or None
        connect_extremes (bool):
            - Whether to join the leftmost and rightmost boxes with lines
        column_color_map (Optional[Dict[str, str]]):
            - Column name to color, from build_column_color_map
        metric_label (str):
            - Val or Test, used in the title and y label
    '''
    if column_color_map is None:
        column_color_map = {}

    def style_bxp_artists(
        artists : Dict[str, List],
        items   : Sequence[Dict[str, float]],
    ) -> None:
        '''
        Color each box, median, whisker and cap to match its column

        Signature:
            artists (Dict[str, List]):
                - What ax.bxp returned
            items (Sequence[Dict[str, float]]):
                - Box statistics in the same order as the artists
        '''
        for i, item in enumerate(items):
            color = column_color_map.get(
                item['column'], pai_style.stream_palette[2]
            )
            artists['boxes'][i].set_facecolor(to_rgba(color, alpha=0.35))
            artists['boxes'][i].set_edgecolor(color)
            artists['boxes'][i].set_linewidth(1.2)

            artists['medians'][i].set_color(color)
            artists['medians'][i].set_linewidth(1.8)

            artists['whiskers'][2 * i].set_color(color)
            artists['whiskers'][2 * i + 1].set_color(color)
            artists['caps'][2 * i].set_color(color)
            artists['caps'][2 * i + 1].set_color(color)

    def add_inside_legend(
        ax    : plt.Axes,
        items : Sequence[Dict[str, float]],
    ) -> None:
        '''
        Add a legend with one square marker for each box

        Signature:
            ax (plt.Axes):
                - Axes the legend attaches to
            items (Sequence[Dict[str, float]]):
                - Box statistics, one legend entry each
        '''
        handles = []
        labels  = []
        for item in items:
            color = column_color_map.get(
                item['column'], pai_style.stream_palette[2]
            )
            handles.append(Line2D(
                [0],
                [0],
                marker          = 's',
                linestyle       = 'none',
                markersize      = 6,
                markerfacecolor = color,
                markeredgecolor = color,
            ))
            labels.append(item['label'])
        if handles:
            ax.legend(
                handles, labels, loc='lower right',
                **pai_style.legend_kwargs,
            )

    stats_with_counts = [
        item for item in stats
        if item['column'] in param_counts_by_column
    ]

    if not stats_with_counts:
        raise ValueError(
            'No parameter-count metadata found for dendrite columns.'
        )

    base_positions = [
        param_counts_by_column[item['column']] for item in stats_with_counts
    ]

    # Columns sharing a param_count get tiny deterministic offsets
    grouped_indices : Dict[float, List[int]] = defaultdict(list)
    for i, pos in enumerate(base_positions):
        grouped_indices[pos].append(i)

    if base_positions:
        x_min  = min(base_positions)
        x_max  = max(base_positions)
        x_span = x_max - x_min
    else:
        x_span = 0.0

    offset_step        = max(1.0, x_span * 0.002)
    adjusted_positions = base_positions[:]

    for _, idxs in grouped_indices.items():
        if len(idxs) <= 1:
            continue
        center = (len(idxs) - 1) / 2.0
        for k, idx in enumerate(idxs):
            adjusted_positions[idx] = (
                base_positions[idx] + (k - center) * offset_step
            )

    def positive_gaps(values: Sequence[float]) -> List[float]:
        '''
        List the gaps between neighboring distinct x positions

        Signature:
            values (Sequence[float]):
                - Box x positions, duplicates allowed
        '''
        unique_vals        = sorted(set(values))
        gaps : List[float] = []
        for i in range(1, len(unique_vals)):
            gap = unique_vals[i] - unique_vals[i - 1]
            if gap <= 0:
                continue
            gaps.append(gap)
        return gaps

    def reference_gap(
        values   : Sequence[float],
        quantile : float           = 0.25,
    ) -> Optional[float]:
        '''
        Pick a quantile of the gaps as the typical box spacing

        Notes:
            - None when there is only one distinct position, so there
              are no gaps to measure

        Signature:
            values (Sequence[float]):
                - Box x positions
            quantile (float):
                - Which quantile of the gaps to return, 0.25 by default
        '''
        gaps = positive_gaps(values)
        if not gaps:
            return None
        return float(pd.Series(gaps).quantile(quantile))

    def compute_width_px(
        default_axis_width_px : float,
        min_gap_px            : Optional[float],
    ) -> float:
        '''
        Pick a box width in pixels that limits overlap

        Notes:
            - overlap_fraction = (width_px - gap_px) / width_px and we
              want it under 0.5, which means width_px < gap_px / 0.5
            - We divide by 0.51 rather than 0.5 so a width of 100 puts
              the left side at 51

        Signature:
            default_axis_width_px (float):
                - Width of the axes in pixels
            min_gap_px (Optional[float]):
                - Typical gap between box centers in pixels, or None
        '''
        default_width_px = max(1.0, default_axis_width_px * 0.015)
        if min_gap_px is None:
            return default_width_px
        cap_width_px = min_gap_px / 0.51
        return max(1.0, min(default_width_px, cap_width_px))

    def enforce_min_center_gap(
        positions    : Sequence[float],
        min_gap_data : float,
    ) -> List[float]:
        '''
        Push box centers right until each pair is min_gap_data apart

        Signature:
            positions (Sequence[float]):
                - Box centers in data units
            min_gap_data (float):
                - Smallest gap we allow between neighbors, in data units
        '''
        adjusted = list(positions)
        sorted_indices = sorted(
            range(len(adjusted)), key=lambda i: adjusted[i]
        )
        prev = None
        for idx in sorted_indices:
            current = adjusted[idx]
            if prev is None:
                prev = current
                continue
            if current - prev < min_gap_data:
                current       = prev + min_gap_data
                adjusted[idx] = current
            prev = current
        return adjusted

    def to_bxp_stats(
        items : Sequence[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        '''
        Keep only the keys ax.bxp accepts from each box statistics dict

        Signature:
            items (Sequence[Dict[str, float]]):
                - Box statistics from build_box_stats
        '''
        # Only the five box keys and the label are passed to bxp
        return [
            {
                'label' : item['label'],
                'whislo': item['whislo'],
                'q1'    : item['q1'],
                'med'   : item['med'],
                'q3'    : item['q3'],
                'whishi': item['whishi'],
            }
            for item in items
        ]

    def draw_extreme_connectors_single(
        ax        : plt.Axes,
        items     : Sequence[Dict[str, float]],
        positions : Sequence[float],
    ) -> None:
        '''
        Join the first and last box at whislo, med and whishi

        Signature:
            ax (plt.Axes):
                - Axes to draw into
            items (Sequence[Dict[str, float]]):
                - Box statistics in position order
            positions (Sequence[float]):
                - Box centers in data units, one for each item
        '''
        if len(items) < 2:
            return
        left_idx  = min(range(len(positions)), key=lambda i: positions[i])
        right_idx = max(range(len(positions)), key=lambda i: positions[i])
        x_left    = positions[left_idx]
        x_right   = positions[right_idx]
        y_keys    = ['whislo', 'med', 'whishi']
        for y_key in y_keys:
            y_left  = items[left_idx][y_key]
            y_right = items[right_idx][y_key]
            ax.plot(
                [x_left, x_right],
                [y_left, y_right],
                color     = pai_style.stream_palette[3],
                linewidth = 1.5,
                alpha     = 0.9,
            )

    def interpolate_y(
        x1 : float,
        y1 : float,
        x2 : float,
        y2 : float,
        x  : float,
    ) -> float:
        '''
        Find y at x on the straight line through two points

        Signature:
            x1 (float):
                - x of the first point
            y1 (float):
                - y of the first point
            x2 (float):
                - x of the second point
            y2 (float):
                - y of the second point
            x (float):
                - Where along that line we want y
        '''
        if x2 == x1:
            return y1
        return y1 + (y2 - y1) * ((x - x1) / (x2 - x1))

    def draw_extreme_connectors_broken(
        ax_left         : plt.Axes,
        ax_right        : plt.Axes,
        left_items      : Sequence[Dict[str, float]],
        left_positions  : Sequence[float],
        right_items     : Sequence[Dict[str, float]],
        right_positions : Sequence[float],
        break_start_val : float,
        break_end_val   : float,
    ) -> None:
        '''
        Join the first and last box with lines across the x-axis break

        Notes:
            - We interpolate each line to the break edges, so it ends at
              the axis boundary on both sides instead of running off

        Signature:
            ax_left (plt.Axes):
                - Axes left of the break
            ax_right (plt.Axes):
                - Axes right of the break
            left_items (Sequence[Dict[str, float]]):
                - Box statistics drawn on ax_left
            left_positions (Sequence[float]):
                - Box centers on ax_left
            right_items (Sequence[Dict[str, float]]):
                - Box statistics drawn on ax_right
            right_positions (Sequence[float]):
                - Box centers on ax_right
            break_start_val (float):
                - x where ax_left ends
            break_end_val (float):
                - x where ax_right starts
        '''
        if not left_items or not right_items:
            return

        left_idx = min(
            range(len(left_positions)), key=lambda i: left_positions[i]
        )
        right_idx = max(
            range(len(right_positions)), key=lambda i: right_positions[i]
        )
        x_left  = left_positions[left_idx]
        x_right = right_positions[right_idx]

        y_keys = ['whislo', 'med', 'whishi']
        for y_key in y_keys:
            y_left  = left_items[left_idx][y_key]
            y_right = right_items[right_idx][y_key]

            y_at_break_start = interpolate_y(
                x_left, y_left, x_right, y_right, break_start_val
            )
            y_at_break_end = interpolate_y(
                x_left, y_left, x_right, y_right, break_end_val
            )

            ax_left.plot(
                [x_left, break_start_val],
                [y_left, y_at_break_start],
                color     = pai_style.stream_palette[3],
                linewidth = 1.5,
                alpha     = 0.9,
            )
            ax_right.plot(
                [break_end_val, x_right],
                [y_at_break_end, y_right],
                color     = pai_style.stream_palette[3],
                linewidth = 1.5,
                alpha     = 0.9,
            )

    if x_break is None:
        fig, ax = plt.subplots(figsize=(12, 6))

        x_pad    = max(1.0, x_span * 0.08)
        xlim_min = min(adjusted_positions) - x_pad
        xlim_max = max(adjusted_positions) + x_pad
        ax.set_xlim(xlim_min, xlim_max)

        fig.canvas.draw()
        axis_width_px = ax.get_window_extent().width
        axis_span     = xlim_max - xlim_min
        px_per_data   = axis_width_px / axis_span if axis_span > 0 else 0.0

        ref_gap_data = reference_gap(base_positions, quantile=0.25)
        ref_gap_px   = None
        if ref_gap_data is not None and px_per_data > 0:
            ref_gap_px = ref_gap_data * px_per_data
        width_px = compute_width_px(axis_width_px, ref_gap_px)
        if px_per_data > 0:
            width = width_px / px_per_data
        else:
            width = max(1.0, x_span * 0.015)

        if px_per_data > 0:
            required_gap_data = (width_px * 0.51) / px_per_data
            plot_positions    = enforce_min_center_gap(
                adjusted_positions, required_gap_data
            )
        else:
            plot_positions = adjusted_positions

        artists = ax.bxp(
            to_bxp_stats(stats_with_counts),
            positions    = plot_positions,
            widths       = width,
            showfliers   = False,
            manage_ticks = False,
            patch_artist = True,
        )
        style_bxp_artists(artists, stats_with_counts)

        ax.set_title(f'Dendrite {metric_label} Distribution by Parameter Count')
        ax.set_xlabel('Parameter Count')
        ax.set_ylabel(f'Max {metric_label}')
        pai_style.apply_axes_style(ax)

        add_inside_legend(ax, stats_with_counts)

        if connect_extremes:
            draw_extreme_connectors_single(
                ax, stats_with_counts, plot_positions
            )

        fig.tight_layout()
        pai_style.save_figure(fig, output_path)
        return

    break_start, break_end = x_break
    if break_start >= break_end:
        raise ValueError('x-break must have start < end.')

    left_items      : List[Dict[str, float]] = []
    left_positions  : List[float]            = []
    right_items     : List[Dict[str, float]] = []
    right_positions : List[float]            = []

    for item, base_pos, adjusted_pos in zip(
        stats_with_counts, base_positions, adjusted_positions
    ):
        if base_pos <= break_start:
            left_items.append(item)
            left_positions.append(adjusted_pos)
        elif base_pos >= break_end:
            right_items.append(item)
            right_positions.append(adjusted_pos)

    if not left_items or not right_items:
        data_min           = min(base_positions)
        data_max           = max(base_positions)
        distinct_positions = sorted(set(base_positions))
        formatted_positions = ', '.join(
            f'{int(v):,}' if float(v).is_integer() else f'{v:,.3f}'
            for v in distinct_positions
        )
        raise ValueError(
            'x-break range removes one side of the chart. '
            f'Data param_count range is [{data_min:,.0f}, {data_max:,.0f}] '
            f'with values: {formatted_positions}. '
            'Choose START/END so there are points <= START and >= END.'
        )

    left_base_positions = [
        param_counts_by_column[item['column']]
        for item in left_items
    ]
    right_base_positions = [
        param_counts_by_column[item['column']]
        for item in right_items
    ]

    # Shared tight padding from the nearest left value to the break start
    shared_pad = break_start - max(left_base_positions)
    if shared_pad <= 0:
        shared_pad = max(1.0, x_span * 0.01)

    left_xlim_min  = min(left_base_positions) - shared_pad
    left_xlim_max  = break_start
    right_xlim_min = break_end
    right_xlim_max = max(right_base_positions) + shared_pad

    left_span  = left_xlim_max - left_xlim_min
    right_span = right_xlim_max - right_xlim_min
    left_span  = left_span if left_span > 0 else 1.0
    right_span = right_span if right_span > 0 else 1.0

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey      = True,
        figsize     = (14, 6),
        gridspec_kw = {'width_ratios': [left_span, right_span]},
    )

    ax_left.set_xlim(left_xlim_min, left_xlim_max)
    ax_right.set_xlim(right_xlim_min, right_xlim_max)

    fig.canvas.draw()
    left_axis_width_px  = ax_left.get_window_extent().width
    right_axis_width_px = ax_right.get_window_extent().width
    left_axis_span      = left_xlim_max - left_xlim_min
    right_axis_span     = right_xlim_max - right_xlim_min
    left_px_per_data    = 0.0
    right_px_per_data   = 0.0
    if left_axis_span > 0:
        left_px_per_data = left_axis_width_px / left_axis_span
    if right_axis_span > 0:
        right_px_per_data = right_axis_width_px / right_axis_span

    left_ref_gap_data  = reference_gap(left_base_positions, quantile=0.25)
    right_ref_gap_data = reference_gap(right_base_positions, quantile=0.25)

    gap_candidates_px : List[float] = []
    if left_ref_gap_data is not None and left_px_per_data > 0:
        gap_candidates_px.append(left_ref_gap_data * left_px_per_data)
    if right_ref_gap_data is not None and right_px_per_data > 0:
        gap_candidates_px.append(right_ref_gap_data * right_px_per_data)
    ref_gap_px = min(gap_candidates_px) if gap_candidates_px else None

    total_axis_width_px = left_axis_width_px + right_axis_width_px
    global_width_px     = compute_width_px(total_axis_width_px, ref_gap_px)
    left_width          = 1.0
    right_width         = 1.0
    if left_px_per_data > 0:
        left_width = global_width_px / left_px_per_data
    if right_px_per_data > 0:
        right_width = global_width_px / right_px_per_data

    if left_px_per_data > 0:
        left_required_gap_data = (global_width_px * 0.51) / left_px_per_data
        left_plot_positions    = enforce_min_center_gap(
            left_positions, left_required_gap_data
        )
    else:
        left_plot_positions = left_positions

    if right_px_per_data > 0:
        right_required_gap_data = (global_width_px * 0.51) / right_px_per_data
        right_plot_positions    = enforce_min_center_gap(
            right_positions, right_required_gap_data
        )
    else:
        right_plot_positions = right_positions

    left_artists = ax_left.bxp(
        to_bxp_stats(left_items),
        positions    = left_plot_positions,
        widths       = left_width,
        showfliers   = False,
        manage_ticks = False,
        patch_artist = True,
    )
    right_artists = ax_right.bxp(
        to_bxp_stats(right_items),
        positions    = right_plot_positions,
        widths       = right_width,
        showfliers   = False,
        manage_ticks = False,
        patch_artist = True,
    )
    style_bxp_artists(left_artists, left_items)
    style_bxp_artists(right_artists, right_items)

    ax_left.set_title(
        f'Dendrite {metric_label} Distribution by Parameter Count '
        '(Broken X-Axis)'
    )
    ax_left.set_xlabel('Parameter Count')
    ax_right.set_xlabel('Parameter Count')
    ax_left.set_ylabel(f'Max {metric_label}')
    pai_style.apply_axes_style(ax_left)
    pai_style.apply_axes_style(ax_right)

    add_inside_legend(ax_right, stats_with_counts)

    if connect_extremes:
        draw_extreme_connectors_broken(
            ax_left,
            ax_right,
            left_items,
            left_plot_positions,
            right_items,
            right_plot_positions,
            break_start,
            break_end,
        )

    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(labelright=False)

    marker_kwargs = dict(
        marker     = [(-1, -1), (1, 1)],
        markersize = 8,
        linestyle  = 'none',
        color      = 'k',
        mec        = 'k',
        mew        = 1,
        clip_on    = False,
    )
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **marker_kwargs)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **marker_kwargs)

    fig.tight_layout()
    pai_style.save_figure(fig, output_path)


def create_param_count_scatter_plot(
    df                     : pd.DataFrame,
    dendrite_columns       : Sequence[str],
    param_counts_by_column : Dict[str, float],
    output_path            : str,
    x_break                : Optional[Tuple[float, float]] = None,
    model_name_map         : Optional[Dict[str, str]]      = None,
    column_color_map       : Optional[Dict[str, str]]      = None,
    metric_label           : str                           = 'Val',
) -> None:
    '''
    Scatter every raw value of each column against its parameter count

    Signature:
        df (pd.DataFrame):
            - Data rows from read_by_dendrite_separate_csv
        dendrite_columns (Sequence[str]):
            - Columns to scatter
        param_counts_by_column (Dict[str, float]):
            - Column name to parameter count, from the CSV metadata rows
        output_path (str):
            - Where the PNG goes
        x_break (Optional[Tuple[float, float]]):
            - (start, end) range to cut out of the x axis, or None
        model_name_map (Optional[Dict[str, str]]):
            - Model id to display name, passed on to display_label
        column_color_map (Optional[Dict[str, str]]):
            - Column name to color, from build_column_color_map
        metric_label (str):
            - Val or Test, used in the title and y label
    '''
    column_points : List[Dict[str, object]] = []
    for col in dendrite_columns:
        if col not in param_counts_by_column:
            continue
        y_vals = df[col].dropna().tolist()
        if not y_vals:
            continue
        column_points.append({
            'column': col,
            'label' : display_label(col, model_name_map),
            'x'     : float(param_counts_by_column[col]),
            'y'     : y_vals,
        })

    if not column_points:
        raise ValueError('No non-empty dendrite metric data found to scatter.')

    if column_color_map is None:
        column_color_map = {}

    x_values = [item['x'] for item in column_points]
    x_min    = min(x_values)
    x_max    = max(x_values)
    x_span   = x_max - x_min

    if x_break is None:
        fig, ax = plt.subplots(figsize=(12, 6))

        x_pad = max(1.0, x_span * 0.08)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

        for item in column_points:
            color = column_color_map.get(
                item['column'], pai_style.stream_palette[2]
            )
            x      = item['x']
            y_vals = item['y']
            ax.scatter(
                [x] * len(y_vals),
                y_vals,
                s     = 12,
                color = color,
                alpha = 0.75,
                label = item['label'],
            )

        ax.set_title(f'Dendrite {metric_label} Scatter by Parameter Count')
        ax.set_xlabel('Parameter Count')
        ax.set_ylabel(f'Max {metric_label}')
        pai_style.apply_axes_style(ax)
        ax.legend(loc='best', **pai_style.legend_kwargs)

        fig.tight_layout()
        pai_style.save_figure(fig, output_path)
        return

    break_start, break_end = x_break
    if break_start >= break_end:
        raise ValueError('x-break must have start < end.')

    left_points  = [item for item in column_points if item['x'] <= break_start]
    right_points = [item for item in column_points if item['x'] >= break_end]

    if not left_points or not right_points:
        distinct_positions = sorted(set(x_values))
        formatted_positions = ', '.join(
            f'{int(v):,}' if float(v).is_integer() else f'{v:,.3f}'
            for v in distinct_positions
        )
        raise ValueError(
            'x-break range removes one side of the scatter chart. '
            f'Data param_count values: {formatted_positions}. '
            'Choose START/END so there are points <= START and >= END.'
        )

    left_x_vals  = [item['x'] for item in left_points]
    right_x_vals = [item['x'] for item in right_points]

    shared_pad = break_start - max(left_x_vals)
    if shared_pad <= 0:
        shared_pad = max(1.0, x_span * 0.01)

    left_xlim_min  = min(left_x_vals) - shared_pad
    left_xlim_max  = break_start
    right_xlim_min = break_end
    right_xlim_max = max(right_x_vals) + shared_pad

    left_span  = left_xlim_max - left_xlim_min
    right_span = right_xlim_max - right_xlim_min
    left_span  = left_span if left_span > 0 else 1.0
    right_span = right_span if right_span > 0 else 1.0

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey      = True,
        figsize     = (14, 6),
        gridspec_kw = {'width_ratios': [left_span, right_span]},
    )

    ax_left.set_xlim(left_xlim_min, left_xlim_max)
    ax_right.set_xlim(right_xlim_min, right_xlim_max)

    for item in column_points:
        color = column_color_map.get(
            item['column'], pai_style.stream_palette[2]
        )
        x      = item['x']
        y_vals = item['y']
        if x <= break_start:
            ax_left.scatter(
                [x] * len(y_vals),
                y_vals,
                s     = 12,
                color = color,
                alpha = 0.75,
                label = item['label'],
            )
        elif x >= break_end:
            ax_right.scatter(
                [x] * len(y_vals),
                y_vals,
                s     = 12,
                color = color,
                alpha = 0.75,
                label = item['label'],
            )

    ax_left.set_title(
        f'Dendrite {metric_label} Scatter by Parameter Count (Broken X-Axis)'
    )
    ax_left.set_xlabel('Parameter Count')
    ax_right.set_xlabel('Parameter Count')
    ax_left.set_ylabel(f'Max {metric_label}')
    pai_style.apply_axes_style(ax_left)
    pai_style.apply_axes_style(ax_right)

    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(labelright=False)

    marker_kwargs = dict(
        marker     = [(-1, -1), (1, 1)],
        markersize = 8,
        linestyle  = 'none',
        color      = 'k',
        mec        = 'k',
        mew        = 1,
        clip_on    = False,
    )
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **marker_kwargs)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **marker_kwargs)

    handles, labels = ax_left.get_legend_handles_labels()
    if handles:
        ax_left.legend(handles, labels, loc='best', **pai_style.legend_kwargs)

    fig.tight_layout()
    pai_style.save_figure(fig, output_path)


def create_categorical_summary_plot(
    stats            : Sequence[Dict[str, float]],
    output_path      : str,
    value_key        : str,
    value_label      : str,
    column_color_map : Optional[Dict[str, str]]   = None,
    metric_label     : str                        = 'Val',
) -> None:
    '''
    Draw a bar chart of one statistic for each column

    Signature:
        stats (Sequence[Dict[str, float]]):
            - Box statistics from build_box_stats, one entry for each
              column
        output_path (str):
            - Where the PNG goes
        value_key (str):
            - Which statistic to plot, mean or max
        value_label (str):
            - Average or Max, how the title and y label name it
        column_color_map (Optional[Dict[str, str]]):
            - Column name to color, from build_column_color_map
        metric_label (str):
            - Val or Test, used in the title and y label
    '''
    if column_color_map is None:
        column_color_map = {}

    labels = [str(item['label']) for item in stats]
    values = [float(item[value_key]) for item in stats]
    colors = [
        column_color_map.get(item['column'], pai_style.stream_palette[2])
        for item in stats
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(stats) * 0.45), 6))
    ax.bar(labels, values, color=colors, alpha=0.75)
    ax.set_title(f'Dendrite {metric_label} {value_label} by Dendrite Count')
    ax.set_xlabel('Dendrite Count')
    ax.set_ylabel(f'{value_label} {metric_label}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    pai_style.apply_axes_style(ax)
    fig.tight_layout()
    pai_style.save_figure(fig, output_path)


def build_top_half_stats(
    stats : Sequence[Dict[str, float]],
) -> List[Dict[str, float]]:
    '''
    Rebuild the box statistics using only each column's top half

    Notes:
        - The top_ keys from build_box_stats overwrite the regular keys,
          so the result plots values at or above the median

    Signature:
        stats (Sequence[Dict[str, float]]):
            - Box statistics from build_box_stats, one entry for each
              column
    '''
    top_stats : List[Dict[str, float]] = []
    for item in stats:
        top_stats.append({
            **item,
            'whislo': float(item.get('top_whislo', item['med'])),
            'q1'    : float(item.get('top_q1', item['med'])),
            'med'   : float(item.get('top_med', item['med'])),
            'q3'    : float(item.get('top_q3', item['q3'])),
            'whishi': float(item.get('top_whishi', item['whishi'])),
            'mean'  : float(
                item.get('top_mean', item.get('mean', item['med']))
            ),
            'max'   : float(
                item.get('top_max', item.get('max', item['whishi']))
            ),
        })
    return top_stats


def create_param_count_summary_plot(
    stats                  : Sequence[Dict[str, float]],
    param_counts_by_column : Dict[str, float],
    output_path            : str,
    value_key              : str,
    value_label            : str,
    x_break                : Optional[Tuple[float, float]] = None,
    column_color_map       : Optional[Dict[str, str]]      = None,
    metric_label           : str                           = 'Val',
) -> None:
    '''
    Plot one statistic against parameter count as points joined by a line

    Signature:
        stats (Sequence[Dict[str, float]]):
            - Box statistics from build_box_stats, one entry for each
              column
        param_counts_by_column (Dict[str, float]):
            - Column name to parameter count, from the CSV metadata rows
        output_path (str):
            - Where the PNG goes
        value_key (str):
            - Which statistic to plot, mean or max
        value_label (str):
            - Average or Max, how the title and y label name it
        x_break (Optional[Tuple[float, float]]):
            - (start, end) range to cut out of the x axis, or None
        column_color_map (Optional[Dict[str, str]]):
            - Column name to color, from build_column_color_map
        metric_label (str):
            - Val or Test, used in the title and y label
    '''
    stats_with_counts = [
        item for item in stats if item['column'] in param_counts_by_column
    ]
    if not stats_with_counts:
        raise ValueError('No parameter-count metadata found for summary plot.')

    if column_color_map is None:
        column_color_map = {}

    points = []
    for item in stats_with_counts:
        points.append({
            'x'     : float(param_counts_by_column[item['column']]),
            'y'     : float(item[value_key]),
            'label' : str(item['label']),
            'column': str(item['column']),
        })

    points   = sorted(points, key=lambda p: p['x'])
    x_values = [p['x'] for p in points]
    y_values = [p['y'] for p in points]
    x_min    = min(x_values)
    x_max    = max(x_values)
    x_span   = x_max - x_min

    def draw_points(ax: plt.Axes, pts: Sequence[Dict[str, Any]]) -> None:
        '''
        Scatter the points and join them left to right with a line

        Signature:
            ax (plt.Axes):
                - Axes to draw into
            pts (Sequence[Dict[str, Any]]):
                - Points already sorted by x
        '''
        for p in pts:
            color = column_color_map.get(
                p['column'], pai_style.stream_palette[2]
            )
            ax.scatter([p['x']], [p['y']], color=color, s=30, alpha=0.9)
        if len(pts) >= 2:
            ax.plot(
                [p['x'] for p in pts],
                [p['y'] for p in pts],
                color     = pai_style.stream_palette[3],
                linewidth = 1.5,
                alpha     = 0.8,
            )

    def add_legend(ax: plt.Axes, pts: Sequence[Dict[str, Any]]) -> None:
        '''
        Add a legend with one marker for each distinct column

        Signature:
            ax (plt.Axes):
                - Axes the legend attaches to
            pts (Sequence[Dict[str, Any]]):
                - Points whose columns the legend lists
        '''
        handles : List[Line2D] = []
        labels  : List[str]    = []
        seen                   = set()
        for p in pts:
            col = p['column']
            if col in seen:
                continue
            seen.add(col)
            color = column_color_map.get(col, pai_style.stream_palette[2])
            handles.append(Line2D(
                [0],
                [0],
                marker          = 'o',
                linestyle       = 'none',
                markersize      = 5,
                markerfacecolor = color,
                markeredgecolor = color,
            ))
            labels.append(str(p['label']))
        if handles:
            ax.legend(
                handles, labels, loc='lower right',
                **pai_style.legend_kwargs,
            )

    if x_break is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pad   = max(1.0, x_span * 0.08)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        draw_points(ax, points)
        add_legend(ax, points)
        ax.set_title(
            f'Dendrite {metric_label} {value_label} by Parameter Count'
        )
        ax.set_xlabel('Parameter Count')
        ax.set_ylabel(f'{value_label} {metric_label}')
        pai_style.apply_axes_style(ax)
        fig.tight_layout()
        pai_style.save_figure(fig, output_path)
        return

    break_start, break_end = x_break
    if break_start >= break_end:
        raise ValueError('x-break must have start < end.')

    left_points  = [p for p in points if p['x'] <= break_start]
    right_points = [p for p in points if p['x'] >= break_end]
    if not left_points or not right_points:
        raise ValueError('x-break range removes one side of the summary chart.')

    left_x_vals  = [p['x'] for p in left_points]
    right_x_vals = [p['x'] for p in right_points]
    shared_pad   = break_start - max(left_x_vals)
    if shared_pad <= 0:
        shared_pad = max(1.0, x_span * 0.01)

    left_xlim_min  = min(left_x_vals) - shared_pad
    left_xlim_max  = break_start
    right_xlim_min = break_end
    right_xlim_max = max(right_x_vals) + shared_pad

    left_span  = max(1.0, left_xlim_max - left_xlim_min)
    right_span = max(1.0, right_xlim_max - right_xlim_min)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey      = True,
        figsize     = (14, 6),
        gridspec_kw = {'width_ratios': [left_span, right_span]},
    )

    ax_left.set_xlim(left_xlim_min, left_xlim_max)
    ax_right.set_xlim(right_xlim_min, right_xlim_max)
    draw_points(ax_left, left_points)
    draw_points(ax_right, right_points)
    add_legend(ax_right, points)

    ax_left.set_title(
        f'Dendrite {metric_label} {value_label} by Parameter Count '
        '(Broken X-Axis)'
    )
    ax_left.set_xlabel('Parameter Count')
    ax_right.set_xlabel('Parameter Count')
    ax_left.set_ylabel(f'{value_label} {metric_label}')
    pai_style.apply_axes_style(ax_left)
    pai_style.apply_axes_style(ax_right)

    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(labelright=False)

    marker_kwargs = dict(
        marker     = [(-1, -1), (1, 1)],
        markersize = 8,
        linestyle  = 'none',
        color      = 'k',
        mec        = 'k',
        mew        = 1,
        clip_on    = False,
    )
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **marker_kwargs)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **marker_kwargs)

    fig.tight_layout()
    pai_style.save_figure(fig, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Process a by-dendrite-separate CSV and generate '
                      'candlestick summary graphs.',
    )
    parser.add_argument(
        '--csv',
        required = True,
        help     = 'Path to by-dendrite-separate CSV file',
    )
    parser.add_argument(
        '--output',
        type    = str,
        default = '',
        help    = 'Output directory path. If omitted, uses a folder named '
                  'after the input CSV stem.',
    )
    parser.add_argument(
        '--x-break',
        type    = str,
        default = '',
        help    = 'Optional x-axis break range for param-count plot, '
                  'format: START,END (e.g., 13000000,22000000)',
    )
    parser.add_argument(
        '--filter-key',
        type    = str,
        default = '',
        help    = 'Optional split key for grouped plotting. Can be a CSV '
                  'column or run_name hyperparameter key (for example: '
                  'data_percent).',
    )
    parser.add_argument(
        '--filter-value',
        type    = str,
        default = '',
        help    = 'Optional group values to include. Comma-separated values '
                  'are OR (for example: 12,25).',
    )
    args = parser.parse_args()

    x_break : Optional[Tuple[float, float]] = None
    if args.x_break:
        parts = [p.strip() for p in args.x_break.split(',')]
        if len(parts) != 2:
            print(
                'Error: --x-break must be in format START,END',
                file = sys.stderr,
            )
            sys.exit(1)
        try:
            break_start = float(parts[0])
            break_end   = float(parts[1])
        except ValueError:
            print('Error: --x-break values must be numeric', file=sys.stderr)
            sys.exit(1)
        if break_start >= break_end:
            print('Error: --x-break requires START < END', file=sys.stderr)
            sys.exit(1)
        x_break = (break_start, break_end)

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f'Error: CSV file not found: {csv_path}', file=sys.stderr)
        sys.exit(1)

    base_dir   = os.path.dirname(os.path.abspath(csv_path))
    csv_stem   = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = args.output
    if not output_dir:
        output_dir = os.path.join(base_dir, csv_stem)
    os.makedirs(output_dir, exist_ok=True)

    try:
        df, dendrite_columns, param_counts_by_column = (
            read_by_dendrite_separate_csv(csv_path)
        )

        if bool(args.filter_key) != bool(args.filter_value):
            raise ValueError('--filter-value requires --filter-key.')

        group_key            = (args.filter_key or '').strip()
        include_group_values = parse_group_values(args.filter_value)

        group_series : Optional[pd.Series] = None
        if group_key:
            group_series = extract_group_series(df, group_key)

        val_columns  = [c for c in dendrite_columns if c.endswith('_max_val')]
        test_columns = [c for c in dendrite_columns if c.endswith('_max_test')]

        metric_runs : List[Tuple[str, str, List[str]]] = []
        if val_columns:
            metric_runs.append(('val', 'Val', val_columns))
        if test_columns:
            metric_runs.append(('test', 'Test', test_columns))

        if not metric_runs:
            raise ValueError(
                'No val/test dendrite columns found. Expected columns like '
                'model_0_dendrite_2_max_val or model_0_dendrite_2_max_test.'
            )

        created_files : List[str] = []

        for metric_key, metric_label, metric_columns in metric_runs:
            suffix = f'_{metric_key}'
            if group_series is not None:
                stats, grouped_param_counts = (
                    build_grouped_box_stats_by_dendrite(
                        df,
                        metric_columns,
                        group_series,
                        param_counts_by_column,
                        include_groups = include_group_values,
                    )
                )
            else:
                stats                = build_box_stats_by_dendrite(
                    df, metric_columns
                )
                grouped_param_counts = aggregate_param_counts_by_dendrite(
                    metric_columns, param_counts_by_column
                )

            # Consistent color per dendrite count after grouping
            column_color_map = {
                s['column']: pai_style.stream_color(i, len(stats))
                for i, s in enumerate(stats)
            }

            categorical_plot_path = os.path.join(
                output_dir, f'candlestick_by_dendrite{suffix}.png'
            )
            param_count_plot_path = os.path.join(
                output_dir, f'candlestick_by_param_count{suffix}.png'
            )
            param_count_with_lines_plot_path = os.path.join(
                output_dir,
                f'candlestick_by_param_count_with_lines{suffix}.png',
            )
            top_categorical_plot_path = os.path.join(
                output_dir, f'candlestick_top50_by_dendrite{suffix}.png'
            )
            top_param_count_plot_path = os.path.join(
                output_dir, f'candlestick_top50_by_param_count{suffix}.png'
            )
            top_param_count_with_lines_plot_path = os.path.join(
                output_dir,
                f'candlestick_top50_by_param_count_with_lines{suffix}.png',
            )
            avg_by_dendrite_plot_path = os.path.join(
                output_dir, f'average_by_dendrite{suffix}.png'
            )
            avg_by_param_count_plot_path = os.path.join(
                output_dir, f'average_by_param_count{suffix}.png'
            )
            max_by_dendrite_plot_path = os.path.join(
                output_dir, f'max_by_dendrite{suffix}.png'
            )
            max_by_param_count_plot_path = os.path.join(
                output_dir, f'max_by_param_count{suffix}.png'
            )

            create_categorical_plot(
                stats,
                categorical_plot_path,
                column_color_map = column_color_map,
                metric_label     = metric_label,
            )
            stats_df           = pd.DataFrame(stats)
            stats_df['metric'] = metric_key
            stats_df['param_count'] = stats_df['column'].map(
                grouped_param_counts
            )
            created_files.append(
                write_companion_csv_for_png(categorical_plot_path, stats_df)
            )
            create_param_count_plot(
                stats,
                grouped_param_counts,
                param_count_plot_path,
                x_break          = x_break,
                column_color_map = column_color_map,
                metric_label     = metric_label,
            )
            created_files.append(
                write_companion_csv_for_png(param_count_plot_path, stats_df)
            )
            create_param_count_plot(
                stats,
                grouped_param_counts,
                param_count_with_lines_plot_path,
                x_break          = x_break,
                connect_extremes = True,
                column_color_map = column_color_map,
                metric_label     = metric_label,
            )
            created_files.append(write_companion_csv_for_png(
                param_count_with_lines_plot_path, stats_df
            ))

            top_stats              = build_top_half_stats(stats)
            top_stats_df           = pd.DataFrame(top_stats)
            top_stats_df['metric'] = metric_key
            top_stats_df['param_count'] = top_stats_df['column'].map(
                grouped_param_counts
            )

            create_categorical_plot(
                top_stats,
                top_categorical_plot_path,
                column_color_map = column_color_map,
                metric_label     = f'{metric_label} (Top 50%)',
            )
            created_files.append(write_companion_csv_for_png(
                top_categorical_plot_path, top_stats_df
            ))

            create_param_count_plot(
                top_stats,
                grouped_param_counts,
                top_param_count_plot_path,
                x_break          = x_break,
                column_color_map = column_color_map,
                metric_label     = f'{metric_label} (Top 50%)',
            )
            created_files.append(write_companion_csv_for_png(
                top_param_count_plot_path, top_stats_df
            ))

            create_param_count_plot(
                top_stats,
                grouped_param_counts,
                top_param_count_with_lines_plot_path,
                x_break          = x_break,
                connect_extremes = True,
                column_color_map = column_color_map,
                metric_label     = f'{metric_label} (Top 50%)',
            )
            created_files.append(write_companion_csv_for_png(
                top_param_count_with_lines_plot_path, top_stats_df
            ))

            create_categorical_summary_plot(
                stats,
                avg_by_dendrite_plot_path,
                value_key        = 'mean',
                value_label      = 'Average',
                column_color_map = column_color_map,
                metric_label     = metric_label,
            )
            created_files.append(write_companion_csv_for_png(
                avg_by_dendrite_plot_path, stats_df
            ))

            create_param_count_summary_plot(
                stats,
                grouped_param_counts,
                avg_by_param_count_plot_path,
                value_key        = 'mean',
                value_label      = 'Average',
                x_break          = x_break,
                column_color_map = column_color_map,
                metric_label     = metric_label,
            )
            created_files.append(write_companion_csv_for_png(
                avg_by_param_count_plot_path, stats_df
            ))

            create_categorical_summary_plot(
                stats,
                max_by_dendrite_plot_path,
                value_key        = 'max',
                value_label      = 'Max',
                column_color_map = column_color_map,
                metric_label     = metric_label,
            )
            created_files.append(write_companion_csv_for_png(
                max_by_dendrite_plot_path, stats_df
            ))

            create_param_count_summary_plot(
                stats,
                grouped_param_counts,
                max_by_param_count_plot_path,
                value_key        = 'max',
                value_label      = 'Max',
                x_break          = x_break,
                column_color_map = column_color_map,
                metric_label     = metric_label,
            )
            created_files.append(write_companion_csv_for_png(
                max_by_param_count_plot_path, stats_df
            ))

            created_files.extend([
                categorical_plot_path,
                param_count_plot_path,
                param_count_with_lines_plot_path,
                top_categorical_plot_path,
                top_param_count_plot_path,
                top_param_count_with_lines_plot_path,
                avg_by_dendrite_plot_path,
                avg_by_param_count_plot_path,
                max_by_dendrite_plot_path,
                max_by_param_count_plot_path,
            ])

        if args.filter_key and args.filter_value:
            print(
                f"Grouped by '{args.filter_key}' with included values: "
                f'{args.filter_value}'
            )
        elif args.filter_key:
            print(f"Grouped by '{args.filter_key}'")
        print(f'Rows analyzed: {len(df)}')
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    print(f'Output directory: {output_dir}')
    for f in created_files:
        print(f'Created: {os.path.basename(f)}')
