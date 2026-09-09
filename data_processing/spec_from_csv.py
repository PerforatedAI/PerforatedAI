################################################################################
# Build a plot_streams.py spec from a sweep CSV or from PAI run folders.       #
################################################################################

#
"""
Imports
"""
import argparse
import csv
import glob
import json
import os.path as osp
import re

import pandas as pd

from typing import Any, Dict, List, Optional, Tuple

#
"""
Config
"""
# Column pattern of a by-dendrite-separate CSV, model prefix optional
column_pattern = re.compile(
    r'(?:model_(\d+)_)?dendrite_(\d+)_max_(val|test)$'
)

# Streams with more points than this get annotations turned off
max_annotated_points = 8

# Columns PAI writes into <save_name>_best_arch_scores.csv
param_column   = 'Param Counts'
default_metric = 'Max Valid Scores'

#
"""
Functions
"""
def read_sweep_csv(path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    '''
    Read a by-dendrite-separate CSV and pull out its param counts

    Notes:
        - The first two rows are metadata labels and values, the third
          is the header, and the rest is data
        - We only turn the dendrite metric columns into numbers

    Signature:
        path (str):
            - The CSV that get_wandb_results.py wrote
    '''
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise ValueError(f'{path} does not have the 3 row metadata layout')

    header = rows[2]
    width  = len(header)
    data   = [r[:width] + [''] * (width - len(r)) for r in rows[3:]]
    df     = pd.DataFrame(data, columns=header)

    param_counts: Dict[str, float] = {}
    values = rows[1] + [''] * (width - len(rows[1]))
    for col, raw in zip(header, values):
        if column_pattern.match(col):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if raw.strip():
                param_counts[col] = float(raw)
    return df, param_counts


def read_model_names(csv_path: str) -> Dict[str, str]:
    '''
    Look up model names in the model_info.csv next to the sweep CSV

    Signature:
        csv_path (str):
            - The sweep CSV, we look for model_info.csv in its folder
    '''
    info_path = osp.join(osp.dirname(osp.abspath(csv_path)), 'model_info.csv')
    if not osp.exists(info_path):
        return {}
    info = pd.read_csv(info_path)
    if not {'model_id', 'model_name'}.issubset(info.columns):
        return {}
    # Model id to display name, skipping blank rows
    return {
        str(r['model_id']).strip(): str(r['model_name']).strip()
        for _, r in info.iterrows()
        if str(r['model_id']).strip() and str(r['model_name']).strip()
    }


def streams_from_sweep(
    csv_path : str,
    stat     : str,
    metric   : str,
) -> List[Dict[str, Any]]:
    '''
    Build a stream for each model, one point per dendrite column

    Notes:
        - Each point's y is the stat over every run in that column. With
          stat max and 50 runs, that is the best run at each dendrite
          count
        - Columns with no param count in the metadata row are skipped

    Signature:
        csv_path (str):
            - The by-dendrite-separate CSV
        stat (str):
            - How to collapse the runs in a column, max, mean, or median
        metric (str):
            - Which columns to read, val or test
    '''
    df, param_counts = read_sweep_csv(csv_path)
    names            = read_model_names(csv_path)

    per_model: Dict[str, List[Tuple[int, float, float]]] = {}
    for col in df.columns:
        match = column_pattern.match(col)
        if not match or match.group(3) != metric or col not in param_counts:
            continue
        model_id = f'model_{match.group(1) or 0}'
        series   = df[col].dropna()
        if series.empty:
            continue
        y = float(getattr(series, stat)())
        per_model.setdefault(model_id, []).append(
            (int(match.group(2)), param_counts[col], y))

    if not per_model:
        raise ValueError(f'no {metric} dendrite columns with param counts')

    streams = []
    for model_id in sorted(per_model, key=lambda m: int(m.split('_')[1])):
        points = sorted(per_model[model_id])
        streams.append(dict(
            name   = names.get(model_id, model_id),
            style  = 'line',
            points = [[x, y] for _, x, y in points],
            source = f'{osp.basename(csv_path)} {stat} of {model_id} '
                     f'{metric} columns',
        ))
    return streams


def read_run_folder(folder: str, metric: str) -> pd.DataFrame:
    '''
    Read the best_arch_scores.csv from one PAI run folder

    Notes:
        - Rows with no score are architectures that never finished, so
          we drop them

    Signature:
        folder (str):
            - A run folder holding <save_name>_best_arch_scores.csv
        metric (str):
            - The column of best_arch_scores.csv we are plotting
    '''
    matches = glob.glob(osp.join(folder, '*_best_arch_scores.csv'))
    if len(matches) != 1:
        raise ValueError(
            f'{folder} needs exactly one *_best_arch_scores.csv, '
            f'found {len(matches)}')
    scores = pd.read_csv(matches[0])
    if param_column not in scores.columns or metric not in scores.columns:
        raise ValueError(
            f'{matches[0]} needs columns {param_column!r} and {metric!r}')
    # Keep only the architectures that produced a score
    return scores.dropna(subset=[metric])


def streams_from_runs(
    specs         : List[str],
    metric        : str,
    best          : str,
    all_dendrites : bool,
) -> List[Dict[str, Any]]:
    '''
    Build a stream for each --stream argument from its run folders

    Notes:
        - By default each folder contributes its best row, so a stream
          of four model sizes has four points
        - With all_dendrites every row of every folder becomes a point,
          labeled by its dendrite count

    Signature:
        specs (List[str]):
            - The NAME:folder1,folder2 strings from the command line
        metric (str):
            - The column of best_arch_scores.csv we are plotting
        best (str):
            - Whether the highest or lowest row counts as best, max or min
        all_dendrites (bool):
            - Take every architecture row instead of just the best one
    '''
    streams = []
    for raw in specs:
        if ':' not in raw:
            raise ValueError(f'--stream {raw!r} must look like NAME:folder')
        name, folders = raw.split(':', 1)
        points  = []
        sources = []
        for folder in folders.split(','):
            scores = read_run_folder(folder, metric)
            if all_dendrites:
                for i, row in scores.iterrows():
                    points.append(dict(
                        x     = float(row[param_column]),
                        y     = float(row[metric]),
                        label = f'{i}d',
                    ))
            else:
                idx = scores[metric].idxmax() if best == 'max' \
                    else scores[metric].idxmin()
                row = scores.loc[idx]
                points.append([float(row[param_column]), float(row[metric])])
            sources.append(osp.basename(osp.normpath(folder)))
        streams.append(dict(
            name   = name,
            style  = 'line',
            points = points,
            source = f'{best} {metric!r} from ' + ', '.join(sources),
        ))
    return streams


def build_spec(
    out     : str,
    title   : str,
    y_label : str,
    y_fmt   : str,
    streams : List[Dict[str, Any]],
) -> Dict[str, Any]:
    '''
    Put together the spec dict that plot_streams.py reads

    Notes:
        - If any stream has more than max_annotated_points we turn
          annotations off, otherwise the labels pile up

    Signature:
        out (str):
            - The PNG stem to write into the spec
        title (str):
            - Figure title, or empty for none
        y_label (str):
            - Label for the y axis
        y_fmt (str):
            - Format spec for the annotated values
        streams (List[Dict[str, Any]]):
            - Streams from streams_from_sweep or streams_from_runs
    '''
    crowded = any(len(s['points']) > max_annotated_points for s in streams)
    spec: Dict[str, Any] = dict(
        out      = out,
        y_axis   = dict(label=y_label, format=y_fmt),
        annotate = not crowded,
        streams  = streams,
    )
    if title:
        spec['title'] = title
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Build a plot_streams.py spec from a sweep CSV or '
                      'from PAI run folders'
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--csv',
        type = str,
        help = 'by-dendrite-separate CSV from get_wandb_results.py',
    )
    source.add_argument(
        '--stream',
        type   = str,
        action = 'append',
        help   = 'NAME:folder1,folder2 of PAI run folders, repeatable',
    )
    parser.add_argument(
        '--out',
        type     = str,
        required = True,
        help     = 'PNG stem the spec will render to',
    )
    parser.add_argument(
        '--spec-path',
        type    = str,
        default = '',
        help    = 'where to write the spec, defaults to <out>.json here',
    )
    parser.add_argument(
        '--title',
        type    = str,
        default = '',
        help    = 'figure title, none by default',
    )
    parser.add_argument(
        '--y-label',
        type    = str,
        default = 'Score',
        help    = 'y axis label',
    )
    parser.add_argument(
        '--y-format',
        type    = str,
        default = '.4f',
        help    = 'format spec for annotated y values',
    )
    parser.add_argument(
        '--stat',
        type    = str,
        default = 'max',
        choices = ['max', 'mean', 'median'],
        help    = 'sweep CSV only, how to collapse runs in a column',
    )
    parser.add_argument(
        '--csv-metric',
        type    = str,
        default = 'val',
        choices = ['val', 'test'],
        help    = 'sweep CSV only, which metric columns to read',
    )
    parser.add_argument(
        '--metric',
        type    = str,
        default = default_metric,
        help    = 'run folders only, column of best_arch_scores.csv',
    )
    parser.add_argument(
        '--best',
        type    = str,
        default = 'max',
        choices = ['max', 'min'],
        help    = 'run folders only, which row of a folder is best',
    )
    parser.add_argument(
        '--all-dendrites',
        action = 'store_true',
        help   = 'run folders only, one point per architecture row',
    )
    args = parser.parse_args()

    if args.csv:
        streams = streams_from_sweep(args.csv, args.stat, args.csv_metric)
    else:
        streams = streams_from_runs(
            args.stream, args.metric, args.best, args.all_dendrites)

    spec      = build_spec(
        args.out, args.title, args.y_label, args.y_format, streams)
    spec_path = args.spec_path or f'{args.out}.json'
    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2)
    print(f'wrote {spec_path} with {len(streams)} stream(s)')
