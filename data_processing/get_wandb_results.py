################################################################################
# Fetch a wandb sweep and build CSVs of dendrite architecture progression.     #
################################################################################

#
"""
Imports
"""
import os
import re
import csv
import sys
import wandb
import argparse
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple

#
"""
Config
"""

# Metric names and the column names wandb may log them under
metric_column_aliases = {
    'arch_param_count': [
        'Arch Param Count', 'arch_param_count', 'Arch_Param_Count'
    ],
    'arch_max_val': [
        'Arch Max Val', 'arch_max_val', 'Arch_Max_Val'
    ],
    'arch_max_test': [
        'Arch Max Test', 'arch_max_test', 'Arch_Max_Test'
    ],
    'arch_dendrite_count': [
        'Arch Dendrite Count', 'arch_dendrite_count', 'Arch_Dendrite_Count'
    ],
    'final_param_count': [
        'Final Param Count', 'final_param_count', 'Final_Param_Count'
    ],
    'final_max_val': [
        'Final Max Val', 'final_max_val', 'Final_Max_Val'
    ],
    'final_max_test': [
        'Final Max Test', 'final_max_test', 'Final_Max_Test'
    ],
    'final_dendrite_count': [
        'Final Dendrite Count', 'final_dendrite_count', 'Final_Dendrite_Count'
    ],
}

#
"""
Functions
"""
def parse_wandb_url(url: str) -> Tuple[str, str, str]:
    '''
    Split a wandb sweep URL into its entity, project and sweep_id

    Notes:
        - We expect the URL to look like
          https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}, with or
          without the app. subdomain
        - If it does not match we raise ValueError so the caller can print
          what the URL should have looked like

    Signature:
        url (str):
            - The sweep URL copied from the wandb browser page
    '''
    pattern = r'https?://(?:app\.)?wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/?]+)'
    match   = re.match(pattern, url)
    if not match:
        raise ValueError(
            'Invalid wandb URL format. Expected: '
            'https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}\n'
            f'Got: {url}'
        )
    entity, project, sweep_id = match.groups()
    return entity, project, sweep_id

def find_metric_column(columns: pd.Index, keys: List[str]) -> Optional[str]:
    '''
    Find the first of keys that this run logged as a column

    Notes:
        - If none of the keys is a column we return None, so the caller
          knows this run never logged that metric under any spelling

    Signature:
        columns (pd.Index):
            - Column names of one run's history DataFrame
        keys (List[str]):
            - The spellings of one metric, tried in order
    '''
    for key in keys:
        if key in columns:
            return key
    return None

def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Rename whichever spelling of each metric column wandb used to our name

    Notes:
        - If df already has the column under our name we leave it alone
          and do not rename any alias, so we never end up with two
          columns for one metric

    Signature:
        df (pd.DataFrame):
            - The raw rows straight from wandb, still under the column
              names wandb logged
    '''
    rename_map = {}
    for our_name, aliases in metric_column_aliases.items():
        if our_name in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = our_name
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def infer_model_index_from_row(row: pd.Series) -> Optional[int]:
    '''
    Work out which model a run used from one of its rows

    Notes:
        - We read config_model_index first, since the sweep config is the
          most reliable place for it
        - If the run has no config_model_index, or it is not an integer, we
          fall back to the model_index_N token in the run name, and give up
          with None if that is missing too

    Signature:
        row (pd.Series):
            - Any single raw row of the run, the model index is the same
              on all of them
    '''
    config_model_index = row.get('config_model_index', None)
    if pd.notna(config_model_index):
        try:
            return int(config_model_index)
        except (TypeError, ValueError):
            # A non integer config value falls through to the name pattern
            pass
    run_name = str(row.get('run_name', ''))
    match    = re.search(r'model_index_(\d+)', run_name)
    if match:
        return int(match.group(1))
    return None

def extract_model_index(run_name: str) -> Optional[int]:
    '''
    Read the model index off a model_index_N run name

    Notes:
        - This is stricter than infer_model_index_from_row, the name has to
          start with the token and there is no config fallback. We return
          None when it does not match

    Signature:
        run_name (str):
            - The wandb run name, model_index_3_lr_0.1 for example
    '''
    match = re.match(r'model_index_(\d+)', run_name)
    if match:
        return int(match.group(1))
    return None

def get_model_type(row: pd.Series) -> str:
    '''
    Name the model a row belongs to, for the by-model pivot

    Notes:
        - We call it model_N when infer_model_index_from_row finds an index
          and model_unknown when it does not, so those rows still get a
          column instead of being dropped

    Signature:
        row (pd.Series):
            - One raw row of the run, only used for its model index
    '''
    model_index = infer_model_index_from_row(row)
    if model_index is not None:
        return f'model_{model_index}'
    return 'model_unknown'

def model_sort_key(item: Tuple[str, Any]) -> Tuple[int, str, int]:
    '''
    Sort (model_type, dendrite_count) pivot columns by model then dendrite

    Notes:
        - model_N columns come first, ordered by N and then by dendrite
          count. Any other model type goes after them, sorted by name, so
          model_unknown ends up at the end

    Signature:
        item (Tuple[str, Any]):
            - One column label from the pivot, a (model_type,
              dendrite_count) pair
    '''
    model_type, dendrite_count = item
    model_match = re.match(r'model_(\d+)$', str(model_type))
    if model_match:
        model_rank = int(model_match.group(1)) * 100000
        return (0, '', model_rank + int(dendrite_count))
    return (1, str(model_type), int(dendrite_count))

def flat_column_sort_key(col: str) -> Tuple[int, str, int]:
    '''
    Sort flattened model_N_dendrite_M_max_val or _max_test column names

    Notes:
        - Same order as model_sort_key, by N then M, but read from the
          flattened name after the pivot columns were joined into strings
        - Names that do not match the pattern go after the rest, sorted
          alphabetically

    Signature:
        col (str):
            - A flattened column name such as model_0_dendrite_2_max_val
    '''
    pattern = r'(model_\d+)_dendrite_(\d+)_max_(?:val|test)$'
    match   = re.match(pattern, str(col))
    if not match:
        return (1, str(col), 0)
    model_match = re.match(r'model_(\d+)$', match.group(1))
    if model_match:
        model_rank = int(model_match.group(1)) * 100000
        return (0, '', model_rank + int(match.group(2)))
    return (1, str(col), int(match.group(2)))

def flat_dendrite_sort_key(col: str) -> Tuple[int, str, int]:
    '''
    Sort flattened dendrite_M_max_val or _max_test column names by M

    Notes:
        - Names that do not match the pattern go after the rest, sorted
          alphabetically

    Signature:
        col (str):
            - A flattened column name such as dendrite_2_max_val
    '''
    match = re.match(r'dendrite_(\d+)_max_(?:val|test)$', str(col))
    if not match:
        return (1, str(col), 0)
    return (0, '', int(match.group(1)))

def get_dendrite_offset(
    run_name        : str,
    dendrite_offsets: Dict[str, int],
) -> int:
    '''
    Look up how many dendrites a run started with

    Notes:
        - We take the first prefix in dendrite_offsets that appears
          anywhere in run_name, so dict order matters if two prefixes
          could both match
        - A run with no matching prefix starts at dendrite 0, which is
          right for any model trained from scratch

    Signature:
        run_name (str):
            - The wandb run name, which carries the model_index_N prefix
        dendrite_offsets (Dict[str, int]):
            - Run name prefix to starting dendrite count, for pretrained
              models that begin with dendrites
    '''
    for prefix, offset in dendrite_offsets.items():
        if prefix in run_name:
            return offset
    return 0

def filter_ignored_models(
    df            : pd.DataFrame,
    ignored_models: List[int],
) -> pd.DataFrame:
    '''
    Drop every run whose model index is in ignored_models

    Notes:
        - We take the model index from the first row of each run with
          infer_model_index_from_row, since every row of a run shares it
        - We print how many runs and rows were removed so a typo in
          --ignore-models is easy to spot

    Signature:
        df (pd.DataFrame):
            - The raw rows straight from wandb, with a run_id column
        ignored_models (List[int]):
            - Model indices to remove, from the --ignore-models flag
    '''
    if not ignored_models or df.empty:
        return df
    ignored_set     = set(ignored_models)
    run_model_index = (
        df.groupby('run_id')
        .apply(lambda g: infer_model_index_from_row(g.iloc[0]))
        .reset_index(name='model_index')
    )
    ignored_run_ids = set(
        run_model_index[
            run_model_index['model_index'].isin(ignored_set)
        ]['run_id'].tolist()
    )
    print('\nIgnored model filter applied:')
    print(f'  Ignored model indices: {sorted(ignored_set)}')
    if ignored_run_ids:
        before_runs = df['run_id'].nunique() if 'run_id' in df.columns else 0
        before_rows = len(df)
        df          = df[~df['run_id'].isin(ignored_run_ids)].copy()
        after_runs  = df['run_id'].nunique() if 'run_id' in df.columns else 0
        after_rows  = len(df)
        print(f'  Removed runs: {before_runs - after_runs}')
        print(f'  Removed rows: {before_rows - after_rows}')
    else:
        print('  No matching runs found to remove')
    return df

def validate_input_dataframe(
    df         : pd.DataFrame,
    input_label: str,
) -> pd.DataFrame:
    '''
    Check that an input CSV is the raw arch_scores download

    Notes:
        - We normalize the column aliases first so a CSV saved under the
          wandb spellings still passes
        - If the file looks like a by-run pivot or a by-dendrite output we
          raise ValueError naming the file, since those are outputs of this
          script and cannot be read back in. A file missing the required
          columns raises the same way

    Signature:
        df (pd.DataFrame):
            - The rows read from the --csv file, before any checks
        input_label (str):
            - The file name, only used to make the error messages specific
    '''
    df               = normalize_metric_columns(df)
    has_param_count  = 'arch_param_count' in df.columns
    has_score_column = (
        'arch_max_val' in df.columns or 'arch_max_test' in df.columns
    )
    if has_param_count and has_score_column:
        return df
    if 'Arch Param Count' in df.columns and not has_score_column:
        raise ValueError(
            f"Input file '{input_label}' appears to be a by-run pivot "
            'CSV, not a raw arch_scores CSV. Use the raw download file '
            "(for example '*_arch_scores.csv') as --csv."
        )
    is_by_dendrite_output = 'param_count' in df.columns and any(
        col.startswith('dendrite_') for col in df.columns
    )
    if is_by_dendrite_output:
        raise ValueError(
            f"Input file '{input_label}' appears to be a by-dendrite "
            'output CSV, not a raw arch_scores CSV. Use the raw download '
            "file (for example '*_arch_scores.csv') as --csv."
        )
    missing_columns = []
    if not has_param_count:
        missing_columns.append('arch_param_count')
    if not has_score_column:
        missing_columns.append('arch_max_val or arch_max_test')
    raise ValueError(
        f"Input file '{input_label}' is missing required columns: "
        f"{', '.join(missing_columns)}. Expected a raw arch_scores CSV "
        'containing columns like arch_param_count and arch_max_val '
        '(or arch_max_test).'
    )

def get_sweep_results(
    entity       : str,
    project      : str,
    sweep_id     : str,
    include_final: bool = False,
    max_completed: int  = 0,
) -> pd.DataFrame:
    '''
    Collect the raw log rows from every run of a wandb sweep

    Notes:
        - We only report runs in the finished state that logged both Final
          Param Count and Final Max Val. The rest count as excluded so the
          summary at the end explains any runs that are missing
        - Every history row that holds at least one arch metric becomes
          one row here, tagged with the run's config values as config_*
          columns
        - We use scan_history() so wandb does not sample rows away, and
          fall back to run.history() only if it fails

    Signature:
        entity (str):
            - The wandb entity, a username or a team name
        project (str):
            - The wandb project that holds the sweep
        sweep_id (str):
            - The sweep ID, either bare or as a full entity/project/id path
        include_final (bool):
            - Whether to keep the Final metrics on each row too
        max_completed (int):
            - Keep only the first N reported runs, 0 keeps all of them
    '''
    api = wandb.Api()
    # A sweep_id that already holds slashes is a full path
    if '/' in sweep_id:
        sweep_path = sweep_id
    else:
        sweep_path = f'{entity}/{project}/{sweep_id}'
    print(f'Fetching sweep: {sweep_path}')
    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f'Error fetching sweep: {e}')
        sys.exit(1)
    runs        = sweep.runs
    all_results = []
    print(f'Processing {len(runs)} runs...')

    arch_names  = [n for n in metric_column_aliases if n.startswith('arch_')]
    final_names = [n for n in metric_column_aliases if n.startswith('final_')]

    total_runs                = len(runs)
    reported_runs             = 0
    excluded_runs             = 0
    failed_or_incomplete_runs = 0
    completed_discarded_runs  = 0
    selected_run_ids          = []

    for i, run in enumerate(runs):
        print(f'  Run {i+1}/{len(runs)}: {run.name} ({run.id})')
        # scan_history() returns every logged row
        # run.history() may sample or limit the data
        try:
            history_list = list(run.scan_history())
            if not history_list:
                print('    No history data found')
                continue
            history = pd.DataFrame(history_list)
        except Exception as e:
            print(f'    Error fetching history: {e}')
            print('    Falling back to run.history()...')
            history = run.history()
            if history.empty:
                print('    No history data found')
                continue
        print(f'    Fetched {len(history)} history entries')

        # Resolve which alias of each metric this run logged
        metric_cols = {
            name: find_metric_column(history.columns, aliases)
            for name, aliases in metric_column_aliases.items()
        }
        arch_param_count_col  = metric_cols['arch_param_count']
        arch_max_val_col      = metric_cols['arch_max_val']
        arch_max_test_col     = metric_cols['arch_max_test']
        final_param_count_col = metric_cols['final_param_count']
        final_max_val_col     = metric_cols['final_max_val']
        final_max_test_col    = metric_cols['final_max_test']

        # Only report runs that finished and logged final metrics
        run_finished_state   = str(run.state).lower() == 'finished'
        run_has_final_scores = (
            final_param_count_col is not None
            and final_max_val_col is not None
            and history[final_param_count_col].notna().any()
            and history[final_max_val_col].notna().any()
        )
        if not run_finished_state or not run_has_final_scores:
            excluded_runs             += 1
            failed_or_incomplete_runs += 1
            continue
        if max_completed > 0 and reported_runs >= max_completed:
            excluded_runs            += 1
            completed_discarded_runs += 1
            continue
        reported_runs += 1
        selected_run_ids.append(run.id)

        has_arch = (
            arch_param_count_col is not None
            or arch_max_val_col is not None
            or arch_max_test_col is not None
        )
        has_final = (
            final_param_count_col is not None
            or final_max_val_col is not None
            or final_max_test_col is not None
        )
        if not has_arch and not has_final:
            print('    No relevant metrics found in history')
            continue

        # Keep every row that holds at least one requested metric
        for idx, row in history.iterrows():
            values = {
                name: row.get(col) if col else None
                for name, col in metric_cols.items()
            }
            if include_final:
                checked_names = arch_names + final_names
            else:
                checked_names = arch_names
            if all(pd.isna(values[name]) for name in checked_names):
                continue
            entry = {
                'run_id'   : run.id,
                'run_name' : run.name,
                'state'    : run.state,
                'step'     : row.get('_step', None),
                'timestamp': row.get('_timestamp', None),
            }
            for name in checked_names:
                entry[name] = values[name]
            for config_key, config_val in run.config.items():
                entry[f'config_{config_key}'] = config_val
            all_results.append(entry)

        run_entries = [e for e in all_results if e['run_id'] == run.id]
        print(f'    Found {len(run_entries)} log entries')

    df = pd.DataFrame(all_results)
    if max_completed > 0 and not df.empty and selected_run_ids:
        df = df[df['run_id'].isin(selected_run_ids)].copy()
    df = normalize_metric_columns(df)

    print('\nRun completion summary (wandb fetch):')
    print(f'  Total runs: {total_runs}')
    print(f'  Reported runs (finished with final scores): {reported_runs}')
    print(
        '  Excluded runs (unfinished or missing final scores): '
        f'{excluded_runs}'
    )
    if max_completed > 0:
        print(f'  Requested max completed runs: {max_completed}')
        print(f'  Failed/incomplete runs: {failed_or_incomplete_runs}')
        print(
            '  Completed runs discarded by max-completed: '
            f'{completed_discarded_runs}'
        )
    print(f'\nTotal raw log entries: {len(df)}')
    return df

def create_graph_by_run(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Pivot the raw rows into one column for each run

    Notes:
        - Rows are Arch Param Count, columns are run names and the values
          are Arch Max Val, so each run plots as one line
        - If a (param_count, run_name) pair repeats we keep the max value

    Signature:
        df (pd.DataFrame):
            - The raw rows straight from wandb, one for each logged
              architecture
    '''
    both_present = df['arch_param_count'].notna() & df['arch_max_val'].notna()
    df_filtered  = df[both_present].copy()
    if df_filtered.empty:
        print(
            'Warning: No entries with both Arch Param Count and Arch Max '
            'Val found!'
        )
        return pd.DataFrame()
    pivot_df = df_filtered.pivot_table(
        index   = 'arch_param_count',
        columns = 'run_name',
        values  = 'arch_max_val',
        aggfunc = 'max',
    )
    pivot_df = pivot_df.sort_index()
    # Name the index so the CSV header reads as the X axis
    pivot_df.index.name = 'Arch Param Count'
    print('\nCreated pivot table:')
    print(f'  Rows (Arch Param Count): {len(pivot_df)}')
    print(f'  Columns (Runs): {len(pivot_df.columns)}')
    return pivot_df

def create_graph_by_dendrite(
    df               : pd.DataFrame,
    dendrite_offsets : Optional[Dict[str, int]] = None,
    separate_by_model: bool                     = False,
) -> pd.DataFrame:
    '''
    Pivot the raw rows into one score column for each dendrite count

    Notes:
        - Each output row is one (run_id, run_name, param_count) and the
          columns are dendrite_N_max_val followed by dendrite_N_max_test
          for every dendrite count we saw
        - With separate_by_model we split the columns by model type first,
          so they read model_0_dendrite_2_max_val and so on
        - If the run logged Arch Dendrite Count we use it, and compare it
          against the count we get from row order plus dendrite_offsets.
          We print any mismatches so a wrong offset shows up
        - If nothing was logged we use the computed count

    Signature:
        df (pd.DataFrame):
            - The raw rows straight from wandb, one for each logged
              architecture
        dendrite_offsets (Optional[Dict[str, int]]):
            - Run name prefix to starting dendrite count, for pretrained
              models that begin with dendrites
        separate_by_model (bool):
            - Whether to split the columns by model type as well as by
              dendrite count
    '''
    if dendrite_offsets is None:
        dendrite_offsets = {}
    available_score_columns = [
        col for col in ('arch_max_val', 'arch_max_test')
        if col in df.columns and df[col].notna().any()
    ]
    if not available_score_columns:
        print('Warning: No entries with Arch Max Val/Test found!')
        return pd.DataFrame()

    # Keep rows with a param_count and at least one score metric
    has_any_score = pd.Series(False, index=df.index)
    for score_col in available_score_columns:
        has_any_score = has_any_score | df[score_col].notna()
    df_filtered = df[df['arch_param_count'].notna() & has_any_score].copy()
    if df_filtered.empty:
        print(
            'Warning: No entries with both Arch Param Count and Arch Max '
            'Val found!'
        )
        return pd.DataFrame()
    df_filtered = df_filtered.sort_values(['run_name', 'step'])

    has_logged_dendrite_count = (
        'arch_dendrite_count' in df_filtered.columns
        and df_filtered['arch_dendrite_count'].notna().any()
    )
    df_filtered['run_offset'] = df_filtered['run_name'].apply(
        lambda name: get_dendrite_offset(name, dendrite_offsets)
    )
    if has_logged_dendrite_count:
        print('\n=== Using LOGGED Arch Dendrite Count ===')
        df_filtered['dendrite_count'] = df_filtered['arch_dendrite_count']
        # Compare against the count implied by row order for diagnostics
        df_filtered['computed_dendrite_count'] = (
            df_filtered.groupby('run_id').cumcount()
            + df_filtered['run_offset']
        )
        mismatch_mask = (
            df_filtered['dendrite_count']
            != df_filtered['computed_dendrite_count']
        )
        mismatches = df_filtered[mismatch_mask]
        if not mismatches.empty:
            print(
                f'\n⚠️  WARNING: Found {len(mismatches)} entries where '
                'logged dendrite count differs from computed!'
            )
            print('\nShowing first 10 mismatches:')
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_columns', None)
            mismatch_cols = [
                'run_id', 'step', 'arch_param_count',
                'dendrite_count', 'computed_dendrite_count',
            ]
            print(mismatches[mismatch_cols].head(10))
        else:
            print('✓ Logged dendrite counts match computed counts')
    else:
        print('\n=== Computing dendrite count (no logged values found) ===')
        df_filtered['dendrite_count'] = (
            df_filtered.groupby('run_id').cumcount()
            + df_filtered['run_offset']
        )

    if separate_by_model:
        df_filtered['model_type'] = df_filtered.apply(get_model_type, axis=1)
        pivot_columns   = ['model_type', 'dendrite_count']
        column_sort_key = flat_column_sort_key
    else:
        pivot_columns   = 'dendrite_count'
        column_sort_key = flat_dendrite_sort_key

    # Take the first value when a (run, param_count, dendrite) repeats
    score_pivots = []
    for score_col in available_score_columns:
        metric_suffix = 'val' if score_col == 'arch_max_val' else 'test'
        metric_pivot  = df_filtered.pivot_table(
            index   = ['run_id', 'run_name', 'arch_param_count'],
            columns = pivot_columns,
            values  = score_col,
            aggfunc = 'first',
        )
        if separate_by_model:
            sorted_columns = sorted(
                metric_pivot.columns.tolist(), key=model_sort_key
            )
            metric_pivot = metric_pivot.reindex(columns=sorted_columns)
            metric_pivot.columns = [
                f'{model_type}_dendrite_{int(count)}_max_{metric_suffix}'
                for model_type, count in metric_pivot.columns
            ]
        else:
            metric_pivot.columns = [
                f'dendrite_{int(col)}_max_{metric_suffix}'
                for col in metric_pivot.columns
            ]
        score_pivots.append(metric_pivot)
    scatter_df = pd.concat(score_pivots, axis=1)

    # Order the score columns as every _max_val then every _max_test
    ordered_score_columns: List[str] = []
    for suffix in ('_max_val', '_max_test'):
        suffix_columns = [
            c for c in scatter_df.columns if c.endswith(suffix)
        ]
        ordered_score_columns.extend(
            sorted(suffix_columns, key=column_sort_key)
        )
    scatter_df = scatter_df[ordered_score_columns]

    scatter_df = scatter_df.reset_index()
    scatter_df = scatter_df.rename(columns={'arch_param_count': 'param_count'})
    scatter_df = scatter_df.sort_values(['run_id', 'param_count'])
    dendrite_columns = [col for col in scatter_df.columns if 'dendrite' in col]
    print('\nCreated scatter plot data:')
    print(f'  Total rows: {len(scatter_df)}')
    print(f"  Unique runs: {scatter_df['run_id'].nunique()}")
    print(f'  Dendrite columns: {len(dendrite_columns)}')
    return scatter_df

def diagnose_data(
    df              : pd.DataFrame,
    dendrite_offsets: Optional[Dict[str, int]] = None,
) -> None:
    '''
    Print a summary of the raw rows and flag dendrite gaps

    Notes:
        - If Arch Dendrite Count was logged we check every run for
          dendrite counts missing between where it should start and its
          max, and for repeated (param_count, dendrite_count) pairs
        - Where a run should start comes from dendrite_offsets, and is 0
          when no prefix matches, so a pretrained model with its offset
          set is not flagged for the dendrites it began with

    Signature:
        df (pd.DataFrame):
            - The raw rows straight from wandb, one for each logged
              architecture
        dendrite_offsets (Optional[Dict[str, int]]):
            - Run name prefix to starting dendrite count, for pretrained
              models that begin with dendrites
    '''
    if dendrite_offsets is None:
        dendrite_offsets = {}
    print('\n' + '=' * 70)
    print('DIAGNOSTIC SUMMARY')
    print('=' * 70)

    score_column = None
    if 'arch_max_val' in df.columns and df['arch_max_val'].notna().any():
        score_column = 'arch_max_val'
    elif 'arch_max_test' in df.columns and df['arch_max_test'].notna().any():
        score_column = 'arch_max_test'
    if score_column is None:
        print('No data with arch_param_count and arch_max_val/arch_max_test')
        return
    both_present = df['arch_param_count'].notna() & df[score_column].notna()
    df_filtered  = df[both_present].copy()
    if df_filtered.empty:
        print(
            'No data with both arch_param_count and '
            'arch_max_val/arch_max_test'
        )
        return

    has_dendrite_count = (
        'arch_dendrite_count' in df_filtered.columns
        and df_filtered['arch_dendrite_count'].notna().any()
    )
    print(f'\nTotal entries: {len(df_filtered)}')
    print(f'Has logged dendrite count: {has_dendrite_count}')
    print(f"Total unique runs: {df_filtered['run_id'].nunique()}")

    if has_dendrite_count:
        print('\n--- Checking for GAPS/MISSING DENDRITES ---')
        issues_found = False
        for run_id in df_filtered['run_id'].unique():
            run_data = df_filtered[df_filtered['run_id'] == run_id]
            run_data = run_data.sort_values('step')
            run_name = run_data['run_name'].iloc[0]
            dendrite_counts = sorted(run_data['arch_dendrite_count'].unique())
            if dendrite_counts:
                actual_sequence = [int(d) for d in dendrite_counts]
                max_dend        = max(actual_sequence)
                expected_start  = get_dendrite_offset(
                    run_name,
                    dendrite_offsets,
                )
                # A model with pretrained dendrites is expected to begin at
                # expected_start rather than 0
                expected_sequence = list(range(expected_start, max_dend + 1))
                missing = set(expected_sequence) - set(actual_sequence)
                if missing:
                    issues_found = True
                    print(f'\n⚠️  ISSUE: Run ID: {run_id}')
                    print(f'    Run name: {run_name}')
                    if expected_start > 0:
                        # Name the model index from the matching prefix
                        model_idx_str = None
                        for prefix in dendrite_offsets:
                            if prefix in run_name:
                                match = re.match(r'model_index_(\d+)', prefix)
                                if match:
                                    model_idx_str = match.group(1)
                                break
                        if model_idx_str:
                            print(
                                f'    Model {model_idx_str} configured to '
                                f'start at dendrite {expected_start}'
                            )
                    print(
                        f'    Expected dendrites ({expected_start} to '
                        f'{max_dend}): {expected_sequence}'
                    )
                    print(
                        '    Actual dendrites:                       '
                        f'{actual_sequence}'
                    )
                    print(
                        '    MISSING:                                 '
                        f'{sorted(missing)}'
                    )
                    dendrite_dist = run_data['arch_dendrite_count']
                    dendrite_dist = dendrite_dist.value_counts().sort_index()
                    print('    Dendrite count distribution:')
                    for dend, count in dendrite_dist.items():
                        print(f'      Dendrite {int(dend)}: {count} entries')
        if not issues_found:
            print('  ✓ No missing dendrites or gaps found')

        print(
            '\n--- Checking for duplicate (param_count, dendrite_count) '
            'pairs ---'
        )
        duplicates_found = False
        for run_id in df_filtered['run_id'].unique():
            run_data   = df_filtered[df_filtered['run_id'] == run_id]
            run_name   = run_data['run_name'].iloc[0]
            duplicates = run_data.groupby(
                ['arch_param_count', 'arch_dendrite_count']
            ).size()
            duplicates = duplicates[duplicates > 1]
            if not duplicates.empty:
                duplicates_found = True
                print(
                    f'\n⚠️  Run ID {run_id} ({run_name}) has duplicate '
                    '(param_count, dendrite_count) pairs:'
                )
                for (pc, dc), count in duplicates.items():
                    print(
                        f'    Param={pc}, Dendrite={int(dc)}: {count} entries'
                    )
        if not duplicates_found:
            print('  ✓ No duplicates found')

    print('\n' + '=' * 70 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description     = 'Fetch full results from a wandb sweep',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = (
            '\nExample:\n'
            '  %(prog)s --url '
            'https://wandb.ai/perforated-ai/pets/sweeps/lk4t23x7\n'
            '  %(prog)s --url '
            'https://wandb.ai/perforated-ai/pets/sweeps/lk4t23x7 '
            '--output results.csv\n'
            '  %(prog)s --csv perforated-ai_pets_i00x001o_arch_scores.csv '
            '--mode gen-by-run\n'
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--url',
        default = '',
        help    = (
            'wandb sweep URL '
            '(e.g., https://wandb.ai/entity/project/sweeps/sweep_id)'
        ),
    )
    input_group.add_argument(
        '--csv',
        default = '',
        help    = (
            'path to an existing raw CSV file (download mode output) to '
            'use as input instead of fetching from wandb'
        ),
    )
    parser.add_argument(
        '--mode',
        choices = [
            'download', 'gen-by-run', 'by-dendrite', 'by-dendrite-separate'
        ],
        default = 'download',
        help    = (
            'output mode: "download" for raw data, "gen-by-run" for line '
            'graph by run, "by-dendrite" for scatter plot by dendrite '
            'count, "by-dendrite-separate" for scatter plot split by model '
            'type + dendrite count (default: download)'
        ),
    )
    parser.add_argument(
        '--output',
        default = '',
        help    = (
            'output CSV file path (optional). If not specified, uses '
            'entity_project_sweep_arch_scores.csv'
        ),
    )
    parser.add_argument(
        '--dendrite-offset',
        nargs   = '*',
        default = [],
        metavar = 'MODEL_INDEX:COUNT',
        help    = (
            'specify starting dendrite count for model indices. Format: '
            '"0:2" "1:3" (model_index_0 starts at 2, model_index_1 starts '
            'at 3)'
        ),
    )
    parser.add_argument(
        '--include-final',
        action = 'store_true',
        help   = (
            'include Final Param Count, Final Max Val, and Final Dendrite '
            'Count metrics (logged at end of run). Useful for verification '
            'but not for graph generation.'
        ),
    )
    parser.add_argument(
        '--max-completed',
        type    = int,
        default = 0,
        help    = (
            'only include the first N runs that are finished and have '
            'final scores, 0 means all of them. When set, also prints the '
            'failed/incomplete count and the completed-discarded count '
            '(default: 0)'
        ),
    )
    parser.add_argument(
        '--ignore-models',
        nargs   = '*',
        type    = int,
        default = [],
        metavar = 'MODEL_INDEX',
        help    = (
            'ignore one or more model indices (from model_info.csv), e.g. '
            '--ignore-models 0 2 3'
        ),
    )
    args = parser.parse_args()

    if args.max_completed < 0:
        print(
            'Error: --max-completed must be a positive integer, or 0 for '
            'all runs',
            file = sys.stderr,
        )
        sys.exit(1)

    # Expand each "0:2" spec to the model_index_0 prefix it stands for
    dendrite_offsets = {}
    for offset_spec in args.dendrite_offset:
        try:
            model_idx, count = offset_spec.split(':', 1)
            model_idx = int(model_idx)
            count     = int(count)
            prefix    = f'model_index_{model_idx}'
            dendrite_offsets[prefix] = count
        except (ValueError, AttributeError):
            print(
                f"Warning: Invalid dendrite offset format '{offset_spec}'. "
                "Expected 'model_index:count' (e.g., '0:2')"
            )
            continue
    if dendrite_offsets:
        print('Dendrite offsets configured:')
        for prefix, count in dendrite_offsets.items():
            print(
                f"  Runs starting with '{prefix}' begin at dendrite count "
                f'{count}'
            )

    entity       = None
    project      = None
    sweep_id     = None
    raw_csv_file = None
    output_stem  = None

    if args.csv:
        if args.max_completed > 0:
            print('Warning: --max-completed is ignored when using --csv input')
        if not os.path.exists(args.csv):
            print(
                f'Error: Input CSV file not found: {args.csv}',
                file = sys.stderr,
            )
            sys.exit(1)
        print(f'Loading data from input CSV: {args.csv}')
        try:
            df = validate_input_dataframe(pd.read_csv(args.csv), args.csv)
        except ValueError as e:
            print(f'Error: {e}', file=sys.stderr)
            sys.exit(1)
        print(f'Loaded {len(df)} raw log entries from CSV')
        output_stem = os.path.splitext(os.path.basename(args.csv))[0]
    else:
        try:
            entity, project, sweep_id = parse_wandb_url(args.url)
            print('Parsed URL:')
            print(f'  Entity: {entity}')
            print(f'  Project: {project}')
            print(f'  Sweep ID: {sweep_id}\n')
        except ValueError as e:
            print(f'Error: {e}', file=sys.stderr)
            sys.exit(1)
        raw_csv_file = f'{entity}_{project}_{sweep_id}_arch_scores.csv'
        output_stem  = f'{entity}_{project}_{sweep_id}'

        # Reuse a raw CSV from an earlier download unless a run limit is set
        use_existing_csv = (
            args.mode != 'download'
            and os.path.exists(raw_csv_file)
            and args.max_completed == 0
        )
        if use_existing_csv:
            print(f'Found existing raw data file: {raw_csv_file}')
            print(
                'Loading data from file instead of fetching from wandb...\n'
            )
            try:
                df = validate_input_dataframe(
                    pd.read_csv(raw_csv_file),
                    raw_csv_file,
                )
            except ValueError as e:
                print(f'Error: {e}', file=sys.stderr)
                sys.exit(1)
            print(f'Loaded {len(df)} raw log entries from CSV')
        else:
            df = get_sweep_results(
                entity,
                project,
                sweep_id,
                include_final = args.include_final,
                max_completed = args.max_completed,
            )
            if df.empty:
                print('No results found!')
                sys.exit(1)
            if args.mode == 'download':
                output_file = args.output if args.output else raw_csv_file
                df.to_csv(output_file, index=False)
                print(f'\nResults saved to: {output_file}')

    df = filter_ignored_models(df, args.ignore_models)
    if df.empty:
        print('No results remain after applying --ignore-models filter')
        sys.exit(1)
    diagnose_data(df, dendrite_offsets)

    # After a download, suggest offsets for models whose runs all start
    # above dendrite 0
    if args.mode == 'download':
        score_column = None
        if 'arch_max_val' in df.columns and df['arch_max_val'].notna().any():
            score_column = 'arch_max_val'
        elif (
            'arch_max_test' in df.columns
            and df['arch_max_test'].notna().any()
        ):
            score_column = 'arch_max_test'
        if score_column is None:
            df_filtered = pd.DataFrame()
        else:
            both_present = (
                df['arch_param_count'].notna() & df[score_column].notna()
            )
            df_filtered = df[both_present].copy()
        if not df_filtered.empty:
            has_dendrite_count = (
                'arch_dendrite_count' in df_filtered.columns
                and df_filtered['arch_dendrite_count'].notna().any()
            )
            if has_dendrite_count:
                # Group run ids by the model index in their run name
                model_groups = {}
                for run_id in df_filtered['run_id'].unique():
                    run_data  = df_filtered[df_filtered['run_id'] == run_id]
                    run_name  = run_data['run_name'].iloc[0]
                    model_idx = extract_model_index(run_name)
                    if model_idx is not None:
                        if model_idx not in model_groups:
                            model_groups[model_idx] = []
                        model_groups[model_idx].append(run_id)

                # Suggest an offset for each unconfigured model whose runs
                # all start above dendrite 0
                suggestions = []
                for model_idx, run_ids in model_groups.items():
                    prefix_full = f'model_index_{model_idx}'
                    if prefix_full in dendrite_offsets:
                        continue
                    all_dendrite_counts = []
                    for run_id in run_ids:
                        run_data = df_filtered[df_filtered['run_id'] == run_id]
                        dendrite_counts = (
                            run_data['arch_dendrite_count'].dropna().unique()
                        )
                        all_dendrite_counts.extend(dendrite_counts)
                    if all_dendrite_counts:
                        min_dendrite = int(min(all_dendrite_counts))
                        if min_dendrite > 0:
                            suggestions.append((model_idx, min_dendrite))

                if suggestions:
                    print('\n' + '=' * 70)
                    print('💡 SUGGESTION')
                    print('=' * 70)
                    print(
                        'Some models start at dendrite counts above 0, '
                        'indicating pretrained dendrites.'
                    )
                    print(
                        "To suppress warnings about 'missing' dendrites, "
                        'add:\n'
                    )
                    offset_args = ' '.join(
                        [f'"{m}:{d}"' for m, d in suggestions]
                    )
                    if args.url:
                        script_name = os.path.basename(sys.argv[0])
                        print(
                            f'  python {script_name} --url {args.url} '
                            f'--dendrite-offset {offset_args}'
                        )
                    print('\nDetails:')
                    for model_idx, start_dend in suggestions:
                        print(
                            f'  Model {model_idx} starts at dendrite '
                            f'{start_dend}'
                        )
                    print('=' * 70 + '\n')

    # Download mode is finished once the raw CSV and diagnostics are written
    if args.mode == 'download':
        sys.exit(0)

    if args.mode == 'gen-by-run':
        output_df = create_graph_by_run(df)
        if output_df.empty:
            print('Failed to create pivot table!')
            sys.exit(1)
        mode_suffix = 'by_run'
        save_index  = True
    elif args.mode == 'by-dendrite':
        output_df = create_graph_by_dendrite(df, dendrite_offsets)
        if output_df.empty:
            print('Failed to create scatter plot data!')
            sys.exit(1)
        mode_suffix = 'by_dendrite'
        save_index  = False
    elif args.mode == 'by-dendrite-separate':
        output_df = create_graph_by_dendrite(
            df,
            dendrite_offsets,
            separate_by_model = True,
        )
        if output_df.empty:
            print('Failed to create scatter plot data!')
            sys.exit(1)
        mode_suffix = 'by_dendrite_separate'
        save_index  = False
    else:
        # argparse choices make this branch unreachable
        output_df   = df
        mode_suffix = 'arch_scores'
        save_index  = False

    if args.output:
        output_file = args.output
    else:
        output_file = f'{output_stem}_{mode_suffix}.csv'

    if args.mode == 'by-dendrite-separate':
        # Two header rows give the param_count of every dendrite column.
        # All runs share one param_count for each model and dendrite pair, so
        # any row with a value in the column supplies it
        dendrite_cols = [col for col in output_df.columns if 'dendrite' in col]
        dendrite_cols = (
            [col for col in dendrite_cols if col.endswith('_max_val')]
            + [col for col in dendrite_cols if col.endswith('_max_test')]
        )
        non_dendrite_cols = [
            col for col in output_df.columns if 'dendrite' not in col
        ]
        n_prefix         = len(non_dendrite_cols)
        col_param_counts = {}
        for col in dendrite_cols:
            non_null = output_df.loc[output_df[col].notna(), 'param_count']
            if non_null.empty:
                col_param_counts[col] = ''
            else:
                col_param_counts[col] = non_null.iloc[0]
        label_row = [''] * n_prefix + [
            f'param_count {col}' for col in dendrite_cols
        ]
        value_row = [''] * n_prefix + [
            col_param_counts[col] for col in dendrite_cols
        ]
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(label_row)
            writer.writerow(value_row)
        output_df.to_csv(output_file, index=save_index, mode='a')
    else:
        output_df.to_csv(output_file, index=save_index)
    print(f'\nResults saved to: {output_file}')
