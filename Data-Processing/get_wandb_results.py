#!/usr/bin/env python3
"""
Script to fetch and analyze results from Weights & Biases sweeps.
Extracts architecture progression and final metrics from wandb logs.

QUICK START:
    python get_wandb_results.py "https://wandb.ai/entity/project/sweeps/SWEEP_ID"

MODES:
    --mode download (default) - Download raw CSV with all metrics, other modes use that csv
    --mode gen-by-run         - Generate pivot table for line graphs (runs as lines)
    --mode by-dendrite        - Generate scatter plot data (grouped by dendrite count)
    --mode by-dendrite-separate - Generate scatter plot data grouped by model type + dendrite count

COMMON OPTIONS:
    --include-final       - Include final metrics (for verification, not graphing)
    --dendrite-offset     - Specify starting dendrite counts (e.g., "0:2" for pretrained models)

FULL DOCUMENTATION:
    For detailed usage, output formats, and workflows, see:
    .github/skills/perforatedai/api-docs/wandb.md (Analyzing Sweep Results section)

EXAMPLES:
    # Download raw data
    python get_wandb_results.py "https://wandb.ai/myteam/project/sweeps/abc123"
    
    # Generate by-run comparison
    python get_wandb_results.py "URL" --mode gen-by-run
    
    # Analyze dendrite progression with pretrained model offset
    python get_wandb_results.py "URL" --mode by-dendrite --dendrite-offset "0:2"
    
    # Generate scatter plot with dendrite offset
    python get_wandb_results.py "https://wandb.ai/myteam/myproject/sweeps/abc123" \\
        --mode by-dendrite --dendrite-offset "0:2"
    
    # Multiple models with different offsets
    python get_wandb_results.py "https://wandb.ai/myteam/myproject/sweeps/abc123" \\
        --mode by-dendrite --dendrite-offset "0:2" "1:3"
    
    # Custom output filename
    python get_wandb_results.py "URL" --mode by-dendrite --output my_results.csv
    
    # Include Final metrics to verify they match last Arch values
    python get_wandb_results.py "https://wandb.ai/myteam/myproject/sweeps/abc123" --include-final
"""

import wandb
import argparse
import pandas as pd
import sys
import re
import os
from typing import List, Dict, Any, Tuple


METRIC_COLUMN_ALIASES = {
    'arch_param_count': ['Arch Param Count', 'arch_param_count', 'Arch_Param_Count'],
    'arch_max_val': ['Arch Max Val', 'arch_max_val', 'Arch_Max_Val'],
    'arch_dendrite_count': ['Arch Dendrite Count', 'arch_dendrite_count', 'Arch_Dendrite_Count'],
    'final_param_count': ['Final Param Count', 'final_param_count', 'Final_Param_Count'],
    'final_max_val': ['Final Max Val', 'final_max_val', 'Final_Max_Val'],
    'final_dendrite_count': ['Final Dendrite Count', 'final_dendrite_count', 'Final_Dendrite_Count'],
}


def parse_wandb_url(url: str) -> Tuple[str, str, str]:
    """
    Parse a wandb sweep URL to extract entity, project, and sweep_id.
    
    Expected format: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}/...
    
    Args:
        url: wandb sweep URL
    
    Returns:
        Tuple of (entity, project, sweep_id)
    """
    # Match pattern: https://wandb.ai/entity/project/sweeps/sweep_id
    pattern = r'https?://(?:app\.)?wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/?]+)'
    
    match = re.match(pattern, url)
    if not match:
        raise ValueError(
            f"Invalid wandb URL format. Expected: https://wandb.ai/{{entity}}/{{project}}/sweeps/{{sweep_id}}\n"
            f"Got: {url}"
        )
    
    entity, project, sweep_id = match.groups()
    return entity, project, sweep_id


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported wandb metric column aliases to canonical names."""
    rename_map = {}

    for canonical_name, aliases in METRIC_COLUMN_ALIASES.items():
        if canonical_name in df.columns:
            continue

        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical_name
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def validate_input_dataframe(df: pd.DataFrame, input_label: str) -> pd.DataFrame:
    """Validate that an input DataFrame has the raw arch score schema this script expects."""
    df = normalize_metric_columns(df)

    required_columns = {'arch_param_count', 'arch_max_val'}
    missing_columns = sorted(required_columns - set(df.columns))
    if not missing_columns:
        return df

    if 'Arch Param Count' in df.columns and 'arch_max_val' not in df.columns:
        raise ValueError(
            f"Input file '{input_label}' appears to be a by-run pivot CSV, not a raw arch_scores CSV. "
            f"Use the raw download file (for example '*_arch_scores.csv') as --csv."
        )

    if 'param_count' in df.columns and any(col.startswith('dendrite_') for col in df.columns):
        raise ValueError(
            f"Input file '{input_label}' appears to be a by-dendrite output CSV, not a raw arch_scores CSV. "
            f"Use the raw download file (for example '*_arch_scores.csv') as --csv."
        )

    raise ValueError(
        f"Input file '{input_label}' is missing required columns: {', '.join(missing_columns)}. "
        f"Expected a raw arch_scores CSV containing columns like arch_param_count and arch_max_val."
    )


def get_sweep_results(entity: str, project: str, sweep_id: str, include_final: bool = False) -> pd.DataFrame:
    """
    Fetch all runs from a wandb sweep and extract all raw log entries.
    
    Args:
        entity: wandb entity (username or team name)
        project: wandb project name
        sweep_id: sweep ID (can be full path or just the ID)
        include_final: whether to include Final metrics (Final Param Count, etc.)
    
    Returns:
        DataFrame with all raw log entries containing arch metrics and optionally 
        final metrics (param_count, max_val, dendrite_count)
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Parse sweep_id if it's a full path
    if '/' in sweep_id:
        sweep_path = sweep_id
    else:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    
    print(f"Fetching sweep: {sweep_path}")
    
    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"Error fetching sweep: {e}")
        sys.exit(1)
    
    # Get all runs from the sweep
    runs = sweep.runs
    
    all_results = []
    
    print(f"Processing {len(runs)} runs...")
    
    # Look for these metric names in history
    arch_param_count_keys = METRIC_COLUMN_ALIASES['arch_param_count']
    arch_max_val_keys = METRIC_COLUMN_ALIASES['arch_max_val']
    arch_dendrite_count_keys = METRIC_COLUMN_ALIASES['arch_dendrite_count']
    
    final_param_count_keys = METRIC_COLUMN_ALIASES['final_param_count']
    final_max_val_keys = METRIC_COLUMN_ALIASES['final_max_val']
    final_dendrite_count_keys = METRIC_COLUMN_ALIASES['final_dendrite_count']

    total_runs = len(runs)
    reported_runs = 0
    excluded_runs = 0
    
    for i, run in enumerate(runs):
        print(f"  Run {i+1}/{len(runs)}: {run.name} ({run.id})")
        
        # Fetch full history (all logged values) - use scan_history() to get ALL entries
        # run.history() might sample/limit data, scan_history() returns everything
        try:
            # scan_history() returns iterator, convert to list then DataFrame
            history_list = list(run.scan_history())
            if not history_list:
                print(f"    No history data found")
                continue
            history = pd.DataFrame(history_list)
        except Exception as e:
            print(f"    Error fetching history: {e}")
            print(f"    Falling back to run.history()...")
            history = run.history()
            if history.empty:
                print(f"    No history data found")
                continue
        
        print(f"    Fetched {len(history)} history entries")
        
        # Find which column names exist in this run's history
        arch_param_count_col = None
        for key in arch_param_count_keys:
            if key in history.columns:
                arch_param_count_col = key
                break
        
        arch_max_val_col = None
        for key in arch_max_val_keys:
            if key in history.columns:
                arch_max_val_col = key
                break
        
        arch_dendrite_count_col = None
        for key in arch_dendrite_count_keys:
            if key in history.columns:
                arch_dendrite_count_col = key
                break
        
        final_param_count_col = None
        final_max_val_col = None
        final_dendrite_count_col = None

        for key in final_param_count_keys:
            if key in history.columns:
                final_param_count_col = key
                break

        for key in final_max_val_keys:
            if key in history.columns:
                final_max_val_col = key
                break

        for key in final_dendrite_count_keys:
            if key in history.columns:
                final_dendrite_count_col = key
                break

        # Only report runs that finished and actually logged final metrics.
        run_finished_state = str(run.state).lower() == 'finished'
        run_has_final_scores = (
            final_param_count_col is not None
            and final_max_val_col is not None
            and history[final_param_count_col].notna().any()
            and history[final_max_val_col].notna().any()
        )

        if not run_finished_state or not run_has_final_scores:
            excluded_runs += 1
            continue

        reported_runs += 1
        
        # Check if we have any relevant metrics
        has_arch = arch_param_count_col is not None or arch_max_val_col is not None
        has_final = final_param_count_col is not None or final_max_val_col is not None
        
        if not has_arch and not has_final:
            print(f"    No relevant metrics found in history")
            continue
        
        # Extract rows that have at least one of our metrics
        for idx, row in history.iterrows():
            arch_param_count = row.get(arch_param_count_col) if arch_param_count_col else None
            arch_max_val = row.get(arch_max_val_col) if arch_max_val_col else None
            arch_dendrite_count = row.get(arch_dendrite_count_col) if arch_dendrite_count_col else None
            
            final_param_count = None
            final_max_val = None
            final_dendrite_count = None
            
            if include_final:
                final_param_count = row.get(final_param_count_col) if final_param_count_col else None
                final_max_val = row.get(final_max_val_col) if final_max_val_col else None
                final_dendrite_count = row.get(final_dendrite_count_col) if final_dendrite_count_col else None
            
            # Skip rows where all metrics are NaN
            if include_final:
                all_nan = (pd.isna(arch_param_count) and pd.isna(arch_max_val) and pd.isna(arch_dendrite_count) and
                          pd.isna(final_param_count) and pd.isna(final_max_val) and pd.isna(final_dendrite_count))
            else:
                all_nan = (pd.isna(arch_param_count) and pd.isna(arch_max_val) and pd.isna(arch_dendrite_count))
            
            if all_nan:
                continue
            
            entry = {
                'run_id': run.id,
                'run_name': run.name,
                'state': run.state,
                'step': row.get('_step', None),
                'timestamp': row.get('_timestamp', None),
                'arch_param_count': arch_param_count,
                'arch_max_val': arch_max_val,
                'arch_dendrite_count': arch_dendrite_count,
            }
            
            if include_final:
                entry['final_param_count'] = final_param_count
                entry['final_max_val'] = final_max_val
                entry['final_dendrite_count'] = final_dendrite_count
            
            # Add config parameters
            for config_key, config_val in run.config.items():
                entry[f'config_{config_key}'] = config_val
            
            all_results.append(entry)
        
        print(f"    Found {len([e for e in all_results if e['run_id'] == run.id])} log entries")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    df = normalize_metric_columns(df)

    print("\nRun completion summary (wandb fetch):")
    print(f"  Total runs: {total_runs}")
    print(f"  Reported runs (finished with final scores): {reported_runs}")
    print(f"  Excluded runs (unfinished or missing final scores): {excluded_runs}")
    
    print(f"\nTotal raw log entries: {len(df)}")
    
    return df


def create_graph_by_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table for easy graphing by run.
    
    Rows: Arch Param Count (X-axis values)
    Columns: Run names
    Values: Arch Max Val (Y-axis values)
    
    This allows creating line graphs where each run is a separate line.
    
    Args:
        df: DataFrame with raw log entries
    
    Returns:
        Pivot DataFrame suitable for graphing
    """
    # Filter to only rows with both metrics
    df_filtered = df[df['arch_param_count'].notna() & df['arch_max_val'].notna()].copy()
    
    if df_filtered.empty:
        print("Warning: No entries with both Arch Param Count and Arch Max Val found!")
        return pd.DataFrame()
    
    # Create pivot table
    # If there are multiple values for the same (param_count, run_name), take the max
    pivot_df = df_filtered.pivot_table(
        index='arch_param_count',
        columns='run_name',
        values='arch_max_val',
        aggfunc='max'  # Take max if there are duplicate entries
    )
    
    # Sort by arch_param_count (index)
    pivot_df = pivot_df.sort_index()
    
    # Rename index to make it clear
    pivot_df.index.name = 'Arch Param Count'
    
    print(f"\nCreated pivot table:")
    print(f"  Rows (Arch Param Count): {len(pivot_df)}")
    print(f"  Columns (Runs): {len(pivot_df.columns)}")
    
    return pivot_df


def create_graph_by_dendrite(
    df: pd.DataFrame,
    dendrite_offsets: Dict[str, int] = None,
    separate_by_model: bool = False,
) -> pd.DataFrame:
    """
    Create a scatter plot format grouped by dendrite count.
    
    Each row represents one (run_name, param_count) combination, with columns
    for max_val at each dendrite count.
    
    Format:
    run_name | param_count | dendrite_0_max_val | dendrite_1_max_val | dendrite_2_max_val | ...
    
    Args:
        df: DataFrame with raw log entries
        dendrite_offsets: Dict mapping run name prefixes to starting dendrite counts
    
    Returns:
        DataFrame with param_count and dendrite max_val columns.
        If separate_by_model is True, columns are split by (model_type, dendrite_count).
    """
    if dendrite_offsets is None:
        dendrite_offsets = {}
    
    # Filter to only rows with both metrics
    df_filtered = df[df['arch_param_count'].notna() & df['arch_max_val'].notna()].copy()
    
    if df_filtered.empty:
        print("Warning: No entries with both Arch Param Count and Arch Max Val found!")
        return pd.DataFrame()
    
    # Sort by run_name and step to ensure consistent ordering
    df_filtered = df_filtered.sort_values(['run_name', 'step'])
    
    # Function to get the starting dendrite count for a run
    def get_dendrite_offset(run_name):
        for prefix, offset in dendrite_offsets.items():
            if prefix in run_name:
                return offset
        return 0
    
    # Check if we have logged dendrite counts
    has_logged_dendrite_count = 'arch_dendrite_count' in df_filtered.columns and df_filtered['arch_dendrite_count'].notna().any()
    
    if has_logged_dendrite_count:
        print("\n=== Using LOGGED Arch Dendrite Count ===")
        # Use the logged dendrite count from wandb
        df_filtered['dendrite_count'] = df_filtered['arch_dendrite_count']
        
        # Also compute what it would have been for comparison
        df_filtered['computed_dendrite_count'] = df_filtered.groupby('run_id').cumcount()
        df_filtered['run_offset'] = df_filtered['run_name'].apply(get_dendrite_offset)
        df_filtered['computed_dendrite_count'] = df_filtered['computed_dendrite_count'] + df_filtered['run_offset']
        
        # Show comparison to identify discrepancies
        mismatches = df_filtered[df_filtered['dendrite_count'] != df_filtered['computed_dendrite_count']]
        if not mismatches.empty:
            print(f"\n⚠️  WARNING: Found {len(mismatches)} entries where logged dendrite count differs from computed!")
            print("\nShowing first 10 mismatches:")
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_columns', None)
            print(mismatches[['run_id', 'step', 'arch_param_count', 'dendrite_count', 'computed_dendrite_count']].head(10))
        else:
            print("✓ Logged dendrite counts match computed counts")
    else:
        print("\n=== Computing dendrite count (no logged values found) ===")
        # Compute dendrite count based on order within each run, with offsets
        df_filtered['base_count'] = df_filtered.groupby('run_id').cumcount()
        df_filtered['run_offset'] = df_filtered['run_name'].apply(get_dendrite_offset)
        df_filtered['dendrite_count'] = df_filtered['base_count'] + df_filtered['run_offset']
    
    if separate_by_model:
        # Split columns by model type first, then dendrite count (e.g. model_0_dendrite_2_max_val)
        def get_model_type(row: pd.Series) -> str:
            config_model_index = row.get('config_model_index', None)
            if pd.notna(config_model_index):
                try:
                    return f"model_{int(config_model_index)}"
                except (TypeError, ValueError):
                    pass

            run_name = str(row.get('run_name', ''))
            match = re.search(r'model_index_(\d+)', run_name)
            if match:
                return f"model_{int(match.group(1))}"

            return 'model_unknown'

        df_filtered['model_type'] = df_filtered.apply(get_model_type, axis=1)

        scatter_df = df_filtered.pivot_table(
            index=['run_id', 'run_name', 'arch_param_count'],
            columns=['model_type', 'dendrite_count'],
            values='arch_max_val',
            aggfunc='first'  # Take first value if duplicates
        )

        def model_sort_key(item: Tuple[str, Any]) -> Tuple[int, str, int]:
            model_type, dendrite_count = item
            model_match = re.match(r'model_(\d+)$', str(model_type))
            if model_match:
                return (0, '', int(model_match.group(1)) * 100000 + int(dendrite_count))
            return (1, str(model_type), int(dendrite_count))

        sorted_columns = sorted(scatter_df.columns.tolist(), key=model_sort_key)
        scatter_df = scatter_df.reindex(columns=sorted_columns)
        scatter_df.columns = [
            f'{model_type}_dendrite_{int(dendrite_count)}_max_val'
            for model_type, dendrite_count in scatter_df.columns
        ]
    else:
        # Pivot: rows are (run_id, run_name, param_count), columns are dendrite counts
        scatter_df = df_filtered.pivot_table(
            index=['run_id', 'run_name', 'arch_param_count'],
            columns='dendrite_count',
            values='arch_max_val',
            aggfunc='first'  # Take first value if duplicates
        )

        # Rename columns to be more descriptive
        scatter_df.columns = [f'dendrite_{int(col)}_max_val' for col in scatter_df.columns]
    
    # Reset index to make run_id, run_name and param_count regular columns
    scatter_df = scatter_df.reset_index()
    scatter_df = scatter_df.rename(columns={'arch_param_count': 'param_count'})
    
    # Sort by run_id and param_count
    scatter_df = scatter_df.sort_values(['run_id', 'param_count'])
    
    print(f"\nCreated scatter plot data:")
    print(f"  Total rows: {len(scatter_df)}")
    print(f"  Unique runs: {scatter_df['run_id'].nunique()}")
    print(f"  Dendrite columns: {len([col for col in scatter_df.columns if 'dendrite' in col])}")
    
    return scatter_df


def diagnose_data(df: pd.DataFrame, dendrite_offsets: Dict[str, int] = None):
    """
    Print diagnostic information about the data to help identify issues.
    
    Args:
        df: DataFrame with raw log entries
        dendrite_offsets: Dict mapping run name prefixes to starting dendrite count
    """
    if dendrite_offsets is None:
        dendrite_offsets = {}
    
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    # Filter to relevant data
    df_filtered = df[df['arch_param_count'].notna() & df['arch_max_val'].notna()].copy()
    
    if df_filtered.empty:
        print("No data with both arch_param_count and arch_max_val")
        return
    
    # Check for dendrite count column
    has_dendrite_count = 'arch_dendrite_count' in df_filtered.columns and df_filtered['arch_dendrite_count'].notna().any()
    
    print(f"\nTotal entries: {len(df_filtered)}")
    print(f"Has logged dendrite count: {has_dendrite_count}")
    print(f"Total unique runs: {df_filtered['run_id'].nunique()}")
    
    # Function to get expected starting dendrite for a run name
    def get_expected_start(run_name):
        """Get the expected starting dendrite count for a run based on configured offsets"""
        for prefix, start_count in dendrite_offsets.items():
            if prefix in run_name:
                return start_count
        return 0  # Default: expect to start from 0
    
    # Check ALL runs for issues (not just first 5)
    if has_dendrite_count:
        print("\n--- Checking for GAPS/MISSING DENDRITES ---")
        issues_found = False
        
        for run_id in df_filtered['run_id'].unique():
            run_data = df_filtered[df_filtered['run_id'] == run_id].sort_values('step')
            run_name = run_data['run_name'].iloc[0]  # Get run_name for this run_id
            dendrite_counts = sorted(run_data['arch_dendrite_count'].unique())
            
            # Check for issues
            if dendrite_counts:
                actual_sequence = [int(d) for d in dendrite_counts]
                min_dend = min(actual_sequence)
                max_dend = max(actual_sequence)
                
                # Get expected starting dendrite for this run
                expected_start = get_expected_start(run_name)
                
                # Expected sequence should start from expected_start, not 0
                # e.g., if model starts with 2 dendrites, expected is [2, 3, 4, ...]
                expected_sequence = list(range(expected_start, max_dend + 1))
                missing = set(expected_sequence) - set(actual_sequence)
                
                if missing:
                    issues_found = True
                    print(f"\n⚠️  ISSUE: Run ID: {run_id}")
                    print(f"    Run name: {run_name}")
                    if expected_start > 0:
                        # Extract model index from prefix for clearer messaging
                        model_idx_str = None
                        for prefix in dendrite_offsets:
                            if prefix in run_name:
                                # Extract number from "model_index_0" format
                                import re
                                match = re.match(r'model_index_(\d+)', prefix)
                                if match:
                                    model_idx_str = match.group(1)
                                break
                        if model_idx_str:
                            print(f"    Model {model_idx_str} configured to start at dendrite {expected_start}")
                    print(f"    Expected dendrites ({expected_start} to {max_dend}): {expected_sequence}")
                    print(f"    Actual dendrites:                       {actual_sequence}")
                    print(f"    MISSING:                                 {sorted(missing)}")
                    
                    # Show distribution
                    dendrite_dist = run_data['arch_dendrite_count'].value_counts().sort_index()
                    print(f"    Dendrite count distribution:")
                    for dend, count in dendrite_dist.items():
                        print(f"      Dendrite {int(dend)}: {count} entries")
        
        if not issues_found:
            print("  ✓ No missing dendrites or gaps found")
        
        # Check for duplicate (param_count, dendrite_count) pairs within runs
        print("\n--- Checking for duplicate (param_count, dendrite_count) pairs ---")
        duplicates_found = False
        for run_id in df_filtered['run_id'].unique():
            run_data = df_filtered[df_filtered['run_id'] == run_id]
            run_name = run_data['run_name'].iloc[0]  # Get run_name for this run_id
            duplicates = run_data.groupby(['arch_param_count', 'arch_dendrite_count']).size()
            duplicates = duplicates[duplicates > 1]
            if not duplicates.empty:
                duplicates_found = True
                print(f"\n⚠️  Run ID {run_id} ({run_name}) has duplicate (param_count, dendrite_count) pairs:")
                for (pc, dc), count in duplicates.items():
                    print(f"    Param={pc}, Dendrite={int(dc)}: {count} entries")
        
        if not duplicates_found:
            print("  ✓ No duplicates found")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch full results from a wandb sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --url https://wandb.ai/perforated-ai/pets/sweeps/lk4t23x7
  %(prog)s --url https://wandb.ai/perforated-ai/pets/sweeps/lk4t23x7 --output results.csv
    %(prog)s --csv perforated-ai_pets_i00x001o_arch_scores.csv --mode gen-by-run
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        '--url',
        help='wandb sweep URL (e.g., https://wandb.ai/entity/project/sweeps/sweep_id)'
    )

    input_group.add_argument(
        '--csv',
        help='path to an existing raw CSV file (download mode output) to use as input instead of fetching from wandb'
    )
    
    parser.add_argument(
        '--mode',
        choices=['download', 'gen-by-run', 'by-dendrite', 'by-dendrite-separate'],
        default='download',
        help='output mode: "download" for raw data, "gen-by-run" for line graph by run, "by-dendrite" for scatter plot by dendrite count, "by-dendrite-separate" for scatter plot split by model type + dendrite count (default: download)'
    )
    
    parser.add_argument(
        '--output',
        help='output CSV file path (optional). If not specified, uses entity_project_sweep_arch_scores.csv'
    )
    
    parser.add_argument(
        '--dendrite-offset',
        nargs='*',
        default=[],
        metavar='MODEL_INDEX:COUNT',
        help='specify starting dendrite count for model indices. Format: "0:2" "1:3" (model_index_0 starts at 2, model_index_1 starts at 3)'
    )
    
    parser.add_argument(
        '--include-final',
        action='store_true',
        help='include Final Param Count, Final Max Val, and Final Dendrite Count metrics (logged at end of run). Useful for verification but not for graph generation.'
    )
    
    args = parser.parse_args()
    
    # Parse dendrite offsets - support only numeric format "0:2" which expands to "model_index_0:2"
    dendrite_offsets = {}
    for offset_spec in args.dendrite_offset:
        try:
            model_idx, count = offset_spec.split(':', 1)
            model_idx = int(model_idx)
            count = int(count)
            
            # Expand to full prefix format
            prefix = f"model_index_{model_idx}"
            dendrite_offsets[prefix] = count
        except (ValueError, AttributeError):
            print(f"Warning: Invalid dendrite offset format '{offset_spec}'. Expected 'model_index:count' (e.g., '0:2')")
            continue
    
    if dendrite_offsets:
        print(f"Dendrite offsets configured:")
        for prefix, count in dendrite_offsets.items():
            print(f"  Runs starting with '{prefix}' begin at dendrite count {count}")
    
    entity = None
    project = None
    sweep_id = None
    raw_csv_file = None
    output_stem = None

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: Input CSV file not found: {args.csv}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading data from input CSV: {args.csv}")
        try:
            df = validate_input_dataframe(pd.read_csv(args.csv), args.csv)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(df)} raw log entries from CSV")
        output_stem = os.path.splitext(os.path.basename(args.csv))[0]
    else:
        # Parse URL to extract entity, project, and sweep_id
        try:
            entity, project, sweep_id = parse_wandb_url(args.url)
            print(f"Parsed URL:")
            print(f"  Entity: {entity}")
            print(f"  Project: {project}")
            print(f"  Sweep ID: {sweep_id}\n")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Determine if we need to fetch or can use existing CSV
        raw_csv_file = f"{entity}_{project}_{sweep_id}_arch_scores.csv"
        output_stem = f"{entity}_{project}_{sweep_id}"

        if args.mode != 'download' and os.path.exists(raw_csv_file):
            # Load existing CSV instead of fetching
            print(f"Found existing raw data file: {raw_csv_file}")
            print(f"Loading data from file instead of fetching from wandb...\n")
            try:
                df = validate_input_dataframe(pd.read_csv(raw_csv_file), raw_csv_file)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            print(f"Loaded {len(df)} raw log entries from CSV")
        else:
            # Fetch sweep results from wandb
            df = get_sweep_results(entity, project, sweep_id, include_final=args.include_final)

            if df.empty:
                print("No results found!")
                sys.exit(1)

            # Save raw data if in download mode
            if args.mode == 'download':
                output_file = args.output if args.output else raw_csv_file
                df.to_csv(output_file, index=False)
                print(f"\nResults saved to: {output_file}")
    
    # Always show diagnostic info
    diagnose_data(df, dendrite_offsets)
    
    # After download, check each model_index to suggest offsets if needed
    if args.mode == 'download':
        df_filtered = df[df['arch_param_count'].notna() & df['arch_max_val'].notna()].copy()
        if not df_filtered.empty:
            has_dendrite_count = 'arch_dendrite_count' in df_filtered.columns and df_filtered['arch_dendrite_count'].notna().any()
            
            if has_dendrite_count:
                import re
                def extract_model_index(run_name):
                    match = re.match(r'model_index_(\d+)', run_name)
                    if match:
                        return int(match.group(1))
                    return None
                
                # Group runs by model_index
                model_groups = {}
                for run_id in df_filtered['run_id'].unique():
                    run_data = df_filtered[df_filtered['run_id'] == run_id]
                    run_name = run_data['run_name'].iloc[0]
                    model_idx = extract_model_index(run_name)
                    if model_idx is not None:
                        if model_idx not in model_groups:
                            model_groups[model_idx] = []
                        model_groups[model_idx].append(run_id)
                
                # Check each model_index for consistent starting dendrite
                suggestions = []
                for model_idx, run_ids in model_groups.items():
                    # Check if already configured (either as model_index_N or just N)
                    prefix_full = f"model_index_{model_idx}"
                    if prefix_full in dendrite_offsets:
                        continue
                    
                    # Get all dendrite counts for this model
                    all_dendrite_counts = []
                    for run_id in run_ids:
                        run_data = df_filtered[df_filtered['run_id'] == run_id]
                        dendrite_counts = run_data['arch_dendrite_count'].dropna().unique()
                        all_dendrite_counts.extend(dendrite_counts)
                    
                    if all_dendrite_counts:
                        min_dendrite = int(min(all_dendrite_counts))
                        
                        # If all runs of this model start above 0, suggest offset
                        if min_dendrite > 0:
                            suggestions.append((model_idx, min_dendrite))
                
                if suggestions:
                    print(f"\n{'='*70}")
                    print(f"💡 SUGGESTION")
                    print(f"{'='*70}")
                    print(f"Some models start at dendrite counts above 0, indicating pretrained dendrites.")
                    print(f"To suppress warnings about 'missing' dendrites, add:\n")
                    
                    # Build the command with short format (just model index)
                    offset_args = " ".join([f'"{m}:{d}"' for m, d in suggestions])
                    if args.url:
                        print(f"  python {os.path.basename(sys.argv[0])} --url {args.url} --dendrite-offset {offset_args}")
                    
                    print(f"\nDetails:")
                    for model_idx, start_dend in suggestions:
                        print(f"  Model {model_idx} starts at dendrite {start_dend}")
                    print(f"{'='*70}\n")
    
    # If download mode, we're done
    if args.mode == 'download':
        return
    
    # Process non-download modes
    if args.mode == 'gen-by-run':
        # Create pivot table for graphing
        output_df = create_graph_by_run(df)
        if output_df.empty:
            print("Failed to create pivot table!")
            sys.exit(1)
        mode_suffix = "by_run"
        save_index = True
    elif args.mode == 'by-dendrite':
        # Create scatter plot data grouped by dendrite count
        output_df = create_graph_by_dendrite(df, dendrite_offsets)
        if output_df.empty:
            print("Failed to create scatter plot data!")
            sys.exit(1)
        mode_suffix = "by_dendrite"
        save_index = False
    elif args.mode == 'by-dendrite-separate':
        # Create scatter plot data grouped by model type + dendrite count
        output_df = create_graph_by_dendrite(df, dendrite_offsets, separate_by_model=True)
        if output_df.empty:
            print("Failed to create scatter plot data!")
            sys.exit(1)
        mode_suffix = "by_dendrite_separate"
        save_index = False
    else:
        # Should not reach here
        output_df = df
        mode_suffix = "arch_scores"
        save_index = False
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        output_file = f"{output_stem}_{mode_suffix}.csv"
    
    # Save to CSV
    if args.mode == 'by-dendrite-separate':
        dendrite_cols = [col for col in output_df.columns if 'dendrite' in col]
        non_dendrite_cols = [col for col in output_df.columns if 'dendrite' not in col]
        n_prefix = len(non_dendrite_cols)

        # Build param_count metadata rows: for each dendrite column, find the param_count
        # of any row that has a non-null value in it (all runs share the same param_count
        # for a given model+dendrite combination).
        col_param_counts = {}
        for col in dendrite_cols:
            non_null = output_df.loc[output_df[col].notna(), 'param_count']
            col_param_counts[col] = non_null.iloc[0] if not non_null.empty else ''

        label_row = [''] * n_prefix + [f'param_count {col}' for col in dendrite_cols]
        value_row = [''] * n_prefix + [col_param_counts[col] for col in dendrite_cols]

        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(label_row)
            writer.writerow(value_row)
        output_df.to_csv(output_file, index=save_index, mode='a')
    else:
        output_df.to_csv(output_file, index=save_index)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
