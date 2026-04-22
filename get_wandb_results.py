#!/usr/bin/env python3
"""
Script to fetch full results from a Weights & Biases (wandb) sweep.
Extracts all raw log entries for Arch Param Count (X-axis) and Arch Max Val (Y-axis).

USAGE:
    python get_wandb_results.py <WANDB_URL> [OPTIONS]

MODES:
    The script supports three output modes controlled by the --mode/-m argument:

    1. download (default)
       - Downloads all raw log entries from the sweep
       - Output: entity_project_sweep_arch_scores.csv
       - Columns: run_id, run_name, state, step, timestamp, arch_param_count, 
                  arch_max_val, config_*
       - Use when: You need the raw data for custom analysis
       - Example: python get_wandb_results.py "https://wandb.ai/entity/project/sweeps/abc123"

    2. gen-by-run
       - Creates a pivot table for line graphs where each run is a separate line
       - Output: entity_project_sweep_by_run.csv
       - Format: Rows = Arch Param Count, Columns = Run names, Values = Arch Max Val
       - Use when: You want to compare different runs as lines on a single graph
       - Example: python get_wandb_results.py "URL" -m gen-by-run

    3. by-dendrite
       - Creates scatter plot data grouped by dendrite count
       - Output: entity_project_sweep_by_dendrite.csv
       - Format: Columns = run_name, param_count, dendrite_0_max_val, dendrite_1_max_val, ...
       - Use when: You want to visualize how different dendrite configurations perform
       - Example: python get_wandb_results.py "URL" -m by-dendrite

DENDRITE OFFSET:
    Use --dendrite-offset when runs don't all start at dendrite count 0.
    Specify "prefix:count" pairs to set starting dendrite counts for runs with specific prefixes.
    
    Example:
        If runs starting with "model_index_0" actually have 2 dendrites already:
        python get_wandb_results.py "URL" -m by-dendrite --dendrite-offset "model_index_0:2"
    
    Multiple offsets:
        python get_wandb_results.py "URL" -m by-dendrite \\
            --dendrite-offset "model_index_0:2" "model_index_1:3"

CSV CACHING:
    When using gen-by-run or by-dendrite modes, the script automatically checks for an
    existing raw CSV file (entity_project_sweep_arch_scores.csv) and uses it if found,
    avoiding unnecessary wandb API calls. To force a fresh download, delete the CSV
    or use download mode first.

EXAMPLES:
    # Download raw data
    python get_wandb_results.py "https://wandb.ai/myteam/myproject/sweeps/abc123"
    
    # Generate line graph format (using cached CSV if available)
    python get_wandb_results.py "https://wandb.ai/myteam/myproject/sweeps/abc123" -m gen-by-run
    
    # Generate scatter plot with dendrite offsets
    python get_wandb_results.py "https://wandb.ai/myteam/myproject/sweeps/abc123" \\
        -m by-dendrite --dendrite-offset "model_index_0:2"
    
    # Custom output filename
    python get_wandb_results.py "URL" -m by-dendrite -o my_results.csv
"""

import wandb
import argparse
import pandas as pd
import sys
import re
import os
from typing import List, Dict, Any, Tuple


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


def get_sweep_results(entity: str, project: str, sweep_id: str) -> pd.DataFrame:
    """
    Fetch all runs from a wandb sweep and extract all raw log entries.
    
    Args:
        entity: wandb entity (username or team name)
        project: wandb project name
        sweep_id: sweep ID (can be full path or just the ID)
    
    Returns:
        DataFrame with all raw log entries containing arch_param_count and arch_max_val
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
    arch_param_count_keys = ['Arch Param Count', 'arch_param_count', 'Arch_Param_Count']
    arch_max_val_keys = ['Arch Max Val', 'arch_max_val', 'Arch_Max_Val']
    
    for i, run in enumerate(runs):
        print(f"  Run {i+1}/{len(runs)}: {run.name} ({run.id})")
        
        # Fetch full history (all logged values)
        history = run.history()
        
        if history.empty:
            print(f"    No history data found")
            continue
        
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
        
        if arch_param_count_col is None and arch_max_val_col is None:
            print(f"    No relevant metrics found in history")
            continue
        
        # Extract rows that have at least one of our metrics
        for idx, row in history.iterrows():
            arch_param_count = row.get(arch_param_count_col) if arch_param_count_col else None
            arch_max_val = row.get(arch_max_val_col) if arch_max_val_col else None
            
            # Skip rows where both metrics are NaN
            if pd.isna(arch_param_count) and pd.isna(arch_max_val):
                continue
            
            entry = {
                'run_id': run.id,
                'run_name': run.name,
                'state': run.state,
                'step': row.get('_step', None),
                'timestamp': row.get('_timestamp', None),
                'arch_param_count': arch_param_count,
                'arch_max_val': arch_max_val,
            }
            
            # Add config parameters
            for config_key, config_val in run.config.items():
                entry[f'config_{config_key}'] = config_val
            
            all_results.append(entry)
        
        print(f"    Found {len([e for e in all_results if e['run_id'] == run.id])} log entries")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
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


def create_graph_by_dendrite(df: pd.DataFrame, dendrite_offsets: Dict[str, int] = None) -> pd.DataFrame:
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
        DataFrame with param_count and dendrite max_val columns
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
            if run_name.startswith(prefix):
                return offset
        return 0
    
    # Assign dendrite count based on order within each run, with offsets
    df_filtered['base_count'] = df_filtered.groupby('run_name').cumcount()
    df_filtered['run_offset'] = df_filtered['run_name'].apply(get_dendrite_offset)
    df_filtered['dendrite_count'] = df_filtered['base_count'] + df_filtered['run_offset']
    
    # Pivot: rows are (run_name, param_count), columns are dendrite counts
    scatter_df = df_filtered.pivot_table(
        index=['run_name', 'arch_param_count'],
        columns='dendrite_count',
        values='arch_max_val',
        aggfunc='first'  # Take first value if duplicates
    )
    
    # Rename columns to be more descriptive
    scatter_df.columns = [f'dendrite_{int(col)}_max_val' for col in scatter_df.columns]
    
    # Reset index to make run_name and param_count regular columns
    scatter_df = scatter_df.reset_index()
    scatter_df = scatter_df.rename(columns={'arch_param_count': 'param_count'})
    
    # Sort by run_name and param_count
    scatter_df = scatter_df.sort_values(['run_name', 'param_count'])
    
    print(f"\nCreated scatter plot data:")
    print(f"  Total rows: {len(scatter_df)}")
    print(f"  Unique runs: {scatter_df['run_name'].nunique()}")
    print(f"  Dendrite columns: {len([col for col in scatter_df.columns if 'dendrite' in col])}")
    
    return scatter_df


def main():
    parser = argparse.ArgumentParser(
        description='Fetch full results from a wandb sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s https://wandb.ai/perforated-ai/pets/sweeps/lk4t23x7
  %(prog)s https://wandb.ai/perforated-ai/pets/sweeps/lk4t23x7 --output results.csv
        """
    )
    
    parser.add_argument(
        'url',
        help='wandb sweep URL (e.g., https://wandb.ai/entity/project/sweeps/sweep_id)'
    )
    
    parser.add_argument(
        '-m', '--mode',
        choices=['download', 'gen-by-run', 'by-dendrite'],
        default='download',
        help='output mode: "download" for raw data, "gen-by-run" for line graph by run, "by-dendrite" for scatter plot by dendrite count (default: download)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='output CSV file path (optional). If not specified, uses entity_project_sweep_arch_scores.csv'
    )
    
    parser.add_argument(
        '--dendrite-offset',
        nargs='*',
        default=[],
        metavar='PREFIX:COUNT',
        help='specify starting dendrite count for run name prefixes (e.g., "model_index_0:2" "model_index_1:3")'
    )
    
    args = parser.parse_args()
    
    # Parse dendrite offsets
    dendrite_offsets = {}
    for offset_spec in args.dendrite_offset:
        try:
            prefix, count = offset_spec.split(':', 1)
            dendrite_offsets[prefix] = int(count)
        except (ValueError, AttributeError):
            print(f"Warning: Invalid dendrite offset format '{offset_spec}'. Expected 'prefix:count'")
            continue
    
    if dendrite_offsets:
        print(f"Dendrite offsets configured:")
        for prefix, count in dendrite_offsets.items():
            print(f"  '{prefix}' starts at dendrite count {count}")
    
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
    
    if args.mode != 'download' and os.path.exists(raw_csv_file):
        # Load existing CSV instead of fetching
        print(f"Found existing raw data file: {raw_csv_file}")
        print(f"Loading data from file instead of fetching from wandb...\n")
        df = pd.read_csv(raw_csv_file)
        print(f"Loaded {len(df)} raw log entries from CSV")
    else:
        # Fetch sweep results from wandb
        df = get_sweep_results(entity, project, sweep_id)
        
        if df.empty:
            print("No results found!")
            sys.exit(1)
        
        # Save raw data if in download mode
        if args.mode == 'download':
            output_file = args.output if args.output else raw_csv_file
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
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
    else:
        # Should not reach here
        output_df = df
        mode_suffix = "arch_scores"
        save_index = False
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        output_file = f"{entity}_{project}_{sweep_id}_{mode_suffix}.csv"
    
    # Save to CSV
    output_df.to_csv(output_file, index=save_index)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
