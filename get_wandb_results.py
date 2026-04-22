#!/usr/bin/env python3
"""
Script to fetch full results from a Weights & Biases (wandb) sweep.
Extracts all raw log entries for Arch Param Count (X-axis) and Arch Max Val (Y-axis).
"""

import wandb
import argparse
import pandas as pd
import sys
import re
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
        '-o', '--output',
        help='output CSV file path (optional). If not specified, uses entity_project_sweep_arch_scores.csv'
    )
    
    args = parser.parse_args()
    
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
    
    # Fetch sweep results
    df = get_sweep_results(entity, project, sweep_id)
    
    if df.empty:
        print("No results found!")
        sys.exit(1)
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        output_file = f"{entity}_{project}_{sweep_id}_arch_scores.csv"
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
