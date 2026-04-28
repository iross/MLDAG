#!/usr/bin/env python3
"""
Find Slow Transfers

Reads a CSV file and outputs Cluster IDs where transfer input duration
exceeds a specified threshold.

Usage:
    python find_slow_transfers.py <threshold_seconds> [input_file]

Examples:
    python find_slow_transfers.py 300                    # > 5 minutes, default full.csv
    python find_slow_transfers.py 600 job_summary.csv    # > 10 minutes, custom file
    python find_slow_transfers.py 1800 full.csv > slow_transfers  # Save to file
"""

import sys
import pandas as pd

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python find_slow_transfers.py <threshold_seconds> [input_file]")
        print("Example: python find_slow_transfers.py 300 full.csv")
        sys.exit(1)

    threshold = float(sys.argv[1])
    input_file = sys.argv[2] if len(sys.argv) > 2 else "full.csv"

    # Read CSV
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Check if required column exists
    duration_col = 'Transfer Input Duration (seconds)'
    cluster_col = 'HTCondor Cluster ID'

    if duration_col not in df.columns:
        print(f"Error: Column '{duration_col}' not found in CSV", file=sys.stderr)
        print(f"Available columns: {', '.join(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if cluster_col not in df.columns:
        print(f"Error: Column '{cluster_col}' not found in CSV", file=sys.stderr)
        sys.exit(1)

    # Filter for slow transfers
    slow_transfers = df[
        (df[duration_col].notna()) &
        (df[duration_col] > threshold)
    ]

    # Output Cluster IDs (one per line)
    for cluster_id in slow_transfers[cluster_col]:
        print(cluster_id)

    # Print summary to stderr so it doesn't interfere with piped output
    print(f"\nFound {len(slow_transfers)} transfers > {threshold}s ({threshold/60:.1f} min)", file=sys.stderr)
    if len(slow_transfers) > 0:
        print(f"Min: {slow_transfers[duration_col].min():.1f}s, "
              f"Max: {slow_transfers[duration_col].max():.1f}s, "
              f"Mean: {slow_transfers[duration_col].mean():.1f}s", file=sys.stderr)

if __name__ == "__main__":
    main()
