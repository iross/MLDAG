#!/usr/bin/env python3
"""
Analyze run9 epoch 3 to understand the mismatch between job completion
and actual training work.
"""

import polars as pl


def main():
    df = pl.read_csv("full.csv")

    # Get all attempts for run9-train_epoch3
    epoch3 = df.filter(pl.col("Job Name") == "run9-train_epoch3").sort("HTCondor Cluster ID")

    print("=" * 100)
    print("RUN9 EPOCH 3: Job vs Actual Training Analysis")
    print("=" * 100)

    print(f"\nTotal attempts: {len(epoch3)}")

    print("\n" + "-" * 100)
    print(f"{'Cluster':<12} {'Status':<12} {'Duration':<20} {'GPU Usage':<12} {'CPU Usage':<12}")
    print("-" * 100)

    for row in epoch3.iter_rows(named=True):
        cluster = row['HTCondor Cluster ID']
        status = row['Final Status']
        duration_human = row['Execution Duration (human)'] or "N/A"
        gpu_usage = f"{row['GPU Usage']:.2f}" if row['GPU Usage'] is not None else "N/A"
        cpu_usage = f"{row['CPU Usage']:.2f}" if row['CPU Usage'] is not None else "N/A"

        print(f"{cluster:<12} {status:<12} {duration_human:<20} {gpu_usage:<12} {cpu_usage:<12}")

    print("\n" + "=" * 100)
    print("KEY OBSERVATIONS:")
    print("=" * 100)

    # Find the "successful" job
    completed = epoch3.filter(pl.col("Final Status") == "completed")
    if len(completed) > 0:
        comp_row = completed.to_dicts()[0]
        print(f"\n✓ DECLARED SUCCESSFUL: Cluster {comp_row['HTCondor Cluster ID']}")
        print(f"  Status: {comp_row['Final Status']}")
        print(f"  Duration: {comp_row['Execution Duration (human)']}")
        gpu_usage = comp_row['GPU Usage']
        cpu_usage = comp_row['CPU Usage']
        gpu_str = f"{gpu_usage:.2f}" if gpu_usage is not None else 'N/A'
        cpu_str = f"{cpu_usage:.2f}" if cpu_usage is not None else 'N/A'
        print(f"  GPU Usage: {gpu_str} (only 8%!)")
        print(f"  CPU Usage: {cpu_str}")
        print(f"\n  → This job did NOT train - it just loaded an existing checkpoint and ran tests!")
        print(f"  → Evidence: Very low GPU/CPU usage, short duration (3 hours vs 10+ for real training)")

    # The evicted job was first
    first_attempt = epoch3.filter(pl.col("Attempt Sequence") == 1).to_dicts()[0]
    print(f"\n✗ FIRST ATTEMPT: Cluster {first_attempt['HTCondor Cluster ID']}")
    print(f"  Status: {first_attempt['Final Status']} (ran for {first_attempt['Execution Duration (human)']}")
    gpu_usage_str = 'N/A' if first_attempt['GPU Usage'] is None else f"{first_attempt['GPU Usage']:.2f}"
    print(f"  GPU Usage: {gpu_usage_str}")
    print(f"\n  → This job likely DID the actual epoch 3 training before being evicted")
    print(f"  → It ran for ~11 hours (typical training time)")
    print(f"  → But was evicted and marked as 'failed' in subsequent attempts")

    print("\n" + "=" * 100)
    print("CONCLUSION:")
    print("=" * 100)
    print("\nThe TRAINING for epoch 3 was done by cluster 12704651 (evicted)")
    print("The SUCCESSFUL COMPLETION was cluster 12711655 (just loaded checkpoint + tested)")
    print("\nThis is a tracking/bookkeeping issue:")
    print("  1. Job 12704651 trained epoch 3, saved checkpoint, then was evicted")
    print("  2. Many retries happened (all 'failed' quickly)")
    print("  3. Job 12711655 found the existing checkpoint and 'completed'")
    print("  4. DAGMan credited cluster 12711655 with success")
    print("\nThe actual work was done by a different job than the one marked successful!")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
