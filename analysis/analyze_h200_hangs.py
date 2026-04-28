#!/usr/bin/env python3
"""
Analyze H200 jobs to understand why some complete in 10 hours
while others run for 40+ hours before being killed.
"""

import polars as pl
from datetime import datetime


def main():
    df = pl.read_csv("full.csv")

    # Filter to H200 OSPool jobs
    df_h200 = df.filter(
        (pl.col("Targeted Resource") == "ospool") &
        (pl.col("GPU Device Name") == "NVIDIA H200")
    )

    # Split into completed and aborted with execution time
    df_completed = df_h200.filter(
        (pl.col("Final Status") == "completed") &
        (pl.col("Execution Duration (seconds)").is_not_null())
    )

    df_aborted = df_h200.filter(
        (pl.col("Final Status") == "aborted") &
        (pl.col("Execution Duration (seconds)").is_not_null())
    )

    print("=" * 100)
    print("H200 JOBS: Why do some hang for 40+ hours?")
    print("=" * 100)

    print(f"\nCompleted jobs: {len(df_completed)}")
    print(f"Aborted jobs: {len(df_aborted)}")

    # Compare execution times
    print("\n" + "-" * 100)
    print("EXECUTION TIME COMPARISON:")
    print("-" * 100)

    comp_exec = df_completed.select(pl.col("Execution Duration (seconds)") / 3600).to_series()
    abort_exec = df_aborted.select(pl.col("Execution Duration (seconds)") / 3600).to_series()

    print(f"\nCompleted jobs (hours):")
    print(f"  Min:    {comp_exec.min():.1f}")
    print(f"  Median: {comp_exec.median():.1f}")
    print(f"  Mean:   {comp_exec.mean():.1f}")
    print(f"  Max:    {comp_exec.max():.1f}")
    print(f"  Std:    {comp_exec.std():.1f}")

    print(f"\nAborted jobs (hours):")
    print(f"  Min:    {abort_exec.min():.1f}")
    print(f"  Median: {abort_exec.median():.1f}")
    print(f"  Mean:   {abort_exec.mean():.1f}")
    print(f"  Max:    {abort_exec.max():.1f}")
    print(f"  Std:    {abort_exec.std():.1f}")

    # Check if specific GPUs are problematic
    print("\n" + "-" * 100)
    print("GPU ANALYSIS:")
    print("-" * 100)

    # Count jobs by GPU UUID and status
    gpu_stats = df_h200.filter(
        pl.col("GPU UUID").is_not_null()
    ).group_by("GPU UUID").agg([
        pl.len().alias("total_jobs"),
        (pl.col("Final Status") == "completed").sum().alias("completed"),
        (pl.col("Final Status") == "aborted").sum().alias("aborted"),
        (pl.col("Final Status") == "failed").sum().alias("failed"),
        pl.col("Execution Duration (seconds)").mean().alias("avg_exec_seconds"),
    ]).sort("aborted", descending=True)

    print("\nGPUs with most aborted jobs:")
    for row in gpu_stats.head(10).iter_rows(named=True):
        uuid = row['GPU UUID'][:20] + "..."
        total = row['total_jobs']
        completed = row['completed']
        aborted = row['aborted']
        failed = row['failed']
        avg_hours = row['avg_exec_seconds'] / 3600 if row['avg_exec_seconds'] else 0
        abort_rate = (aborted / total * 100) if total > 0 else 0
        print(f"  {uuid:<24} Total: {total:>3}  Completed: {completed:>3}  Aborted: {aborted:>3}  Failed: {failed:>3}  Abort rate: {abort_rate:>5.1f}%  Avg: {avg_hours:>5.1f}h")

    # Look at temporal patterns
    print("\n" + "-" * 100)
    print("TEMPORAL PATTERNS:")
    print("-" * 100)

    # Parse start times for aborted jobs
    print("\nAborted jobs timeline:")
    aborted_with_time = df_aborted.select([
        "Job Name",
        "HTCondor Cluster ID",
        "Start Time",
        "Aborted Time",
        "Execution Duration (seconds)",
    ]).sort("Start Time")

    print(f"\nFirst aborted job: {aborted_with_time['Start Time'][0]}")
    print(f"Last aborted job:  {aborted_with_time['Start Time'][-1]}")

    # Group by date
    df_abort_dates = df_aborted.with_columns([
        pl.col("Start Time").str.slice(0, 10).alias("start_date")
    ]).group_by("start_date").agg([
        pl.len().alias("count")
    ]).sort("start_date")

    print("\nAborted jobs by date:")
    for row in df_abort_dates.iter_rows(named=True):
        print(f"  {row['start_date']}: {row['count']} aborted")

    # Look at job names - are specific epochs problematic?
    print("\n" + "-" * 100)
    print("JOB NAME PATTERNS:")
    print("-" * 100)

    abort_names = df_aborted.select("Job Name").to_series()
    # Extract epoch numbers
    import re
    epochs = []
    for name in abort_names:
        match = re.search(r'epoch(\d+)', name)
        if match:
            epochs.append(int(match.group(1)))

    if epochs:
        print(f"\nAborted job epochs: min={min(epochs)}, max={max(epochs)}, median={sorted(epochs)[len(epochs)//2]}")

        # Count by epoch
        from collections import Counter
        epoch_counts = Counter(epochs)
        print("\nMost problematic epochs:")
        for epoch, count in epoch_counts.most_common(10):
            print(f"  Epoch {epoch:>2}: {count} aborted jobs")

    # Check if it's always epoch 0
    print("\n" + "-" * 100)
    print("EPOCH 0 HYPOTHESIS:")
    print("-" * 100)

    epoch0_aborted = df_aborted.filter(pl.col("Job Name").str.contains("epoch0"))
    epoch0_completed = df_completed.filter(pl.col("Job Name").str.contains("epoch0"))

    print(f"\nEpoch 0 aborted:   {len(epoch0_aborted)}/{len(df_aborted)} ({len(epoch0_aborted)/len(df_aborted)*100:.1f}%)")
    print(f"Epoch 0 completed: {len(epoch0_completed)}/{len(df_completed)} ({len(epoch0_completed)/len(df_completed)*100:.1f}%)")

    # Look for patterns in the hung jobs
    print("\n" + "-" * 100)
    print("LONG-RUNNING ABORTED JOBS (>15 hours):")
    print("-" * 100)

    long_aborted = df_aborted.filter(
        pl.col("Execution Duration (seconds)") > 15 * 3600
    ).select([
        "Job Name",
        "HTCondor Cluster ID",
        "GPU UUID",
        "Start Time",
        "Aborted Time",
        "Execution Duration (human)",
    ]).sort("Execution Duration (seconds)", descending=True)

    print("\n")
    print(long_aborted)

    # Check if these share GPUs
    print("\n" + "-" * 100)
    print("SHARED GPU ANALYSIS:")
    print("-" * 100)

    long_abort_gpus = long_aborted.select("GPU UUID").unique().to_series()
    print(f"\n{len(long_abort_gpus)} unique GPUs involved in long aborts")

    for gpu_uuid in long_abort_gpus:
        gpu_jobs = df_h200.filter(pl.col("GPU UUID") == gpu_uuid)
        completed = len(gpu_jobs.filter(pl.col("Final Status") == "completed"))
        aborted = len(gpu_jobs.filter(pl.col("Final Status") == "aborted"))

        if completed > 0:
            print(f"\n  GPU {gpu_uuid[:20]}...")
            print(f"    Total jobs: {len(gpu_jobs)}")
            print(f"    Completed: {completed}")
            print(f"    Aborted: {aborted}")
            print(f"    → This GPU CAN complete jobs successfully!")

    print("\n" + "=" * 100)
    print("HYPOTHESIS:")
    print("=" * 100)
    print("""
The same GPUs that have aborted jobs also have successful completions.
This suggests the issue is NOT hardware-specific.

Likely causes:
1. Software bug that occurs occasionally (race condition, timing issue)
2. Network/filesystem issues during specific time periods
3. Initial training setup issues (epoch 0 problems)
4. Jobs getting stuck waiting for something (I/O, synchronization)

Next steps:
- Examine stdout/stderr from one of the long-running aborted jobs
- Compare to a successful job on the same GPU
- Look for patterns in what the job was doing when it hung
""")

    print("=" * 100)

    return df_h200


if __name__ == "__main__":
    df = main()
