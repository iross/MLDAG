#!/usr/bin/env python3
"""
Analyze OSPool jobs by GPU type to understand runtime variance.
"""

import polars as pl
import argparse
from pathlib import Path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze OSPool jobs by GPU type to understand runtime variance"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="NVIDIA H200",
        help="GPU device name to filter by (default: NVIDIA H200)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="full.csv",
        help="Path to CSV file (default: full.csv)"
    )
    args = parser.parse_args()

    gpu_type = args.gpu
    csv_file = args.csv

    # Load data
    df = pl.read_csv(csv_file)

    # Filter to OSPool jobs with specified GPU type
    df_filtered = df.filter(
        (pl.col("Targeted Resource") == "ospool") &
        (pl.col("GPU Device Name") == gpu_type)
    )

    print("=" * 80)
    print(f"OSPool {gpu_type} Jobs Analysis")
    print("=" * 80)
    print(f"\nTotal {gpu_type} OSPool jobs: {len(df_filtered)}")

    # Status breakdown
    print("\n" + "-" * 80)
    print("Job Status Distribution:")
    print("-" * 80)
    status_counts = df_filtered.group_by("Final Status").agg(pl.len().alias("count")).sort("count", descending=True)
    print(status_counts)

    # Filter to jobs with execution data
    df_executed = df_filtered.filter(pl.col("Execution Duration (seconds)").is_not_null())
    print(f"\nJobs with execution time data: {len(df_executed)}")

    # Execution time statistics
    print("\n" + "-" * 80)
    print("Execution Duration Statistics (hours):")
    print("-" * 80)
    exec_times = df_executed.select(pl.col("Execution Duration (seconds)") / 3600)
    print(f"Min:    {exec_times.min()[0, 0]:.1f}")
    print(f"Q1:     {exec_times.quantile(0.25)[0, 0]:.1f}")
    print(f"Median: {exec_times.median()[0, 0]:.1f}")
    print(f"Mean:   {exec_times.mean()[0, 0]:.1f}")
    print(f"Q3:     {exec_times.quantile(0.75)[0, 0]:.1f}")
    print(f"Max:    {exec_times.max()[0, 0]:.1f}")
    print(f"Std:    {exec_times.std()[0, 0]:.1f}")

    # By GLIDEIN resource
    print("\n" + "-" * 80)
    print("By GLIDEIN Resource Name:")
    print("-" * 80)
    glidein_stats = df_executed.group_by("GLIDEIN Resource Name").agg([
        pl.len().alias("count"),
        (pl.col("Execution Duration (seconds)") / 3600).min().alias("min_hours"),
        (pl.col("Execution Duration (seconds)") / 3600).median().alias("median_hours"),
        (pl.col("Execution Duration (seconds)") / 3600).mean().alias("mean_hours"),
        (pl.col("Execution Duration (seconds)") / 3600).max().alias("max_hours"),
        pl.col("Final Status").value_counts().alias("statuses"),
    ]).sort("mean_hours", descending=True)

    for row in glidein_stats.iter_rows(named=True):
        print(f"\n{row['GLIDEIN Resource Name']}:")
        print(f"  Count:  {row['count']}")
        print(f"  Min:    {row['min_hours']:>6.1f} hours")
        print(f"  Median: {row['median_hours']:>6.1f} hours")
        print(f"  Mean:   {row['mean_hours']:>6.1f} hours")
        print(f"  Max:    {row['max_hours']:>6.1f} hours")

    # By final status
    print("\n" + "-" * 80)
    print("By Final Status:")
    print("-" * 80)
    status_stats = df_executed.group_by("Final Status").agg([
        pl.len().alias("count"),
        (pl.col("Execution Duration (seconds)") / 3600).min().alias("min_hours"),
        (pl.col("Execution Duration (seconds)") / 3600).median().alias("median_hours"),
        (pl.col("Execution Duration (seconds)") / 3600).mean().alias("mean_hours"),
        (pl.col("Execution Duration (seconds)") / 3600).max().alias("max_hours"),
    ]).sort("mean_hours", descending=True)

    for row in status_stats.iter_rows(named=True):
        print(f"\n{row['Final Status']}:")
        print(f"  Count:  {row['count']}")
        print(f"  Min:    {row['min_hours']:>6.1f} hours")
        print(f"  Median: {row['median_hours']:>6.1f} hours")
        print(f"  Mean:   {row['mean_hours']:>6.1f} hours")
        print(f"  Max:    {row['max_hours']:>6.1f} hours")

    # Jobs > 15 hours
    print("\n" + "-" * 80)
    print("Jobs with execution time > 15 hours:")
    print("-" * 80)
    df_slow = df_executed.filter(pl.col("Execution Duration (seconds)") > 15 * 3600)
    print(f"Found {len(df_slow)} slow jobs")

    if len(df_slow) > 0:
        display_cols = [
            "Start Time",
            "Job Name",
            "HTCondor Cluster ID",
            "GPU ID",
            "Final Status",
            "GPU Usage",
            "CPU Usage",
            "GPU Memory Usage MB",
            "Execution Duration (seconds)",
            "Execution Duration (human)",
        ]
        print("\n")
        print(df_slow.select(display_cols).sort("Execution Duration (seconds)", descending=True))

        # Analyze GPU and CPU usage for slow jobs
        print("\n" + "-" * 80)
        print("Resource usage statistics for slow jobs (>15 hours):")
        print("-" * 80)
        slow_with_usage = df_slow.filter(pl.col("GPU Usage").is_not_null())
        if len(slow_with_usage) > 0:
            print(f"Slow jobs with usage data: {len(slow_with_usage)}")
            print(f"  Avg GPU Usage: {slow_with_usage.select(pl.col('GPU Usage').mean())[0, 0]:.2f}")
            print(f"  Avg CPU Usage: {slow_with_usage.select(pl.col('CPU Usage').mean())[0, 0]:.2f}")
            print(f"  Avg GPU Mem (MB): {slow_with_usage.select(pl.col('GPU Memory Usage MB').mean())[0, 0]:.0f}")
        else:
            print("No usage data available for slow jobs (likely aborted before termination)")

    # Jobs 8-12 hours (good range)
    print("\n" + "-" * 80)
    print("Jobs with execution time 8-12 hours (typical good range):")
    print("-" * 80)
    df_good = df_executed.filter(
        (pl.col("Execution Duration (seconds)") >= 8 * 3600) &
        (pl.col("Execution Duration (seconds)") <= 12 * 3600)
    )
    # print(f"Found {len(df_good)} jobs in good range")

    # if len(df_good) > 0:
    #     print("\n")
    #     good_display_cols = [
    #         "Job Name",
    #         "HTCondor Cluster ID",
    #         "Final Status",
    #         "GLIDEIN Resource Name",
    #         "Execution Duration (seconds)",
    #         "Execution Duration (human)",
    #         "Start Time",
    #     ]
    #     print(df_good.select(good_display_cols).sort("Execution Duration (seconds)"))

    # Completed jobs only
    print("\n" + "-" * 80)
    print("COMPLETED JOBS ONLY:")
    print("-" * 80)
    df_completed = df_executed.filter(pl.col("Final Status") == "completed")
    print(f"Completed jobs: {len(df_completed)}")

    if len(df_completed) > 0:
        exec_times_completed = df_completed.select(pl.col("Execution Duration (seconds)") / 3600)
        print(f"\nCompleted job execution times (hours):")
        print(f"  Min:    {exec_times_completed.min()[0, 0]:.1f}")
        print(f"  Median: {exec_times_completed.median()[0, 0]:.1f}")
        print(f"  Mean:   {exec_times_completed.mean()[0, 0]:.1f}")
        print(f"  Max:    {exec_times_completed.max()[0, 0]:.1f}")
        print(f"  Std:    {exec_times_completed.std()[0, 0]:.1f}")

        print("\nCompleted jobs resource usage:")
        completed_with_usage = df_completed.filter(pl.col("GPU Usage").is_not_null())
        if len(completed_with_usage) > 0:
            print(f"  Jobs with usage data: {len(completed_with_usage)}")
            print(f"  Avg GPU Usage: {completed_with_usage.select(pl.col('GPU Usage').mean())[0, 0]:.2f}")
            print(f"  Avg CPU Usage: {completed_with_usage.select(pl.col('CPU Usage').mean())[0, 0]:.2f}")
            print(f"  Avg GPU Mem (MB): {completed_with_usage.select(pl.col('GPU Memory Usage MB').mean())[0, 0]:.0f}")

        print("\nCompleted jobs distribution:")
        print(df_completed.select([
            "Start Time",
            "Job Name",
            "GLIDEIN Resource Name",
            "HTCondor Cluster ID",
            "GPU ID",
            "GPU Usage",
            "CPU Usage",
            "Execution Duration (seconds)",
            "Execution Duration (human)",
        ]).sort("Start Time").head(20))

    print("\n" + "=" * 80)
    return df_filtered


if __name__ == "__main__":
    df = main()
