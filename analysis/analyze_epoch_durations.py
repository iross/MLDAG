#!/usr/bin/env python3
"""
Analyze HTCondor training epoch durations to identify patterns in runtime variance.

This script loads the full.csv data and performs clustering analysis to understand
why some epochs take 10 hours while others take 40+ hours.
"""

import polars as pl
import sys
from pathlib import Path


def load_data(csv_path: str = "full.csv") -> pl.DataFrame:
    """Load the HTCondor job data from CSV."""
    df = pl.read_csv(csv_path)
    print(f"Loaded {len(df)} jobs from {csv_path}")
    print(f"Columns: {df.columns}")
    return df


def print_basic_stats(df: pl.DataFrame) -> None:
    """Print basic statistics about the dataset."""
    print("\n" + "=" * 80)
    print("BASIC DATASET STATISTICS")
    print("=" * 80)

    print(f"\nTotal jobs: {len(df)}")
    print(f"\nJob status distribution:")
    print(df.group_by("Final Status").agg(pl.count()).sort("count", descending=True))

    print(f"\nTargeted resource distribution:")
    print(df.group_by("Targeted Resource").agg(pl.count()).sort("count", descending=True))

    print(f"\nGPU device distribution:")
    print(df.group_by("GPU Device Name").agg(pl.count()).sort("count", descending=True))


def analyze_execution_durations(df: pl.DataFrame) -> None:
    """Analyze execution durations across different dimensions."""
    print("\n" + "=" * 80)
    print("EXECUTION DURATION ANALYSIS")
    print("=" * 80)

    # Filter to only jobs with actual execution time
    df_executed = df.filter(pl.col("Execution Duration (seconds)").is_not_null())

    print(f"\nJobs with execution data: {len(df_executed)}")

    # Overall execution time statistics
    exec_times = df_executed.select("Execution Duration (seconds)")
    print(f"\nOverall execution time statistics (seconds):")
    print(f"  Min:    {exec_times.min()[0, 0]:>10.1f} ({exec_times.min()[0, 0] / 3600:.1f} hours)")
    print(f"  Median: {exec_times.median()[0, 0]:>10.1f} ({exec_times.median()[0, 0] / 3600:.1f} hours)")
    print(f"  Mean:   {exec_times.mean()[0, 0]:>10.1f} ({exec_times.mean()[0, 0] / 3600:.1f} hours)")
    print(f"  Max:    {exec_times.max()[0, 0]:>10.1f} ({exec_times.max()[0, 0] / 3600:.1f} hours)")
    print(f"  Std:    {exec_times.std()[0, 0]:>10.1f} ({exec_times.std()[0, 0] / 3600:.1f} hours)")


def analyze_by_gpu(df: pl.DataFrame) -> None:
    """Group by GPU type and analyze execution durations."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY GPU DEVICE")
    print("=" * 80)

    # Filter to jobs with execution time and GPU info
    df_gpu = df.filter(
        pl.col("Execution Duration (seconds)").is_not_null() &
        pl.col("GPU Device Name").is_not_null()
    )

    gpu_stats = df_gpu.group_by("GPU Device Name").agg([
        pl.count().alias("job_count"),
        pl.col("Execution Duration (seconds)").min().alias("min_duration"),
        pl.col("Execution Duration (seconds)").median().alias("median_duration"),
        pl.col("Execution Duration (seconds)").mean().alias("mean_duration"),
        pl.col("Execution Duration (seconds)").max().alias("max_duration"),
        pl.col("Execution Duration (seconds)").std().alias("std_duration"),
        pl.col("Final Status").value_counts().alias("status_counts"),
        pl.col("Number of GPUs").mean().alias("avg_gpu_count"),
    ]).sort("mean_duration", descending=True)

    print("\nExecution time by GPU type (sorted by mean duration):")
    for row in gpu_stats.iter_rows(named=True):
        print(f"\n{row['GPU Device Name']}:")
        print(f"  Jobs:         {row['job_count']}")
        print(f"  Avg # GPUs:   {row['avg_gpu_count']:.1f}")
        print(f"  Min duration: {row['min_duration'] / 3600:>6.1f} hours")
        print(f"  Med duration: {row['median_duration'] / 3600:>6.1f} hours")
        print(f"  Mean duration:{row['mean_duration'] / 3600:>6.1f} hours")
        print(f"  Max duration: {row['max_duration'] / 3600:>6.1f} hours")
        print(f"  Std duration: {row['std_duration'] / 3600:>6.1f} hours")


def analyze_by_resource(df: pl.DataFrame) -> None:
    """Group by targeted resource and analyze execution durations."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY TARGETED RESOURCE")
    print("=" * 80)

    df_resource = df.filter(
        pl.col("Execution Duration (seconds)").is_not_null() &
        pl.col("Targeted Resource").is_not_null()
    )

    resource_stats = df_resource.group_by("Targeted Resource").agg([
        pl.count().alias("job_count"),
        pl.col("Execution Duration (seconds)").min().alias("min_duration"),
        pl.col("Execution Duration (seconds)").median().alias("median_duration"),
        pl.col("Execution Duration (seconds)").mean().alias("mean_duration"),
        pl.col("Execution Duration (seconds)").max().alias("max_duration"),
        pl.col("Execution Duration (seconds)").std().alias("std_duration"),
        pl.col("Final Status").value_counts().alias("status_counts"),
    ]).sort("mean_duration", descending=True)

    print("\nExecution time by resource (sorted by mean duration):")
    for row in resource_stats.iter_rows(named=True):
        print(f"\n{row['Targeted Resource']}:")
        print(f"  Jobs:         {row['job_count']}")
        print(f"  Min duration: {row['min_duration'] / 3600:>6.1f} hours")
        print(f"  Med duration: {row['median_duration'] / 3600:>6.1f} hours")
        print(f"  Mean duration:{row['mean_duration'] / 3600:>6.1f} hours")
        print(f"  Max duration: {row['max_duration'] / 3600:>6.1f} hours")
        print(f"  Std duration: {row['std_duration'] / 3600:>6.1f} hours")


def analyze_by_status(df: pl.DataFrame) -> None:
    """Analyze execution durations by final job status."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY FINAL STATUS")
    print("=" * 80)

    df_status = df.filter(
        pl.col("Execution Duration (seconds)").is_not_null() &
        pl.col("Final Status").is_not_null()
    )

    status_stats = df_status.group_by("Final Status").agg([
        pl.count().alias("job_count"),
        pl.col("Execution Duration (seconds)").min().alias("min_duration"),
        pl.col("Execution Duration (seconds)").median().alias("median_duration"),
        pl.col("Execution Duration (seconds)").mean().alias("mean_duration"),
        pl.col("Execution Duration (seconds)").max().alias("max_duration"),
        pl.col("Execution Duration (seconds)").std().alias("std_duration"),
    ]).sort("mean_duration", descending=True)

    print("\nExecution time by final status (sorted by mean duration):")
    for row in status_stats.iter_rows(named=True):
        print(f"\n{row['Final Status']}:")
        print(f"  Jobs:         {row['job_count']}")
        print(f"  Min duration: {row['min_duration'] / 3600:>6.1f} hours")
        print(f"  Med duration: {row['median_duration'] / 3600:>6.1f} hours")
        print(f"  Mean duration:{row['mean_duration'] / 3600:>6.1f} hours")
        print(f"  Max duration: {row['max_duration'] / 3600:>6.1f} hours")
        print(f"  Std duration: {row['std_duration'] / 3600:>6.1f} hours")


def analyze_outliers(df: pl.DataFrame) -> None:
    """Identify and analyze execution time outliers."""
    print("\n" + "=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)

    df_executed = df.filter(pl.col("Execution Duration (seconds)").is_not_null())

    # Define outliers as jobs > 40000 seconds (>11.1 hours)
    threshold_seconds = 40000
    df_outliers = df_executed.filter(
        pl.col("Execution Duration (seconds)") > threshold_seconds
    )

    print(f"\nJobs with execution time > {threshold_seconds / 3600:.1f} hours:")
    print(f"Found {len(df_outliers)} outlier jobs")

    if len(df_outliers) > 0:
        # Display key info about outliers
        outlier_cols = [
            "Job Name",
            "Final Status",
            "Targeted Resource",
            "GLIDEIN Resource Name",
            "GPU Device Name",
            "Number of GPUs",
            "Execution Duration (seconds)",
            "Execution Duration (human)",
        ]

        print("\nOutlier job details:")
        print(df_outliers.select(outlier_cols).sort("Execution Duration (seconds)", descending=True))

        # Summary stats for outliers
        print("\nOutlier summary by GPU:")
        print(df_outliers.group_by("GPU Device Name").agg([
            pl.count().alias("count"),
            pl.col("Final Status").value_counts().alias("statuses"),
        ]).sort("count", descending=True))


def analyze_transfer_time_correlation(df: pl.DataFrame) -> None:
    """Analyze correlation between data transfer time and execution duration."""
    print("\n" + "=" * 80)
    print("TRANSFER TIME CORRELATION ANALYSIS")
    print("=" * 80)

    df_transfer = df.filter(
        pl.col("Execution Duration (seconds)").is_not_null() &
        pl.col("Transfer Input Duration (seconds)").is_not_null()
    )

    print(f"\nJobs with both execution and transfer data: {len(df_transfer)}")

    # Compute correlation
    corr = df_transfer.select([
        pl.corr("Transfer Input Duration (seconds)", "Execution Duration (seconds)").alias("correlation")
    ])
    print(f"\nCorrelation between transfer time and execution time: {corr[0, 0]:.3f}")

    # Show jobs with long transfer times
    long_transfer_threshold = 1000  # 16.7 minutes
    df_long_transfer = df_transfer.filter(
        pl.col("Transfer Input Duration (seconds)") > long_transfer_threshold
    )

    print(f"\nJobs with transfer time > {long_transfer_threshold / 60:.1f} minutes: {len(df_long_transfer)}")

    if len(df_long_transfer) > 0:
        print("\nLong transfer time jobs:")
        print(df_long_transfer.select([
            "Job Name",
            "Targeted Resource",
            "GPU Device Name",
            "Transfer Input Duration (seconds)",
            "Execution Duration (seconds)",
            "Final Status",
        ]).sort("Transfer Input Duration (seconds)", descending=True).head(10))


def main():
    """Main analysis function."""
    # Check if CSV file exists
    csv_path = Path("full.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        sys.exit(1)

    # Load data
    df = load_data(str(csv_path))

    # Run analyses
    print_basic_stats(df)
    analyze_execution_durations(df)
    analyze_by_gpu(df)
    analyze_by_resource(df)
    analyze_by_status(df)
    analyze_outliers(df)
    analyze_transfer_time_correlation(df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Return the dataframe for interactive use
    return df


if __name__ == "__main__":
    df = main()
