#!/usr/bin/env python3
"""
Compare two HTCondor cluster IDs to understand differences in behavior.
"""

import polars as pl
import argparse


def compare_clusters(cluster1: int, cluster2: int, csv_file: str = "full.csv"):
    """Compare two cluster IDs and display differences."""
    df = pl.read_csv(csv_file)

    # Find both clusters
    row1 = df.filter(pl.col("HTCondor Cluster ID") == cluster1)
    row2 = df.filter(pl.col("HTCondor Cluster ID") == cluster2)

    if len(row1) == 0:
        print(f"Error: Cluster {cluster1} not found in {csv_file}")
        return
    if len(row2) == 0:
        print(f"Error: Cluster {cluster2} not found in {csv_file}")
        return

    # Get the rows as dictionaries
    r1 = row1.to_dicts()[0]
    r2 = row2.to_dicts()[0]

    print("=" * 100)
    print(f"CLUSTER COMPARISON: {cluster1} vs {cluster2}")
    print("=" * 100)

    # Key fields to compare
    fields = [
        ("Job Name", "Job Name"),
        ("DAG Source", "DAG Source"),
        ("Final Status", "Final Status"),
        ("Targeted Resource", "Targeted Resource"),
        ("GLIDEIN Resource Name", "GLIDEIN Resource Name"),
        ("GPU Device Name", "GPU Device Name"),
        ("GPU UUID", "GPU UUID"),
        ("GPU ID", "GPU ID"),
        ("GPU PCI Bus ID", "GPU PCI Bus ID"),
        ("Number of GPUs", "Number of GPUs"),
        ("Start Time", "Start Time"),
        ("End Time", "End Time"),
        ("Execution Duration (human)", "Execution Duration"),
        ("Execution Duration (seconds)", "Execution Duration (seconds)"),
        ("GPU Usage", "GPU Usage"),
        ("GPU Memory Usage MB", "GPU Memory Usage MB"),
        ("CPU Usage", "CPU Usage"),
        ("Peak Memory Usage MB", "Peak Memory Usage MB"),
        ("Disk Usage KB", "Disk Usage KB"),
        ("Total Bytes Sent", "Total Bytes Sent"),
        ("Total Bytes Received", "Total Bytes Received"),
        ("Transfer Input Duration (seconds)", "Transfer Input Duration (seconds)"),
        ("Held Time", "Held Time"),
        ("Released Time", "Released Time"),
        ("Evicted Time", "Evicted Time"),
        ("Aborted Time", "Aborted Time"),
    ]

    print("\n" + "-" * 100)
    print(f"{'Field':<35} {'Cluster ' + str(cluster1):<30} {'Cluster ' + str(cluster2):<30}")
    print("-" * 100)

    differences = []

    for field, display_name in fields:
        val1 = r1.get(field, "")
        val2 = r2.get(field, "")

        # Format values
        val1_str = str(val1) if val1 is not None else "None"
        val2_str = str(val2) if val2 is not None else "None"

        # Check if different
        is_different = val1 != val2
        marker = " **" if is_different else ""

        print(f"{display_name:<35} {val1_str:<30} {val2_str:<30}{marker}")

        if is_different and field not in ["Job Name", "Start Time", "End Time"]:
            differences.append((display_name, val1_str, val2_str))

    # Summary of key differences
    print("\n" + "=" * 100)
    print("KEY DIFFERENCES:")
    print("=" * 100)

    if differences:
        for field, val1, val2 in differences:
            print(f"\n{field}:")
            print(f"  Cluster {cluster1}: {val1}")
            print(f"  Cluster {cluster2}: {val2}")
    else:
        print("No significant differences found (other than timing)")

    # Special analysis
    print("\n" + "=" * 100)
    print("ANALYSIS:")
    print("=" * 100)

    # Check if same GPU
    if r1.get("GPU UUID") == r2.get("GPU UUID") and r1.get("GPU UUID"):
        print(f"\n✓ SAME PHYSICAL GPU: {r1.get('GPU UUID')}")
        print(f"  Both jobs ran on the same physical GPU device")

    # Check duration difference
    exec1 = r1.get("Execution Duration (seconds)")
    exec2 = r2.get("Execution Duration (seconds)")
    if exec1 and exec2:
        ratio = exec1 / exec2 if exec2 > 0 else 0
        print(f"\n✓ DURATION DIFFERENCE:")
        print(f"  Cluster {cluster1}: {exec1 / 3600:.1f} hours")
        print(f"  Cluster {cluster2}: {exec2 / 3600:.1f} hours")
        print(f"  Ratio: {ratio:.1f}x ({r1.get('Execution Duration (human)')} vs {r2.get('Execution Duration (human)')})")

    # Check resource usage
    if r1.get("Final Status") != r2.get("Final Status"):
        print(f"\n✓ STATUS DIFFERENCE:")
        print(f"  Cluster {cluster1}: {r1.get('Final Status')}")
        print(f"  Cluster {cluster2}: {r2.get('Final Status')}")

        if r1.get("Final Status") == "aborted" and r2.get("Final Status") == "completed":
            print(f"\n  → Cluster {cluster1} was ABORTED (likely hung/stalled)")
            print(f"  → Cluster {cluster2} COMPLETED successfully")
            print(f"  → No resource usage data for aborted job (aborted before termination event)")

    if r2.get("GPU Usage") is not None:
        print(f"\n✓ SUCCESSFUL JOB RESOURCE USAGE (Cluster {cluster2}):")
        print(f"  GPU Usage: {r2.get('GPU Usage')} ({float(r2.get('GPU Usage', 0)) * 100:.0f}% utilization)")
        print(f"  CPU Usage: {r2.get('CPU Usage')} cores")
        print(f"  GPU Memory: {r2.get('GPU Memory Usage MB')} MB")
        print(f"  Peak Memory: {r2.get('Peak Memory Usage MB')} MB")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two HTCondor cluster IDs"
    )
    parser.add_argument("cluster1", type=int, help="First cluster ID")
    parser.add_argument("cluster2", type=int, help="Second cluster ID")
    parser.add_argument("--csv", type=str, default="full.csv", help="CSV file path")

    args = parser.parse_args()

    compare_clusters(args.cluster1, args.cluster2, args.csv)


if __name__ == "__main__":
    main()
