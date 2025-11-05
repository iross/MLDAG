#!/usr/bin/env python3
"""
Compare transfer attempts by matching on end times.

This script automates the full workflow:
1. Uses find_slow_transfers.py to identify slow transfer cluster IDs
2. Uses query.py to fetch ES data for those clusters
3. Compares CSV and ES data to analyze coverage

Usage:
    python3 compare_transfer_attempts.py [--threshold SECONDS] [--input CSV_FILE] [--cluster CLUSTER_ID]

Examples:
    # Use default threshold (300s) and input file (full.csv)
    python3 compare_transfer_attempts.py

    # Custom threshold of 10 minutes
    python3 compare_transfer_attempts.py --threshold 600

    # Custom input file and threshold
    python3 compare_transfer_attempts.py --threshold 1800 --input job_summary.csv

    # Analyze specific cluster ID
    python3 compare_transfer_attempts.py --cluster 12345
"""

import sys
import csv
import argparse
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from query import ES_HOST, ES_USER, ES_PASS, ES_INDEX, get_query


def parse_timestamp(ts_str):
    """Parse ISO timestamp string to datetime."""
    if not ts_str or ts_str.strip() == '':
        return None
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None


def read_csv_attempts(job_ids, csv_file="full.csv"):
    """Read all transfer attempts from CSV for given job IDs.

    Returns:
        Dictionary mapping cluster_id -> list of attempts
        Each attempt is a dict with: start_time, end_time, duration, attempt_seq
    """
    attempts = {}
    job_id_set = set(job_ids)

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = row.get('HTCondor Cluster ID', '').strip()

            if cluster_id not in job_id_set:
                continue

            # Get transfer start and end times and duration
            start_time_str = row.get('Transfer Input Start Time', '').strip()
            end_time_str = row.get('Transfer Input End Time', '').strip()
            duration_str = row.get('Transfer Input Duration (seconds)', '').strip()
            attempt_seq = row.get('Attempt Sequence', '').strip()

            start_time = parse_timestamp(start_time_str)
            end_time = parse_timestamp(end_time_str)

            if start_time and end_time and duration_str:
                try:
                    duration = int(duration_str)
                    if duration > 0:
                        if cluster_id not in attempts:
                            attempts[cluster_id] = []

                        attempts[cluster_id].append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration,
                            'attempt_seq': attempt_seq,
                        })
                except ValueError:
                    pass

    return attempts


def query_es_attempts(job_ids):
    """Query ES and get all transfer attempts with timestamps.

    Returns:
        Dictionary mapping cluster_id -> list of ES attempts
        Each attempt is a dict with: start_time, end_time, attempt_time, attempt_num, transfer_url
    """
    client = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS))
    query = get_query(job_ids)

    es_attempts = {}

    print("Querying Elasticsearch...")
    for doc in scan(client=client, query=query.pop("body"), **query):
        source = doc["_source"]
        cluster_id = str(source.get("ClusterId", ""))

        # Parse attempt end time
        end_time_ts = source.get("AttemptEndTime")
        if end_time_ts:
            try:
                # ES timestamps are in seconds since epoch
                end_time = datetime.fromtimestamp(end_time_ts)
            except (ValueError, TypeError):
                continue
        else:
            continue

        attempt_time = source.get("AttemptTime", 0)
        attempt_num = source.get("Attempt", 0)
        transfer_url = source.get("TransferUrl", "")

        # Calculate start time from end time and duration
        start_time = end_time - timedelta(seconds=attempt_time) if attempt_time else end_time

        if cluster_id and attempt_time:
            if cluster_id not in es_attempts:
                es_attempts[cluster_id] = []

            es_attempts[cluster_id].append({
                'start_time': start_time,
                'end_time': end_time,
                'attempt_time': attempt_time,
                'attempt_num': attempt_num,
                'transfer_url': transfer_url,
            })

    return es_attempts


def match_attempts(csv_attempts, es_attempts, time_tolerance_seconds=300):
    """Match CSV attempts to ES attempts based on end time proximity and filter by time window.

    Args:
        csv_attempts: Dict of cluster_id -> list of CSV attempts
        es_attempts: Dict of cluster_id -> list of ES attempts
        time_tolerance_seconds: Max time difference to consider a match (default 5 min)

    Returns:
        List of matched attempt pairs with comparison data
    """
    matches = []

    for cluster_id in csv_attempts.keys():
        csv_atts = csv_attempts[cluster_id]
        es_atts = es_attempts.get(cluster_id, [])

        if not es_atts:
            # No ES data for this cluster
            for csv_att in csv_atts:
                matches.append({
                    'cluster_id': cluster_id,
                    'csv_attempt': csv_att,
                    'es_attempts_in_window': [],
                    'matched': False,
                })
            continue

        # Try to match each CSV attempt to ES attempts
        for csv_att in csv_atts:
            csv_start = csv_att['start_time']
            csv_end = csv_att['end_time']

            # Find ALL ES attempts that fall within the CSV transfer window
            es_in_window = []
            for es_att in es_atts:
                es_end = es_att['end_time']
                # Check if ES attempt end time is within the CSV transfer window
                if csv_start <= es_end <= csv_end:
                    es_in_window.append(es_att)

            if es_in_window:
                # Found ES attempts in the window!
                matches.append({
                    'cluster_id': cluster_id,
                    'csv_attempt': csv_att,
                    'es_attempts_in_window': es_in_window,
                    'matched': True,
                })
            else:
                # No ES attempts found in the transfer window
                matches.append({
                    'cluster_id': cluster_id,
                    'csv_attempt': csv_att,
                    'es_attempts_in_window': [],
                    'matched': False,
                })

    return matches


def merge_time_ranges(ranges):
    """Merge overlapping time ranges and return total duration.

    Args:
        ranges: List of (start_time, end_time) tuples

    Returns:
        Total duration in seconds covered by the union of all ranges
    """
    if not ranges:
        return 0

    # Sort by start time
    sorted_ranges = sorted(ranges, key=lambda x: x[0])

    merged = []
    current_start, current_end = sorted_ranges[0]

    for start, end in sorted_ranges[1:]:
        if start <= current_end:
            # Overlapping or adjacent - merge
            current_end = max(current_end, end)
        else:
            # No overlap - save current and start new range
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    # Don't forget the last range
    merged.append((current_start, current_end))

    # Calculate total duration
    total_seconds = sum((end - start).total_seconds() for start, end in merged)
    return total_seconds


def print_report(matches):
    """Print detailed comparison report."""

    print("=" * 120)
    print("TRANSFER ATTEMPT MATCHING REPORT (ES records filtered by CSV transfer window)")
    print("=" * 120)
    print()

    # Summary statistics
    total_csv_attempts = len(matches)
    matched_attempts = sum(1 for m in matches if m['matched'])
    unmatched_attempts = total_csv_attempts - matched_attempts

    print(f"Total CSV attempts:     {total_csv_attempts}")
    print(f"Matched to ES:          {matched_attempts} ({matched_attempts/total_csv_attempts*100:.1f}%)")
    print(f"Not matched:            {unmatched_attempts}")
    print()

    if matched_attempts == 0:
        print("No matches found between CSV and ES data.")
        return

    # Calculate coverage for matched attempts
    matched_only = [m for m in matches if m['matched']]

    total_csv_time = sum(m['csv_attempt']['duration'] for m in matched_only)

    # Calculate total ES time as union of all time ranges (handling overlaps)
    all_es_ranges = []
    for m in matched_only:
        for es_att in m['es_attempts_in_window']:
            all_es_ranges.append((es_att['start_time'], es_att['end_time']))

    total_es_time = merge_time_ranges(all_es_ranges)

    overall_coverage = (total_es_time / total_csv_time * 100) if total_csv_time > 0 else 0

    print(f"Matched attempts:")
    print(f"  Total transfer time (CSV):        {total_csv_time:,} seconds ({total_csv_time/3600:.2f} hours)")
    print(f"  Total AttemptTime (ES, union):    {total_es_time:,.0f} seconds ({total_es_time/3600:.2f} hours)")
    print(f"  Overall coverage:                 {overall_coverage:.1f}%")
    print()

    # Per-attempt statistics (using union of time ranges)
    coverages = []
    for m in matched_only:
        csv_dur = m['csv_attempt']['duration']
        if csv_dur > 0:
            # Calculate union of ES time ranges for this attempt
            es_ranges = [(es_att['start_time'], es_att['end_time'])
                         for es_att in m['es_attempts_in_window']]
            es_total = merge_time_ranges(es_ranges)
            coverages.append((es_total / csv_dur) * 100)

    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        min_coverage = min(coverages)
        max_coverage = max(coverages)

        print(f"Per-attempt coverage (union of ES ranges):")
        print(f"  Average: {avg_coverage:.1f}%")
        print(f"  Min:     {min_coverage:.1f}%")
        print(f"  Max:     {max_coverage:.1f}%")
        print()

    # ES records per CSV attempt
    es_counts = [len(m['es_attempts_in_window']) for m in matched_only]
    if es_counts:
        avg_es_count = sum(es_counts) / len(es_counts)
        max_es_count = max(es_counts)
        total_es_records = sum(es_counts)

        print(f"ES records within transfer windows:")
        print(f"  Total ES records:  {total_es_records}")
        print(f"  Average per CSV:   {avg_es_count:.1f}")
        print(f"  Max per CSV:       {max_es_count}")
        print()

    # Detailed breakdown
    print("=" * 120)
    print("DETAILED ATTEMPT BREAKDOWN")
    print("=" * 120)
    print()
    print(f"{'Cluster ID':<12} {'Att':<4} {'CSV Start':<20} {'CSV End':<20} "
          f"{'CSV (s)':<10} {'ES Recs':<8} {'ES Union (s)':<12} {'Coverage':<10}")
    print("-" * 120)

    # Sort by cluster ID and CSV end time
    sorted_matches = sorted(matches, key=lambda m: (m['cluster_id'], m['csv_attempt']['end_time']))

    for m in sorted_matches:
        cluster_id = m['cluster_id']
        csv_att = m['csv_attempt']
        es_atts = m['es_attempts_in_window']

        csv_start = csv_att['start_time'].strftime('%Y-%m-%d %H:%M:%S')
        csv_end = csv_att['end_time'].strftime('%Y-%m-%d %H:%M:%S')
        csv_dur = csv_att['duration']
        att_seq = csv_att['attempt_seq']

        if m['matched']:
            es_count = len(es_atts)
            # Calculate union of ES time ranges
            es_ranges = [(es_att['start_time'], es_att['end_time']) for es_att in es_atts]
            es_total = merge_time_ranges(es_ranges)
            coverage = f"{(es_total/csv_dur)*100:.1f}%"

            print(f"{cluster_id:<12} {att_seq:<4} {csv_start:<20} {csv_end:<20} "
                  f"{csv_dur:<10,} {es_count:<8} {es_total:<12,.0f} {coverage:<10}")
        else:
            print(f"{cluster_id:<12} {att_seq:<4} {csv_start:<20} {csv_end:<20} "
                  f"{csv_dur:<10,} {'0':<8} {'-':<12} {'-':<10}")

    print()
    print("=" * 120)


def print_cluster_comparison(csv_attempts, es_attempts):
    """Print detailed comparison by cluster ID showing all records from both sources."""

    print("=" * 140)
    print("CLUSTER-LEVEL COMPARISON (All records from both sources)")
    print("=" * 140)
    print()

    # Get all cluster IDs from both sources
    all_cluster_ids = sorted(set(list(csv_attempts.keys()) + list(es_attempts.keys())))

    for cluster_id in all_cluster_ids:
        csv_atts = csv_attempts.get(cluster_id, [])
        es_atts = es_attempts.get(cluster_id, [])

        print(f"Cluster ID: {cluster_id}")
        print("-" * 140)

        # CSV Records
        print(f"  CSV Records ({len(csv_atts)}):")
        if csv_atts:
            print(f"    {'Att':<4} {'Start Time':<17} {'End Time':<17} {'Duration (s)':<15}")
            for att in sorted(csv_atts, key=lambda x: x['end_time']):
                start = att['start_time'].strftime('%m-%d %H:%M:%S')
                end = att['end_time'].strftime('%m-%d %H:%M:%S')
                duration = att['duration']
                seq = att['attempt_seq']
                print(f"    {seq:<4} {start:<17} {end:<17} {duration:<15,}")
        else:
            print(f"    (No CSV records)")
        print()

        # ES Records
        print(f"  ES Records ({len(es_atts)}):")
        if es_atts:
            # Detect duplicates
            seen = {}
            duplicates = []
            for att in es_atts:
                key = (att['end_time'], att['attempt_time'], att['attempt_num'])
                if key in seen:
                    duplicates.append(att)
                else:
                    seen[key] = att

            # Determine which ES records fall within CSV transfer windows
            es_in_windows = set()
            for csv_att in csv_atts:
                csv_start = csv_att['start_time']
                csv_end = csv_att['end_time']
                for i, es_att in enumerate(es_atts):
                    if csv_start <= es_att['end_time'] <= csv_end:
                        es_in_windows.add(i)

            print(f"    {'Att#':<5} {'Start Time':<17} {'End Time':<17} {'AttemptTime (s)':<16} {'File':<40} {'Status':<25}")
            for i, att in enumerate(sorted(es_atts, key=lambda x: (x['end_time'], x['attempt_num']))):
                start = att['start_time'].strftime('%m-%d %H:%M:%S')
                end = att['end_time'].strftime('%m-%d %H:%M:%S')
                attempt_time = att['attempt_time']
                attempt_num = att['attempt_num']

                # Extract filename from URL
                transfer_url = att.get('transfer_url', '')
                if transfer_url:
                    # Extract filename from URL path
                    filename = transfer_url.split('/')[-1]
                else:
                    filename = ''

                # Build status notes
                notes = []

                # Check if this is a duplicate
                key = (att['end_time'], att['attempt_time'], att['attempt_num'])
                is_dup = sum(1 for a in es_atts if (a['end_time'], a['attempt_time'], a['attempt_num']) == key) > 1
                if is_dup:
                    notes.append("DUPLICATE")

                # Check if within CSV window
                # Need to find this att in the sorted list
                sorted_atts = sorted(es_atts, key=lambda x: (x['end_time'], x['attempt_num']))
                original_idx = es_atts.index(att)
                if original_idx not in es_in_windows:
                    notes.append("OUTSIDE WINDOW")

                status = ", ".join(notes) if notes else "In window"

                print(f"    {attempt_num:<5} {start:<17} {end:<17} {attempt_time:<16,.1f} {filename:<40} {status:<25}")

            if duplicates:
                print(f"    WARNING: Found {len(duplicates)} exact duplicate ES record(s)")

            outside_count = len(es_atts) - len(es_in_windows)
            if outside_count > 0:
                print(f"    WARNING: {outside_count} ES record(s) fall outside CSV transfer window(s)")
        else:
            print(f"    (No ES records)")
        print()

        # Summary for this cluster
        csv_total = sum(att['duration'] for att in csv_atts)

        # Calculate ES total as simple sum (for comparison)
        es_total_sum = sum(att['attempt_time'] for att in es_atts)

        # Calculate ES total as union of all ranges
        all_es_ranges = [(es_att['start_time'], es_att['end_time']) for es_att in es_atts]
        es_total_union = merge_time_ranges(all_es_ranges)

        # Calculate ES total within windows (union of ranges that fall in windows)
        es_ranges_in_window = []
        for csv_att in csv_atts:
            csv_start = csv_att['start_time']
            csv_end = csv_att['end_time']
            for es_att in es_atts:
                if csv_start <= es_att['end_time'] <= csv_end:
                    es_ranges_in_window.append((es_att['start_time'], es_att['end_time']))

        es_total_in_window_union = merge_time_ranges(es_ranges_in_window)

        print(f"  Summary:")
        print(f"    CSV total time:                {csv_total:,} seconds ({csv_total/3600:.2f} hours)")
        print(f"    ES total (sum of all):         {es_total_sum:,.0f} seconds ({es_total_sum/3600:.2f} hours)")
        print(f"    ES total (union of all):       {es_total_union:,.0f} seconds ({es_total_union/3600:.2f} hours)")
        print(f"    ES total (union in window):    {es_total_in_window_union:,.0f} seconds ({es_total_in_window_union/3600:.2f} hours)")
        if csv_total > 0:
            coverage_sum = (es_total_sum / csv_total) * 100
            coverage_union = (es_total_union / csv_total) * 100
            coverage_window_union = (es_total_in_window_union / csv_total) * 100
            print(f"    ES/CSV ratio (sum):            {coverage_sum:.1f}%")
            print(f"    ES/CSV ratio (union all):      {coverage_union:.1f}%")
            print(f"    ES/CSV ratio (union in win):   {coverage_window_union:.1f}%")
        print()
        print()


def find_slow_transfers(threshold, input_file):
    """Run find_slow_transfers.py to get cluster IDs.

    Args:
        threshold: Transfer duration threshold in seconds
        input_file: CSV file to search

    Returns:
        List of cluster IDs
    """
    print(f"Finding slow transfers (threshold: {threshold}s, file: {input_file})...")

    # Run find_slow_transfers.py
    result = subprocess.run(
        ['python3', 'find_slow_transfers.py', str(threshold), input_file],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR running find_slow_transfers.py: {result.stderr}")
        sys.exit(1)

    # Parse cluster IDs from stdout (one per line)
    cluster_ids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]

    # Print the summary that was sent to stderr
    if result.stderr:
        print(result.stderr.strip())

    return cluster_ids


def fetch_es_data(job_ids):
    """Query Elasticsearch and return CSV data as a temporary file.

    Args:
        job_ids: List of cluster IDs to query

    Returns:
        Path to temporary CSV file with ES data
    """
    print(f"Querying Elasticsearch for {len(job_ids)} cluster IDs...")

    client = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS))
    query = get_query(job_ids)

    # Create temporary file for ES data
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')

    # Write CSV header
    fields = query['body']['_source']
    temp_file.write(','.join(fields) + '\n')

    # Fetch and write data
    doc_count = 0
    for doc in scan(client=client, query=query.pop("body"), **query):
        source = doc["_source"]
        row = [str(source.get(field, "UNKNOWN")) for field in fields]
        temp_file.write(','.join(row) + '\n')
        doc_count += 1

    temp_file.close()
    print(f"Fetched {doc_count} ES records")

    return temp_file.name


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compare transfer attempts between CSV and Elasticsearch",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--threshold', type=int, default=300,
                       help='Transfer duration threshold in seconds (default: 300)')
    parser.add_argument('--input', default='full.csv',
                       help='Input CSV file to analyze (default: full.csv)')
    parser.add_argument('--cluster',
                       help='Analyze specific cluster ID only')

    args = parser.parse_args()

    # Print header
    if args.cluster:
        print("=" * 100)
        print(f"TRANSFER ATTEMPT ANALYSIS FOR CLUSTER {args.cluster}")
        print("=" * 100)
        print()
        job_ids = [args.cluster]
    else:
        print("=" * 100)
        print("AUTOMATED TRANSFER ATTEMPT MATCHING ANALYSIS")
        print("=" * 100)
        print()

        # Step 1: Find slow transfers
        job_ids = find_slow_transfers(args.threshold, args.input)
        if not job_ids:
            print("No slow transfers found. Exiting.")
            sys.exit(0)

    print()

    # Step 2: Read CSV attempts
    print(f"Reading CSV attempts from {args.input}...")
    csv_attempts = read_csv_attempts(job_ids, args.input)
    total_csv_attempts = sum(len(atts) for atts in csv_attempts.values())
    print(f"Found {total_csv_attempts} CSV attempts across {len(csv_attempts)} jobs")
    print()

    # Step 3: Query ES attempts
    print("Querying Elasticsearch for transfer attempts...")
    es_attempts = query_es_attempts(job_ids)
    total_es_attempts = sum(len(atts) for atts in es_attempts.values())
    print(f"Found {total_es_attempts} ES attempts across {len(es_attempts)} jobs")
    print()

    # If specific cluster requested and found no data, exit
    if args.cluster and not csv_attempts and not es_attempts:
        print(f"ERROR: No data found for cluster {args.cluster}")
        print(f"This cluster may not exist in {args.input} or Elasticsearch.")
        sys.exit(1)

    # Step 4: Match attempts
    print("Matching attempts by end time...")
    matches = match_attempts(csv_attempts, es_attempts)
    print()

    # Step 5: Print results
    # For specific cluster, just show cluster comparison
    if args.cluster:
        print_cluster_comparison(csv_attempts, es_attempts)
    else:
        # Print window-filtered report
        print_report(matches)

        # Print cluster-level comparison
        print()
        print()
        print_cluster_comparison(csv_attempts, es_attempts)


if __name__ == "__main__":
    main()
