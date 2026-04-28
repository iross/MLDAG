#!/usr/bin/env python3
"""
Add val_loss column to full.csv by matching checkpoint files to job attempts.

Parses global_checkpoint_files to extract val_loss values from checkpoint filenames,
then joins them to CSV rows using (run_uuid, epoch) as the key.
"""

import csv
import re
import argparse
from collections import defaultdict
from pathlib import Path


def parse_checkpoint_files(checkpoint_list_path: str) -> dict[tuple[str, int], float]:
    """Parse the checkpoint file listing to build a (run_uuid, epoch) -> val_loss map.

    Checkpoint paths look like:
        .../training_logs/<run_uuid>/checkpoints/epoch=<N>-step=<M>-val_loss=<V>.ckpt
    """
    val_loss_map = {}

    with open(checkpoint_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract run_uuid from path
            uuid_match = re.search(r'training_logs/([^/]+)/checkpoints/', line)
            if not uuid_match:
                continue

            run_uuid = uuid_match.group(1)

            # Extract epoch and val_loss from filename
            ckpt_match = re.search(r'epoch=(\d+)-step=\d+-val_loss=([\d.]+)\.ckpt$', line)
            if not ckpt_match:
                continue

            epoch = int(ckpt_match.group(1))
            val_loss = float(ckpt_match.group(2))

            key = (run_uuid, epoch)
            # If multiple checkpoints exist for same (uuid, epoch), keep the latest (last seen)
            val_loss_map[key] = val_loss

    return val_loss_map


def extract_epoch_from_job_name(job_name: str) -> int | None:
    """Extract the epoch number from a job name like 'run0-train_epoch5'."""
    match = re.search(r'_epoch(\d+)$', job_name)
    if match:
        return int(match.group(1))
    return None


def add_val_loss_column(input_csv: str, checkpoint_list: str, output_csv: str):
    """Read input CSV, add Val Loss column from checkpoint data, write output CSV."""
    val_loss_map = parse_checkpoint_files(checkpoint_list)
    print(f"Parsed {len(val_loss_map)} checkpoint entries from {checkpoint_list}")

    matched = 0
    unmatched = 0

    with open(input_csv, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['Val Loss']

        with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for row in reader:
                run_uuid = row.get('Run UUID', '')
                job_name = row.get('Job Name', '')
                epoch = extract_epoch_from_job_name(job_name)

                val_loss = ''
                if run_uuid and epoch is not None:
                    key = (run_uuid, epoch)
                    if key in val_loss_map:
                        val_loss = val_loss_map[key]
                        matched += 1
                    else:
                        unmatched += 1

                row['Val Loss'] = val_loss
                writer.writerow(row)

    print(f"Matched {matched} rows with val_loss, {unmatched} rows without a checkpoint match")
    print(f"Output written to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Add val_loss from checkpoint filenames to experiment CSV"
    )
    parser.add_argument("--input", "-i", default="full.csv", help="Input CSV file (default: full.csv)")
    parser.add_argument("--checkpoints", "-c", default="global_checkpoint_files",
                        help="File listing checkpoint paths (default: global_checkpoint_files)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV file (default: overwrite input)")

    args = parser.parse_args()
    output = args.output if args.output else args.input

    add_val_loss_column(args.input, args.checkpoints, output)


if __name__ == "__main__":
    main()
