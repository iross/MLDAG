#!/usr/bin/env python3
"""Parse metl.log and print an ASCII histogram of job resource usage."""

import argparse
import re
import sys
from datetime import datetime

parser = argparse.ArgumentParser(description="Histogram of HTCondor job resource usage.")
parser.add_argument("log", nargs="?", default="metl.log")
parser.add_argument(
    "--metric",
    choices=["time", "disk", "memory"],
    default="time",
    help="metric to histogram: time (hours), disk (GB), memory (GB) (default: time)",
)
group = parser.add_mutually_exclusive_group()
group.add_argument("--bins", type=int, default=20, metavar="N", help="number of bins (default: 20)")
group.add_argument("--bin-size", type=float, metavar="S", help="bin width in metric units")
args = parser.parse_args()

LOG_FILE = args.log
TS_FMT = "%Y-%m-%d %H:%M:%S"
EVENT_RE = re.compile(r"^(\d{3}) \((\d+\.\d+\.\d+)\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RETVAL_RE = re.compile(r"return value (\d+)")
DISK_RE = re.compile(r"Disk \(KB\)\s*:\s*(\d+)")
MEM_RE = re.compile(r"Memory \(MB\)\s*:\s*(\d+)")

starts: dict[str, datetime] = {}
values: list[float] = []

# State for accumulating a 005 block
pending_005: tuple[datetime, datetime] | None = None  # (start_ts, end_ts)
block_lines: list[str] = []


def process_005_block(start_ts: datetime, end_ts: datetime, lines: list[str]) -> None:
    retval = None
    disk_kb = None
    mem_mb = None
    for line in lines:
        if retval is None:
            m = RETVAL_RE.search(line)
            if m:
                retval = int(m.group(1))
        d = DISK_RE.search(line)
        if d:
            disk_kb = float(d.group(1))
        mm = MEM_RE.search(line)
        if mm:
            mem_mb = float(mm.group(1))

    if retval != 0:
        return

    if args.metric == "time":
        values.append((end_ts - start_ts).total_seconds() / 3600)
    elif args.metric == "disk" and disk_kb is not None:
        values.append(disk_kb / (1024 * 1024))  # KB -> GB
    elif args.metric == "memory" and mem_mb is not None:
        values.append(mem_mb / 1024)  # MB -> GB


with open(LOG_FILE) as f:
    for line in f:
        if line.startswith("..."):
            if pending_005 is not None:
                process_005_block(pending_005[0], pending_005[1], block_lines)
                pending_005 = None
                block_lines = []
            continue

        if pending_005 is not None:
            block_lines.append(line)
            continue

        m = EVENT_RE.match(line)
        if not m:
            continue
        code, job_id, ts_str = m.group(1), m.group(2), m.group(3)
        ts = datetime.strptime(ts_str, TS_FMT)
        if code == "001" and job_id not in starts:
            starts[job_id] = ts
        elif code == "005" and job_id in starts:
            pending_005 = (starts.pop(job_id), ts)
            block_lines = []

# Flush any trailing block without a trailing `...`
if pending_005 is not None and block_lines:
    process_005_block(pending_005[0], pending_005[1], block_lines)

METRIC_LABELS = {"time": ("hours", "h"), "disk": ("GB", "GB"), "memory": ("GB", "GB")}
unit_long, unit_short = METRIC_LABELS[args.metric]

if not values:
    print(f"No completed jobs with {args.metric} data found.")
    sys.exit(0)

values.sort()
n = len(values)
lo, hi = values[0], values[-1]
mean = sum(values) / n
median = values[n // 2]

print(
    f"Completed jobs: {n}  |  "
    f"min: {lo:.2f}{unit_short}  max: {hi:.2f}{unit_short}  "
    f"mean: {mean:.2f}{unit_short}  median: {median:.2f}{unit_short}\n"
)

WIDTH = 60
if args.bin_size:
    bin_size = args.bin_size
    n_bins = max(1, int((hi - lo) / bin_size) + 1)
else:
    n_bins = args.bins
    bin_size = (hi - lo) / n_bins or 1
counts = [0] * n_bins
for v in values:
    idx = min(int((v - lo) / bin_size), n_bins - 1)
    counts[idx] += 1

max_count = max(counts)
for i, count in enumerate(counts):
    label = f"{lo + i * bin_size:6.1f}{unit_short}"
    bar = "#" * round(count / max_count * WIDTH) if max_count else ""
    print(f"{label} | {bar:<{WIDTH}} {count}")
