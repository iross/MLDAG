"""Repair tool for unresolved unknown:<cluster_id>.ndjson provenance files (task-22).

log_monitor.py falls back to "unknown:<cluster_id>" when it cannot map a
cluster_id to a run_id at the time an event fires. This tool resolves what
it can after the fact by cross-referencing:

  1. metl_log's event-000 blocks, which contain `DAGNodeName = "<job_name>"`
     for jobs submitted via DAGMan -- gives cluster_id -> job_name.
  2. DAG VARS lines (job_name -> run_uuid) and/or job.submitted NDJSON
     records already present in the provenance directories (job_name ->
     run_id) -- gives job_name -> run_id.

Chaining the two gives cluster_id -> run_id. Resolution is only as complete
as metl_log's coverage: if metl_log does not span the time window an
unknown file's events were recorded in (e.g. a stale local copy), those
files are reported as unresolved rather than guessed at.

Also merges byte-identical unknown:*.ndjson files that exist in more than
one provenance directory (see task-22's finding that `provenance/` was a
stale copy of `output/provenance/`) so they aren't double-counted.

Usage:
    python -m mldag.provenance.repair [--dry-run] \\
        --provenance-dir output/provenance [--provenance-dir provenance] \\
        --metl-log metl.log [--dag-file many_protein_pretraining_with_ospool.dag]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_EVENT_000_RE = re.compile(r"^000 \((\d+)\.\d+\.\d+\)")
_DAGNODE_NAME_RE = re.compile(r'DAGNodeName\s*=\s*"([^"]+)"')
_DAG_VARS_RE = re.compile(r'^VARS (\S+) .*run_uuid="([^"]+)"')


def build_cluster_to_job(metl_log: Path) -> dict[int, str]:
    """Map cluster_id -> job_name from event-000 DAGNodeName blocks in metl_log."""
    cluster_to_job: dict[int, str] = {}
    current_cluster: int | None = None
    for line in metl_log.read_text().splitlines():
        line = line.strip()
        m = _EVENT_000_RE.match(line)
        if m:
            current_cluster = int(m.group(1))
            continue
        if current_cluster is not None:
            dn = _DAGNODE_NAME_RE.search(line)
            if dn:
                cluster_to_job[current_cluster] = dn.group(1)
                current_cluster = None
    return cluster_to_job


def build_job_to_run_id(dag_file: Path | None, provenance_dirs: list[Path]) -> dict[str, str]:
    """Map job_name -> run_id from DAG VARS lines and NDJSON records.

    Any event carrying a job_name (job.submitted, job.completed, job.failed,
    job.queued, ...) counts as a source -- not just job.submitted -- since
    every event in a properly-named `<run_id>.ndjson` file necessarily
    belongs to that run by construction (see events.py's event_log_path).
    The file's own name is used as the run_id, rather than trusting each
    event's embedded "run_id" field, so this still works once the .dag file
    has been cleaned up. unknown:*.ndjson files are never treated as a
    source -- they are exactly what this tool is trying to fix, not a
    trustworthy mapping.
    """
    job_to_run_id: dict[str, str] = {}
    if dag_file is not None and dag_file.exists():
        for line in dag_file.read_text().splitlines():
            m = _DAG_VARS_RE.match(line)
            if m:
                job_to_run_id[m.group(1)] = m.group(2)

    for prov_dir in provenance_dirs:
        for ndjson_path in sorted(prov_dir.glob("*.ndjson")):
            if ndjson_path.name.startswith("unknown"):
                continue
            run_id = ndjson_path.stem
            try:
                lines = ndjson_path.read_text().splitlines()
            except OSError:
                continue
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                job_name = event.get("job_name")
                if job_name and job_name not in job_to_run_id:
                    job_to_run_id[job_name] = run_id
    return job_to_run_id


class RepairStats:
    """Outcome counts from a recover() run; str()s into a one-line summary."""

    def __init__(self) -> None:
        self.merged = 0
        self.duplicates_removed = 0
        self.skipped_unparseable_name = 0
        self.skipped_no_dagnode = 0
        self.skipped_no_run_id = 0

    def __str__(self) -> str:
        return (
            f"merged={self.merged} duplicates_removed={self.duplicates_removed} "
            f"skipped_no_dagnode={self.skipped_no_dagnode} "
            f"skipped_no_run_id={self.skipped_no_run_id} "
            f"skipped_unparseable_name={self.skipped_unparseable_name}"
        )


def recover(
    provenance_dirs: list[Path],
    metl_log: Path,
    dag_file: Path | None,
    *,
    dry_run: bool = False,
) -> RepairStats:
    """Resolve unknown:<cluster_id>.ndjson files across provenance_dirs in place."""
    stats = RepairStats()
    cluster_to_job = build_cluster_to_job(metl_log)
    job_to_run_id = build_job_to_run_id(dag_file, provenance_dirs)

    seen_content: dict[str, Path] = {}
    for prov_dir in provenance_dirs:
        for path in sorted(prov_dir.glob("unknown*.ndjson")):
            try:
                content = path.read_text()
            except OSError:
                continue

            if content in seen_content:
                print(f"  DUPLICATE {path} (identical to {seen_content[content]}) -- removing")
                stats.duplicates_removed += 1
                if not dry_run:
                    path.unlink()
                continue
            seen_content[content] = path

            try:
                cluster_id = int(path.stem.split(":")[1])
            except (IndexError, ValueError):
                print(f"  SKIP {path} -- cannot parse a cluster_id from the filename")
                stats.skipped_unparseable_name += 1
                continue

            job_name = cluster_to_job.get(cluster_id)
            if job_name is None:
                print(f"  SKIP {path} -- no DAGNodeName found for cluster {cluster_id} in {metl_log}")
                stats.skipped_no_dagnode += 1
                continue

            run_id = job_to_run_id.get(job_name)
            if run_id is None:
                print(f"  SKIP {path} -- job {job_name!r} has no known run_id")
                stats.skipped_no_run_id += 1
                continue

            fixed_lines = []
            for raw in content.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                event = json.loads(raw)
                event["run_id"] = run_id
                fixed_lines.append(json.dumps(event, separators=(",", ":")))
            if not fixed_lines:
                continue

            dest = prov_dir / f"{run_id}.ndjson"
            print(f"  {'[dry] ' if dry_run else ''}{path} -> {dest} ({len(fixed_lines)} events)")
            if not dry_run:
                with dest.open("a") as f:
                    f.write("\n".join(fixed_lines) + "\n")
                path.unlink()
            stats.merged += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report actions without modifying files")
    parser.add_argument(
        "--provenance-dir", action="append", dest="provenance_dirs", metavar="DIR",
        help="Directory to repair; may be given multiple times. Defaults to output/provenance.",
    )
    parser.add_argument("--metl-log", type=Path, default=Path("metl.log"))
    parser.add_argument("--dag-file", type=Path, default=None)
    args = parser.parse_args()

    if not args.metl_log.exists():
        print(f"error: {args.metl_log} not found", file=sys.stderr)
        raise SystemExit(1)

    provenance_dirs = [Path(d) for d in (args.provenance_dirs or ["output/provenance"])]
    stats = recover(provenance_dirs, args.metl_log, args.dag_file, dry_run=args.dry_run)
    print(stats)


if __name__ == "__main__":
    main()
