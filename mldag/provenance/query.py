"""Lineage reconstruction and provenance query library + CLI.

Library functions:
    walk_lineage(checkpoint_path)  -- follow parent_hash chain back to epoch 0
    query_run(run_id, log_dir)     -- read all events from the NDJSON log

CLI:
    mldag-query lineage <checkpoint>   -- print ancestry chain
    mldag-query events <run_id>        -- print all events for a run
    mldag-query db build               -- build/refresh the SQLite database

Both `lineage` and `events` accept --json for machine-readable output and
--log-dir to override the default NDJSON location. See `mldag-query db
build --help` and mldag/provenance/db.py for the database schema and
example queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from mldag.provenance.db import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_EVENT_DIR,
    build_database,
)
from mldag.provenance.events import _DEFAULT_LOG_DIR
from mldag.provenance.event_log_scan import scan_event_log

app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True, help="Build/refresh the local SQLite provenance database.")
app.add_typer(db_app, name="db")


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def _read_sidecar(path: Path) -> dict:
    """Load and return the parsed sidecar for *path*.

    Raises:
        FileNotFoundError: if the sidecar does not exist.
        ValueError: if the sidecar cannot be parsed.
    """
    sidecar_path = Path(str(path) + ".provenance.json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"No sidecar found for {path} (expected {sidecar_path})")
    try:
        return json.loads(sidecar_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt sidecar at {sidecar_path}: {exc}") from exc


def _find_checkpoint_by_hash(start_dir: Path, target_hash: str) -> Path | None:
    """Search start_dir recursively for a checkpoint whose sidecar has checkpoint_hash == target_hash."""
    for sidecar_path in start_dir.rglob("*.provenance.json"):
        try:
            data = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("checkpoint_hash") == target_hash:
            ckpt_path = Path(str(sidecar_path)[: -len(".provenance.json")])
            if ckpt_path.exists():
                return ckpt_path
    return None


def walk_lineage(checkpoint_path: str | Path) -> list[dict]:
    """Return the ordered sidecar chain from epoch 0 to checkpoint_path.

    Follows parent_hash links backwards, then reverses so index 0 is the
    oldest checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (not the sidecar).

    Returns:
        List of sidecar dicts, oldest first.

    Raises:
        FileNotFoundError: if a sidecar in the chain is missing.
        ValueError: if a sidecar is malformed or the chain is cyclic.
    """
    checkpoint_path = Path(checkpoint_path)
    chain: list[dict] = []
    seen_hashes: set[str] = set()
    search_dir = checkpoint_path.parent

    current = checkpoint_path
    while True:
        record = _read_sidecar(current)
        h = record.get("checkpoint_hash", "")
        if h in seen_hashes:
            raise ValueError(f"Cycle detected in lineage chain at hash {h!r}")
        seen_hashes.add(h)
        chain.append(record)

        parent_hash = record.get("parent_hash")
        if parent_hash is None:
            break

        parent_ckpt = _find_checkpoint_by_hash(search_dir, parent_hash)
        if parent_ckpt is None:
            raise FileNotFoundError(
                f"Parent checkpoint with hash {parent_hash!r} not found under {search_dir}"
            )
        current = parent_ckpt

    chain.reverse()
    return chain


def query_run(run_id: str, log_dir: str | Path = _DEFAULT_LOG_DIR) -> list[dict]:
    """Return all provenance events for run_id in chronological order.

    Args:
        run_id: The run identifier (PROVENANCE_RUN_ID).
        log_dir: Directory containing NDJSON event logs.

    Returns:
        List of event dicts sorted by ts field.

    Raises:
        FileNotFoundError: if the NDJSON log for run_id does not exist.
    """
    log_path = Path(log_dir) / f"{run_id}.ndjson"
    if not log_path.exists():
        raise FileNotFoundError(f"No event log found for run {run_id!r} at {log_path}")
    events = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return sorted(events, key=lambda e: e.get("ts", ""))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_lineage(chain: list[dict]) -> str:
    lines = []
    for record in chain:
        epoch = record.get("epoch", "?")
        h = record.get("checkpoint_hash", "?")[:16]
        parent = (record.get("parent_hash") or "genesis")[:16]
        site = record.get("produced_at", {}).get("hostname", "unknown")
        lines.append(f"  epoch {epoch:>3}  {h}  ← {parent}  [{site}]")
    return "\n".join(lines)


def _format_scan(records: list[dict]) -> str:
    lines = []
    for r in records:
        # job_id (cluster.proc) is always unique; run_id/job_name is often
        # shared by every proc in a cluster (job-array batches, or a DAGMan
        # run_id resolved at cluster granularity), so it's shown alongside
        # job_id rather than instead of it.
        job_id = f"{r['cluster_id']}.{r['proc_id']}"
        label = r.get("run_id") or r.get("job_name") or "-"
        site = r.get("site") or r.get("resource_name") or "?"
        wall_time = f"{r['wall_time_s']:.0f}s" if "wall_time_s" in r else "?"
        lines.append(
            f"  {job_id:<14} {label:<24} {r['status']:<10} site={site:<30} wall_time={wall_time}"
        )
    return "\n".join(lines)


def _format_events(events: list[dict]) -> str:
    lines = []
    for e in events:
        ts = e.get("ts", "?")[:19]
        etype = e.get("type", "?")
        extras = {k: v for k, v in e.items() if k not in ("schema_version", "type", "run_id", "ts")}
        extra_str = "  " + "  ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        lines.append(f"  {ts}  {etype:<20}{extra_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def lineage(
    checkpoint: Annotated[str, typer.Argument(help="Path to checkpoint file")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """Print the ancestry chain for a checkpoint."""
    try:
        chain = walk_lineage(checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if json_out:
        typer.echo(json.dumps(chain, indent=2))
    else:
        run_id = chain[0].get("run_id", "?") if chain else "?"
        typer.echo(f"Lineage for run {run_id} ({len(chain)} checkpoint(s)):")
        typer.echo(_format_lineage(chain))


@app.command()
def events(
    run_id: Annotated[str, typer.Argument(help="Run ID (PROVENANCE_RUN_ID)")],
    log_dir: Annotated[str, typer.Option(help="NDJSON log directory")] = _DEFAULT_LOG_DIR,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """Print all provenance events for a run."""
    try:
        run_events = query_run(run_id, log_dir)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if json_out:
        typer.echo(json.dumps(run_events, indent=2))
    else:
        typer.echo(f"Events for run {run_id} ({len(run_events)} total):")
        typer.echo(_format_events(run_events))


@app.command()
def scan(
    log_file: Annotated[str, typer.Argument(help="HTCondor event log to scan (e.g. metl.log)")],
    log_dir: Annotated[
        str,
        typer.Option(help="Classad/.run_id marker directory to opportunistically resolve run_id from; fine if it doesn't exist"),
    ] = _DEFAULT_LOG_DIR,
    provenance_log_dir: Annotated[
        str,
        typer.Option(help="NDJSON provenance directory to opportunistically resolve run_id from job_name; fine if it doesn't exist"),
    ] = _DEFAULT_LOG_DIR,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """Summarize every job in a raw HTCondor event log: duration, site, resource usage.

    Works on any event log, including one from a batch of jobs run without the
    DAGMan PRE/POST provenance pipeline -- run_id/job_name enrichment only
    happens when log_dir/provenance_log_dir actually have matching data.
    """
    records = scan_event_log(log_file, log_dir=log_dir, provenance_log_dir=provenance_log_dir)
    if json_out:
        typer.echo(json.dumps(records, indent=2))
    else:
        typer.echo(f"{len(records)} job(s) in {log_file}:")
        typer.echo(_format_scan(records))


@db_app.command("build")
def db_build(
    db: Annotated[str, typer.Option(help="Path to the SQLite database file")] = DEFAULT_DB_PATH,
    # ruff's UP045 assumes `from __future__ import annotations` makes `X | None`
    # runtime-safe on Python 3.9, but Typer evaluates these annotations at CLI
    # registration time to build the parser, so `list[str] | None` still raises
    # TypeError on 3.9 (types.GenericAlias.__or__ was only added in 3.10).
    checkpoint_dir: Annotated[
        Optional[list[str]],  # noqa: UP045
        typer.Option(help="Checkpoint sidecar directory to scan; repeatable"),
    ] = None,
    event_dir: Annotated[
        Optional[list[str]],  # noqa: UP045
        typer.Option(help="NDJSON event directory to scan; repeatable"),
    ] = None,
    full_rescan: Annotated[
        bool,
        typer.Option("--full-rescan", help="Re-ingest everything, ignoring recorded mtimes/offsets"),
    ] = False,
) -> None:
    """Build or refresh the SQLite database from checkpoint sidecars and NDJSON events."""
    checkpoint_dirs = checkpoint_dir or [DEFAULT_CHECKPOINT_DIR]
    event_dirs = event_dir or [DEFAULT_EVENT_DIR]
    stats = build_database(db, checkpoint_dirs, event_dirs, full_rescan=full_rescan)
    typer.echo(str(stats))


@db_app.command("enrich-history")
def db_enrich_history(
    db: Annotated[str, typer.Option(help="Path to the SQLite database file")] = DEFAULT_DB_PATH,
    schedd: Annotated[
        Optional[str],  # noqa: UP045 -- see db_build's Optional[list[str]] for why
        typer.Option(help="Schedd name to query; defaults to the local schedd"),
    ] = None,
    pool: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="Collector address to resolve --schedd against; defaults to the local pool"),
    ] = None,
    full_rescan: Annotated[
        bool,
        typer.Option("--full-rescan", help="Re-query every cluster_id, ignoring already-enriched ones"),
    ] = False,
) -> None:
    """Backfill the condor_history table from HTCondor job history via the Python bindings.

    Requires htcondor2 (a Linux-only dependency); imported here rather than at
    module load time so the rest of this CLI keeps working without it.
    """
    from mldag.provenance.history_enrich import enrich_from_condor_history

    stats = enrich_from_condor_history(
        db, schedd_name=schedd, pool=pool, full_rescan=full_rescan
    )
    typer.echo(str(stats))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
