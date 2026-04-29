"""Lineage reconstruction and provenance query library + CLI.

Library functions:
    walk_lineage(checkpoint_path)  -- follow parent_hash chain back to epoch 0
    query_run(run_id, log_dir)     -- read all events from the NDJSON log

CLI:
    mldag-query lineage <checkpoint>   -- print ancestry chain
    mldag-query events <run_id>        -- print all events for a run

Both commands accept --json for machine-readable output and --log-dir to
override the default NDJSON location.
"""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from mldag.provenance.events import _DEFAULT_LOG_DIR

app = typer.Typer(no_args_is_help=True)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
