"""NDJSON structured provenance event emitter.

Appends one JSON line per lifecycle event to a per-run NDJSON file on the
shared filesystem.  All writes use O_APPEND mode; on POSIX systems writes
smaller than PIPE_BUF (~4 KB) are atomic, so concurrent epoch writers are
safe without additional locking.

Event log location: <log_dir>/<run_id>.ndjson
Default log_dir:    output/provenance
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"

VALID_EVENT_TYPES: frozenset[str] = frozenset({
    "job.submitted",
    "job.assigned",
    "epoch.started",
    "epoch.completed",
    "job.migrated",
    "job.held",
    "job.released",
    "job.failed",
    "job.completed",
})

_DEFAULT_LOG_DIR = "output/provenance"


def event_log_path(run_id: str, log_dir: str | Path = _DEFAULT_LOG_DIR) -> Path:
    """Return the path to the NDJSON event log for run_id."""
    return Path(log_dir) / f"{run_id}.ndjson"


def emit_event(
    event_type: str,
    run_id: str,
    *,
    log_dir: str | Path = _DEFAULT_LOG_DIR,
    **fields,
) -> None:
    """Append one structured event line to the run's NDJSON log.

    Args:
        event_type: One of the seven defined lifecycle event types.
        run_id: Stable run identifier (PROVENANCE_RUN_ID).
        log_dir: Directory for NDJSON files; created if absent.
        **fields: Event-specific payload fields merged into the envelope.

    Raises:
        ValueError: If event_type is not a recognised event type.
    """
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"Unknown event type {event_type!r}. "
            f"Valid types: {sorted(VALID_EVENT_TYPES)}"
        )

    event = {
        "schema_version": SCHEMA_VERSION,
        "type": event_type,
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }

    log_path = event_log_path(run_id, log_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise to a single line; compact separators keep it under PIPE_BUF.
    line = json.dumps(event, separators=(",", ":")) + "\n"
    with open(log_path, "a") as f:
        f.write(line)
