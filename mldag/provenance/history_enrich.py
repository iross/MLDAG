"""Backfill provenance.db's condor_history table from three independent sources.

condor_history keeps fields the event-log/NDJSON pipeline (log_monitor.py,
post.py) never sees: final Owner, the exact RemoteHost/LastRemoteHost,
ExitCode, and the originally requested resources alongside what was actually
used. Three entry points write into the same `condor_history` table, tagged
by `source`, merging rather than overwriting when more than one applies to
the same (cluster_id, proc_id) -- see db.py's condor_history docstring and
_upsert_history_row:

  enrich_from_condor_history() -- queries condor_history via the HTCondor
      Python bindings (Schedd.history, not the condor_history CLI) for
      cluster_ids already present in provenance.db's `events` table.
      htcondor2 is a Linux-only dependency (see pyproject.toml) -- this
      module is only importable where it's installed, which is why query.py's
      CLI wiring imports it lazily inside the command body rather than at
      module load time. Opportunistic and per-cluster_id: a cluster_id with
      no matching historical record (aged out of the schedd's history
      retention, wrong schedd, or a job that never actually reached
      HTCondor) is counted and skipped, not treated as an error --
      condor_history is a retention-limited cache, not a permanent record.

  write_scan_records() -- writes mldag.provenance.event_log_scan.scan_event_log()'s
      output (parsed directly from a raw HTCondor event log; no condor_history
      query needed).

  enrich_from_jobad_events() -- mirrors job.assigned events (jobad.py's in-job
      $_CONDOR_JOB_AD capture, emitted immediately at job start -- see
      pretrain_local.sh) that are already in provenance.db's `events` table.
      Deliberately excludes wall_time_s/cpu_usage/peak_memory_mb/gpu_usage:
      the job ad is captured before the job has run, so those fields would be
      near-zero placeholders, not real usage.

The ClassAd's Environment attribute is never stored, even in the raw
condor_history_json blob: it's where secrets like WANDB_API_KEY live (see
mldag.provenance.post._SENSITIVE_AD_KEYS). Only run_id is extracted out of it,
via the same regex post.py's run_id_from_classad uses.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mldag.provenance.db import _init_schema
from mldag.provenance.post import _SENSITIVE_AD_KEYS, run_id_from_classad

logger = logging.getLogger(__name__)

# ClassAd attribute -> condor_history column name.
_FIELD_MAPPING = {
    "ClusterId": "cluster_id",
    "ProcId": "proc_id",
    "Owner": "owner",
    "Cmd": "cmd",
    "Arguments": "arguments",
    "JobStatus": "job_status",
    "ExitCode": "exit_code",
    "RemoteHost": "remote_host",
    "LastRemoteHost": "last_remote_host",
    "RequestCpus": "request_cpus",
    "RequestMemory": "request_memory",
    "RequestGpus": "request_gpus",
    "RemoteWallClockTime": "remote_wall_clock_s",
    "CPUsUsage": "cpus_usage",
    "MemoryUsage": "memory_usage_mb",
    "GPUsUsage": "gpus_usage",
    "GLIDEIN_ResourceName": "resource_name",
    "HoldReason": "hold_reason",
    "QDate": "qdate",
    "JobStartDate": "job_start_date",
    "CompletionDate": "completion_date",
}

# Environment is projected (to recover run_id) but is never stored -- see
# module docstring -- so it's kept out of _FIELD_MAPPING's column mapping.
_PROJECTION = list(dict.fromkeys(["ClusterId", "Environment", *_FIELD_MAPPING]))

# job_name/site/gpu_ids are event-log-only fields (event_log_scan.py); status
# is populated by both sources but from different vocabularies (see
# _JOB_STATUS_TO_STATUS / event_log_scan._status) -- the `source` column says
# which applies to a given row.
_HISTORY_COLUMNS = [
    *_FIELD_MAPPING.values(), "run_id", "job_name", "site", "gpu_ids", "status",
    "source", "condor_history_json", "event_log_json", "jobad_json", "queried_at",
]

# HTCondor's numeric JobStatus ClassAd attribute -> a human-readable status
# string, so condor_history-sourced and event-log-sourced rows can both be
# queried via one `status` column (their vocabularies differ, but overlap
# where it matters, e.g. both spell a held job "held").
_JOB_STATUS_TO_STATUS = {
    1: "idle",
    2: "running",
    3: "removed",
    4: "completed",
    5: "held",
    6: "transferring_output",
    7: "suspended",
}

# event_log_scan.py record field -> condor_history column name.
_SCAN_FIELD_MAPPING = {
    "cluster_id": "cluster_id",
    "proc_id": "proc_id",
    "run_id": "run_id",
    "job_name": "job_name",
    "execute_host": "remote_host",
    "site": "site",
    "resource_name": "resource_name",
    "wall_time_s": "remote_wall_clock_s",
    "cpu_usage": "cpus_usage",
    "peak_memory_mb": "memory_usage_mb",
    "gpu_usage": "gpus_usage",
    "gpu_ids": "gpu_ids",
    "status": "status",
}

# job.assigned event field (jobad.py's capture_job_ad_fields(), see
# pretrain_local.sh) -> condor_history column name. Deliberately excludes
# wall_time_s/cpu_usage/peak_memory_mb/gpu_usage even though
# capture_job_ad_fields() can return them: the job ad is captured at
# submission, before the job has run, so those would be near-zero
# placeholders -- see the condor_history table's docstring in db.py.
_JOBAD_FIELD_MAPPING = {
    "cluster_id": "cluster_id",
    "proc_id": "proc_id",
    "run_id": "run_id",
    "resource_name": "resource_name",
    "arguments": "arguments",
    "request_cpus": "request_cpus",
    "request_memory": "request_memory",
    "request_gpus": "request_gpus",
}

_DEFAULT_BATCH_SIZE = 50


@dataclass
class EnrichStats:
    """Outcome counts from an enrich_from_condor_history() run; str()s into a summary line."""

    enriched: int = 0
    not_found: int = 0
    already_enriched: int = 0
    query_errors: int = 0

    def __str__(self) -> str:
        return (
            f"condor_history: {self.enriched} enriched, "
            f"{self.not_found} not found in history, "
            f"{self.already_enriched} already enriched (skipped), "
            f"{self.query_errors} query errors"
        )


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _all_cluster_ids_from_events(conn: sqlite3.Connection) -> set[int]:
    """Return every cluster_id any event in the `events` table carries."""
    rows = conn.execute(
        "SELECT DISTINCT CAST(json_extract(payload_json, '$.cluster_id') AS INTEGER) "
        "FROM events WHERE json_extract(payload_json, '$.cluster_id') IS NOT NULL"
    )
    return {row[0] for row in rows}


def _row_from_ad(ad: dict) -> dict:
    """Map a condor_history ClassAd (as a plain dict) to a condor_history table row.

    Never includes the raw Environment attribute -- see module docstring.
    """
    row = {column: ad[ad_key] for ad_key, column in _FIELD_MAPPING.items() if ad_key in ad}
    run_id = run_id_from_classad(ad)
    row["run_id"] = run_id if run_id != "unknown" else None
    row["status"] = _JOB_STATUS_TO_STATUS.get(row.get("job_status"))
    row["source"] = "condor_history"
    sanitized = {k: v for k, v in ad.items() if k not in _SENSITIVE_AD_KEYS}
    row["condor_history_json"] = json.dumps(sanitized, default=str)
    return row


def _row_from_scan_record(rec: dict, now: str) -> dict:
    """Map an event_log_scan.py record to a condor_history row (source='event_log')."""
    row = {column: rec[key] for key, column in _SCAN_FIELD_MAPPING.items() if key in rec}
    row["source"] = "event_log"
    row["event_log_json"] = json.dumps(rec, default=str)
    row["queried_at"] = now
    return row


def write_scan_records(db_path: str | Path, records: list[dict]) -> int:
    """Upsert event_log_scan.py's scan records into condor_history (source='event_log').

    Args:
        db_path: Path to the provenance SQLite database (see mldag.provenance.db).
        records: Records as returned by mldag.provenance.event_log_scan.scan_event_log.

    Returns:
        Number of rows written.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        _init_schema(conn)
        for rec in records:
            _upsert_history_row(conn, _row_from_scan_record(rec, now))
        conn.commit()
    finally:
        conn.close()
    return len(records)


def _row_from_jobad_event(event: dict) -> dict:
    """Map a job.assigned event's payload to a condor_history row (source='jobad')."""
    row = {column: event[key] for key, column in _JOBAD_FIELD_MAPPING.items() if key in event}
    row["source"] = "jobad"
    row["jobad_json"] = json.dumps(event, default=str)
    return row


def enrich_from_jobad_events(db_path: str | Path) -> int:
    """Mirror job.assigned events (jobad.py's in-job $_CONDOR_JOB_AD capture) into condor_history.

    job.assigned is emitted immediately from within the running job (see
    pretrain_local.sh) and already lands in the `events` table via
    mldag.provenance.db.build_database -- this reads it from there rather
    than re-scanning NDJSON directly, so it only ever needs provenance.db.

    Args:
        db_path: Path to the provenance SQLite database (see mldag.provenance.db).

    Returns:
        Number of rows written.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        _init_schema(conn)
        rows = conn.execute(
            "SELECT payload_json FROM events WHERE type = 'job.assigned' "
            "AND json_extract(payload_json, '$.cluster_id') IS NOT NULL"
        ).fetchall()
        for (payload_json,) in rows:
            event = json.loads(payload_json)
            row = _row_from_jobad_event(event)
            row["queried_at"] = now
            _upsert_history_row(conn, row)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


_MERGE_COLUMNS = [
    c for c in _HISTORY_COLUMNS if c not in ("cluster_id", "proc_id", "source", "queried_at")
]


def _upsert_history_row(conn: sqlite3.Connection, row: dict) -> None:
    """Insert row, or merge it into an existing (cluster_id, proc_id) row.

    A merge is column-by-column COALESCE(new, existing): a column this write
    doesn't know about keeps whatever value was already there, rather than
    being blanked to NULL. Without this, writing from condor_history and then
    from an event-log scan (or vice versa) for the same job would silently
    erase whichever columns only the first write populated -- see db.py's
    condor_history docstring.
    """
    row = {**dict.fromkeys(_HISTORY_COLUMNS), **row}
    new_source = row["source"]
    assert isinstance(new_source, str), "_row_from_ad/_row_from_scan_record always set source"

    existing = conn.execute(
        "SELECT source FROM condor_history WHERE cluster_id = ? AND proc_id = ?",
        (row["cluster_id"], row["proc_id"]),
    ).fetchone()
    if existing and existing[0]:
        row["source"] = ",".join(sorted({*existing[0].split(","), new_source}))

    columns = ", ".join(_HISTORY_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _HISTORY_COLUMNS)
    merge_clause = ", ".join(
        f"{c} = COALESCE(excluded.{c}, condor_history.{c})" for c in _MERGE_COLUMNS
    )
    conn.execute(
        f"""
        INSERT INTO condor_history ({columns}) VALUES ({placeholders})
        ON CONFLICT (cluster_id, proc_id) DO UPDATE SET
            {merge_clause},
            source = excluded.source,
            queried_at = excluded.queried_at
        """,
        row,
    )


def _get_schedd(schedd_name: str | None, pool: str | None):
    """Return an htcondor2.Schedd for schedd_name, or the local schedd if unset."""
    import htcondor2 as htcondor

    if schedd_name:
        collector = htcondor.Collector(pool) if pool else htcondor.Collector()
        location_ad = collector.locate(htcondor.DaemonTypes.Schedd, schedd_name)
        return htcondor.Schedd(location_ad)
    return htcondor.Schedd()


def _noop_progress(_msg: str) -> None:
    pass


def enrich_from_condor_history(
    db_path: str | Path,
    *,
    schedd_name: str | None = None,
    pool: str | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    full_rescan: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> EnrichStats:
    """Backfill db_path's condor_history table for cluster_ids seen in its events table.

    Args:
        db_path: Path to the provenance SQLite database (see mldag.provenance.db).
        schedd_name: Name of the schedd to query, as HTCondor's collector knows
            it. Defaults to the local schedd.
        pool: Collector address to resolve schedd_name against. Only used
            when schedd_name is given; defaults to the local pool.
        batch_size: How many cluster_ids to fold into one condor_history
            constraint query.
        full_rescan: Re-query every cluster_id in the events table, including
            ones already in condor_history, ignoring recorded results.
        on_progress: Optional callback invoked with a one-line status message
            before the first query and after each batch completes -- a batch
            of condor_history queries against a large events table can take a
            while, and this function otherwise gives no sign of life until it
            returns. No-op when omitted.

    Returns:
        EnrichStats with counts of what was enriched/not-found/skipped.

    Raises:
        ImportError: if htcondor2 (a Linux-only dependency) is not installed.
    """
    on_progress = on_progress or _noop_progress

    now = datetime.now(timezone.utc).isoformat()
    stats = EnrichStats()
    conn = sqlite3.connect(db_path)
    try:
        _init_schema(conn)
        all_ids = _all_cluster_ids_from_events(conn)
        if full_rescan:
            target_ids = sorted(all_ids)
        else:
            # Scoped to source='condor_history' (not "any row exists"): a
            # cluster_id touched only by scan --db (source='event_log') was
            # never actually condor_history-queried, so it isn't "already
            # enriched" from this function's point of view.
            existing = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT cluster_id FROM condor_history "
                    "WHERE ',' || source || ',' LIKE '%,condor_history,%'"
                )
            }
            target_ids = sorted(all_ids - existing)
            stats.already_enriched = len(all_ids) - len(target_ids)

        if not target_ids:
            on_progress("No new cluster_ids to enrich.")
            return stats

        batches = list(_chunks(target_ids, batch_size))
        on_progress(
            f"Querying condor_history for {len(target_ids)} cluster_id(s) "
            f"in {len(batches)} batch(es) of up to {batch_size}..."
        )

        schedd = _get_schedd(schedd_name, pool)
        for batch_num, batch in enumerate(batches, start=1):
            constraint = " || ".join(f"ClusterId == {cid}" for cid in batch)
            try:
                ads = schedd.history(constraint=constraint, projection=_PROJECTION, match=len(batch))
                ads = list(ads)
            except Exception as exc:
                # Schedd.history() is an RPC to an external daemon -- timeouts,
                # auth failures, and an unreachable schedd all surface here as
                # different exception types; none should abort batches already
                # enriched, so log and move on rather than letting one bad
                # batch fail the whole run.
                logger.warning(
                    "condor_history query failed for %d cluster_id(s): %s", len(batch), exc
                )
                stats.query_errors += len(batch)
                on_progress(f"Batch {batch_num}/{len(batches)}: query failed ({exc})")
                continue

            found: set[int] = set()
            for ad in ads:
                ad_dict = dict(ad)
                cluster_id = ad_dict.get("ClusterId")
                if cluster_id is None:
                    continue
                cluster_id = int(cluster_id)
                found.add(cluster_id)
                row = _row_from_ad(ad_dict)
                row["queried_at"] = now
                _upsert_history_row(conn, row)
                stats.enriched += 1
            stats.not_found += len(set(batch) - found)
            on_progress(
                f"Batch {batch_num}/{len(batches)}: {len(found)} enriched, "
                f"{len(batch) - len(found)} not found "
                f"({stats.enriched} enriched so far)"
            )
        conn.commit()
    finally:
        conn.close()
    return stats
