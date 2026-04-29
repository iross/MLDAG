import json
from pathlib import Path

import pytest

from mldag.provenance.events import (
    emit_event,
    event_log_path,
    SCHEMA_VERSION,
    VALID_EVENT_TYPES,
)


RUN_ID = "test-run-abc"


def _read_events(log_dir: Path, run_id: str = RUN_ID) -> list[dict]:
    path = event_log_path(run_id, log_dir)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- envelope fields ---

def test_emit_event_envelope_fields_present(tmp_path):
    emit_event("job.submitted", RUN_ID, log_dir=tmp_path)
    events = _read_events(tmp_path)
    assert len(events) == 1
    e = events[0]
    assert e["schema_version"] == SCHEMA_VERSION
    assert e["type"] == "job.submitted"
    assert e["run_id"] == RUN_ID
    assert "ts" in e


def test_emit_event_ts_is_iso8601(tmp_path):
    from datetime import datetime
    emit_event("job.submitted", RUN_ID, log_dir=tmp_path)
    e = _read_events(tmp_path)[0]
    # Should parse without error
    datetime.fromisoformat(e["ts"])


def test_emit_event_extra_fields_included(tmp_path):
    emit_event("epoch.completed", RUN_ID, log_dir=tmp_path,
               epoch=3, loss=0.42, duration_s=900)
    e = _read_events(tmp_path)[0]
    assert e["epoch"] == 3
    assert e["loss"] == 0.42
    assert e["duration_s"] == 900


# --- all seven event types ---

@pytest.mark.parametrize("event_type", sorted(VALID_EVENT_TYPES))
def test_all_event_types_accepted(tmp_path, event_type):
    emit_event(event_type, RUN_ID, log_dir=tmp_path / event_type)


def test_unknown_event_type_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown event type"):
        emit_event("job.invented", RUN_ID, log_dir=tmp_path)


# --- file append behaviour ---

def test_emit_event_appends_not_overwrites(tmp_path):
    emit_event("job.submitted", RUN_ID, log_dir=tmp_path)
    emit_event("job.assigned", RUN_ID, log_dir=tmp_path)
    emit_event("epoch.started", RUN_ID, log_dir=tmp_path, epoch=0)
    events = _read_events(tmp_path)
    assert len(events) == 3
    assert [e["type"] for e in events] == ["job.submitted", "job.assigned", "epoch.started"]


def test_emit_event_creates_log_dir(tmp_path):
    nested = tmp_path / "deep" / "provenance"
    emit_event("job.submitted", RUN_ID, log_dir=nested)
    assert event_log_path(RUN_ID, nested).exists()


def test_emit_event_one_line_per_event(tmp_path):
    for i in range(5):
        emit_event("epoch.completed", RUN_ID, log_dir=tmp_path, epoch=i)
    lines = event_log_path(RUN_ID, tmp_path).read_text().splitlines()
    assert len(lines) == 5
    # every line is valid JSON
    for line in lines:
        json.loads(line)


def test_emit_event_separate_runs_separate_files(tmp_path):
    emit_event("job.submitted", "run-aaa", log_dir=tmp_path)
    emit_event("job.submitted", "run-bbb", log_dir=tmp_path)
    assert event_log_path("run-aaa", tmp_path).exists()
    assert event_log_path("run-bbb", tmp_path).exists()
    assert event_log_path("run-aaa", tmp_path) != event_log_path("run-bbb", tmp_path)
