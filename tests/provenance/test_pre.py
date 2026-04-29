import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mldag.provenance.pre import main


def _read_events(log_dir: Path, run_id: str) -> list[dict]:
    path = log_dir / f"{run_id}.ndjson"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_main_emits_job_submitted(tmp_path):
    with patch("sys.argv", ["provenance_pre", "run-abc", "run0-train_epoch0", "5"]):
        with patch.dict("os.environ", {"PROVENANCE_LOG_DIR": str(tmp_path)}):
            main()
    events = _read_events(tmp_path, "run-abc")
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "job.submitted"
    assert e["run_id"] == "run-abc"
    assert e["job_name"] == "run0-train_epoch0"
    assert e["epoch"] == 5


def test_main_wrong_arg_count_exits_nonzero(tmp_path):
    with patch("sys.argv", ["provenance_pre", "run-abc"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0


def test_main_uses_provenance_log_dir_env(tmp_path):
    custom_dir = tmp_path / "custom"
    with patch("sys.argv", ["provenance_pre", "run-xyz", "run0-train_epoch0", "0"]):
        with patch.dict("os.environ", {"PROVENANCE_LOG_DIR": str(custom_dir)}):
            main()
    assert (custom_dir / "run-xyz.ndjson").exists()
