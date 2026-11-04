"""Tests for mldag.report.csv DAG parsing."""

from pathlib import Path

import pytest

from mldag.report.csv import SimpleCSVGenerator


def _write_dag(path: Path, run_uuid: str) -> str:
    path.write_text(
        "JOB run0-train_epoch0 pretrain.sub\n"
        f'VARS run0-train_epoch0 epoch="1" run_uuid="{run_uuid}"\n'
    )
    return str(path)


def _generator(tmp_path: Path, with_dagman_out: bool) -> SimpleCSVGenerator:
    dag_a = _write_dag(tmp_path / "a.dag", "aaaa")
    dag_b = _write_dag(tmp_path / "b.dag", "bbbb")
    if with_dagman_out:
        for dag in (dag_a, dag_b):
            Path(f"{dag}.dagman.out").write_text("")
    metl_log = tmp_path / "metl.log"
    metl_log.write_text("")
    return SimpleCSVGenerator(dag_files=[dag_a, dag_b], metl_logs=[str(metl_log)])


def test_warns_when_shared_job_names_lack_dagman_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _generator(tmp_path, with_dagman_out=False).parse_dag_files()

    err = capsys.readouterr().err
    assert "WARNING: 1 job names (e.g. run0-train_epoch0)" in err
    assert "a.dag" in err and "b.dag" in err


def test_no_warning_when_dagman_out_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _generator(tmp_path, with_dagman_out=True).parse_dag_files()

    assert "WARNING" not in capsys.readouterr().err
