import argparse
from datetime import date

import pandas as pd
import pytest

from mldag.report.experiment import ExperimentAnalyzer, _parse_month

CSV_COLUMNS = (
    "Job Name,DAG Source,Run UUID,HTCondor Cluster ID,Attempt Sequence,Final Status,"
    "Targeted Resource,GLIDEIN Resource Name,Submit Time,Start Time,End Time,"
    "Execution Duration (seconds),Execution Duration (human),Total Duration (seconds),"
    "Total Duration (human),Total Bytes Sent,Total Bytes Received,Number of GPUs,"
    "GPU Device Name,GPU Memory MB,GPU Capability,GPU Driver Version,GPU ECC Enabled,GPU UUID,"
    "GPU ID,GPU PCI Bus ID,GPU Usage,GPU Memory Usage MB,CPU Usage,Peak Memory Usage MB,"
    "Disk Usage KB,Held Time,Released Time,Evicted Time,Aborted Time,Transfer Input Start Time,"
    "Transfer Input End Time,Transfer Input Duration (seconds),Transfer Input Duration (human),"
    "Epochs Completed"
).split(",")


def write_jobs_csv(path, jobs):
    """Write (job name, submit/start time, end time) tuples as a mldag-csv style file."""
    df = pd.DataFrame(
        [
            {
                "Job Name": name,
                "DAG Source": "test.dag",
                "Submit Time": start,
                "Start Time": start,
                "End Time": end,
                "Final Status": "completed",
                "Targeted Resource": "delta",
                "Execution Duration (seconds)": 7200,
                "Total Bytes Sent": 0,
                "Total Bytes Received": 0,
                "Epochs Completed": 1,
            }
            for name, start, end in jobs
        ],
        columns=CSV_COLUMNS,
    )
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    ("value", "today", "expected"),
    [
        ("2025-09", date(2026, 9, 30), ("2025-09-01", "2025-09-30")),
        ("2024-02", date(2026, 1, 1), ("2024-02-01", "2024-02-29")),
        ("9", date(2026, 9, 30), ("2026-09-01", "2026-09-30")),
        ("11", date(2026, 9, 30), ("2025-11-01", "2025-11-30")),
        ("12", date(2026, 12, 1), ("2026-12-01", "2026-12-31")),
    ],
)
def test_parse_month(value, today, expected):
    assert _parse_month(value, today=today) == expected


@pytest.mark.parametrize("value", ["0", "13", "2025-13", "Sept", "", "2025/09"])
def test_parse_month_rejects_invalid(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_month(value)


def test_month_range_excludes_same_month_of_other_year(tmp_path):
    csv_path = write_jobs_csv(
        tmp_path / "jobs.csv",
        [
            ("run0-epoch1", "2025-09-10 10:00", "2025-09-10 12:00"),
            ("run0-epoch2", "2026-09-10 10:00", "2026-09-10 12:00"),
            ("run0-epoch3", "2026-08-31 10:00", "2026-08-31 12:00"),
            ("run0-epoch4", "2026-09-30 22:00", "2026-10-01 02:00"),
        ],
    )

    start, end = _parse_month("2026-09")
    analyzer = ExperimentAnalyzer(
        str(csv_path), str(tmp_path / "out"), start_date=start, end_date=end
    )

    assert sorted(analyzer.df["Job Name"]) == ["run0-epoch2", "run0-epoch4"]


def test_summary_period_shows_requested_window(tmp_path):
    csv_path = write_jobs_csv(
        tmp_path / "jobs.csv", [("run0-epoch1", "2026-06-18 09:00", "2026-09-02 12:00")]
    )

    analyzer = ExperimentAnalyzer(
        str(csv_path), str(tmp_path / "out"), start_date="2026-09-01", end_date="2026-09-30"
    )
    lines = analyzer._build_summary_report_lines("TITLE")

    assert any(line.startswith("Analysis Period: 2026-09-01 to 2026-09-30") for line in lines)


def test_summary_counts_models_with_final_epoch_done(tmp_path):
    csv_path = write_jobs_csv(
        tmp_path / "jobs.csv",
        [
            ("run0-train_epoch29", "2026-09-01 10:00", "2026-09-01 12:00"),
            ("run0-train_epoch29", "2026-09-02 10:00", "2026-09-02 12:00"),
            ("run1-train_epoch28", "2026-09-01 10:00", "2026-09-01 12:00"),
            ("run2-train_epoch29", "2026-09-01 10:00", "2026-09-01 12:00"),
        ],
    )
    df = pd.read_csv(csv_path)
    df.loc[3, "Final Status"] = "failed"
    df.to_csv(csv_path, index=False)

    analyzer = ExperimentAnalyzer(str(csv_path), str(tmp_path / "out"))
    lines = analyzer._build_summary_report_lines("TITLE")

    assert "Completed Models (30 epochs done): 1" in lines


def _set_gpus(csv_path, gpus):
    df = pd.read_csv(csv_path)
    df["Number of GPUs"] = gpus
    df.to_csv(csv_path, index=False)


def test_executive_summary_unfiltered_shows_all_time_totals(tmp_path):
    csv_path = write_jobs_csv(
        tmp_path / "jobs.csv",
        [
            ("run0-train_epoch28", "2026-09-01 10:00", "2026-09-01 12:00"),
            ("run0-train_epoch29", "2026-09-02 10:00", "2026-09-02 12:00"),
        ],
    )
    _set_gpus(csv_path, [1, 2])

    analyzer = ExperimentAnalyzer(str(csv_path), str(tmp_path / "out"))
    lines = analyzer._build_executive_summary_lines()

    assert "Completed Epochs: 2" in lines
    assert "Completed Models (30 epochs done): 1" in lines
    assert "GPU Hours: 6" in lines


def test_executive_summary_month_splits_period_from_all_time(tmp_path):
    csv_path = write_jobs_csv(
        tmp_path / "jobs.csv",
        [
            ("run0-train_epoch29", "2026-08-20 10:00", "2026-08-20 12:00"),
            ("run1-train_epoch28", "2026-08-31 22:00", "2026-09-01 02:00"),
            ("run1-train_epoch29", "2026-09-10 10:00", "2026-09-10 12:00"),
        ],
    )
    _set_gpus(csv_path, [1, 1, 1])

    start, end = _parse_month("2026-09")
    analyzer = ExperimentAnalyzer(
        str(csv_path), str(tmp_path / "out"), start_date=start, end_date=end
    )
    rows = {
        line.split("  ")[0].strip(): line.split()[-2:]
        for line in analyzer._build_executive_summary_lines()[3:]
    }

    assert rows["Completed Epochs"] == ["2", "3"]
    assert rows["Completed Models (30 epochs done)"] == ["1", "2"]
    assert rows["GPU Hours"] == ["4", "6"]


def test_month_counts_attempts_by_end_time(tmp_path):
    csv_path = write_jobs_csv(
        tmp_path / "jobs.csv",
        [("run0-train_epoch29", "2026-08-31 22:00", "2026-09-01 02:00")],
    )

    start, end = _parse_month("2026-08")
    analyzer = ExperimentAnalyzer(
        str(csv_path), str(tmp_path / "out"), start_date=start, end_date=end
    )
    lines = analyzer._build_summary_report_lines("TITLE")

    assert len(analyzer.df) == 1
    assert "Total Completed Epochs: 0" in lines
    assert "Completed Models (30 epochs done): 0" in lines


def test_executive_summary_lists_gpu_devices_and_sites(tmp_path):
    csv_path = write_jobs_csv(
        tmp_path / "jobs.csv",
        [
            ("run0-train_epoch0", "2026-08-20 10:00", "2026-08-20 12:00"),
            ("run0-train_epoch1", "2026-09-10 10:00", "2026-09-10 12:00"),
            ("run0-train_epoch2", "2026-09-11 10:00", "2026-09-11 12:00"),
        ],
    )
    df = pd.read_csv(csv_path)
    df["Number of GPUs"] = 1
    df["GPU Device Name"] = ["NVIDIA A100", "GPUs_MIG_25b92ee8", "NVIDIA H200"]
    df["Targeted Resource"] = ["delta", "ospool", "ospool"]
    df["GLIDEIN Resource Name"] = [None, None, "UCSD-NRP"]
    df.to_csv(csv_path, index=False)

    start, end = _parse_month("2026-09")
    analyzer = ExperimentAnalyzer(
        str(csv_path), str(tmp_path / "out"), start_date=start, end_date=end
    )
    lines = analyzer._build_executive_summary_lines()

    assert [line.split()[-2:] for line in lines if line.startswith("GPU Device Types")] == [
        ["2", "3"]
    ]
    assert [line.split()[-2:] for line in lines if line.startswith("Sites ")] == [["2", "3"]]
    assert "  • NVIDIA A100" in lines
    assert "  • MIG slice (model not recorded)  [this period]" in lines
    assert "  • UCSD-NRP  [this period]" in lines
    assert "  • delta" in lines
    assert "  • ospool (site not recorded)  [this period]" in lines
