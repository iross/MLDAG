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
