from pathlib import Path
from unittest.mock import patch

import pytest

from mldag.provenance.jobad import capture_job_ad_fields


def _write_job_ad(path: Path, attrs: dict) -> None:
    lines = []
    for k, v in attrs.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    path.write_text("\n".join(lines) + "\n")


SAMPLE_JOB_AD = {
    "Arguments": "pretrain_local.sh 30 run-abc123 42",
    "RequestCpus": 4,
    "RequestMemory": 65536,
    "RequestGpus": 1,
    "GLIDEIN_ResourceName": "CHTC-Spark-CE1",
    "ClusterId": 12345,
}


def test_capture_job_ad_fields_env_var_unset(monkeypatch):
    monkeypatch.delenv("_CONDOR_JOB_AD", raising=False)
    assert capture_job_ad_fields() == {}


def test_capture_job_ad_fields_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.ad"
    with patch.dict("os.environ", {"_CONDOR_JOB_AD": str(missing)}):
        assert capture_job_ad_fields() == {}


def test_capture_job_ad_fields_returns_default_mapping_fields(tmp_path):
    job_ad = tmp_path / ".job.ad"
    _write_job_ad(job_ad, SAMPLE_JOB_AD)
    with patch.dict("os.environ", {"_CONDOR_JOB_AD": str(job_ad)}):
        fields = capture_job_ad_fields()

    assert fields["arguments"] == "pretrain_local.sh 30 run-abc123 42"
    assert fields["request_cpus"] == 4
    assert fields["request_memory"] == 65536
    assert fields["request_gpus"] == 1
    assert fields["resource_name"] == "CHTC-Spark-CE1"
    # ClusterId isn't in the default mapping, so it must not leak through
    assert "ClusterId" not in fields
    assert "cluster_id" not in fields


def test_capture_job_ad_fields_custom_fields_file(tmp_path):
    job_ad = tmp_path / ".job.ad"
    _write_job_ad(job_ad, SAMPLE_JOB_AD)
    fields_file = tmp_path / "provenance_fields.yaml"
    fields_file.write_text("fields:\n  - RequestCpus\n")

    with patch.dict("os.environ", {"_CONDOR_JOB_AD": str(job_ad)}):
        fields = capture_job_ad_fields(fields_file)

    assert fields == {"request_cpus": 4}


def test_capture_job_ad_fields_still_enforces_sensitive_key_blocklist(tmp_path):
    """Confirms this reuses load_classad_field_mapping's blocklist rather than
    reimplementing ClassAd handling and accidentally bypassing it."""
    job_ad = tmp_path / ".job.ad"
    _write_job_ad(job_ad, {**SAMPLE_JOB_AD, "Environment": "WANDB_API_KEY=secret"})
    fields_file = tmp_path / "provenance_fields.yaml"
    fields_file.write_text("fields:\n  Environment: env\n")

    with patch.dict("os.environ", {"_CONDOR_JOB_AD": str(job_ad)}):
        with pytest.raises(ValueError, match="Environment"):
            capture_job_ad_fields(fields_file)
