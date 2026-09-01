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
    "Args": "pretrain_local.sh 30 run-abc123 42",
    "RequestCpus": 4,
    "RequestMemory": 65536,
    "RequestGPUs": 1,
    "JOBGLIDEIN_ResourceName": "CHTC-Spark-CE1",
    "MachineAttrGLIDEIN_ResourceName0": "CHTC-Spark-CE1",
    "MachineAttrMachine0": "gpu08.chtc.wisc.edu",
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
    assert fields["glidein_resource_name"] == "CHTC-Spark-CE1"
    assert fields["machine"] == "gpu08.chtc.wisc.edu"
    # cluster_id/proc_id are structural identifiers, always included
    # regardless of the configured mapping -- not "leaked" past a blocklist.
    assert fields["cluster_id"] == 12345


def test_capture_job_ad_fields_proc_id_included_when_present(tmp_path):
    job_ad = tmp_path / ".job.ad"
    _write_job_ad(job_ad, {**SAMPLE_JOB_AD, "ProcId": 3})
    with patch.dict("os.environ", {"_CONDOR_JOB_AD": str(job_ad)}):
        fields = capture_job_ad_fields()

    assert fields["proc_id"] == 3


def test_capture_job_ad_fields_no_cluster_id_when_absent_from_ad(tmp_path):
    job_ad = tmp_path / ".job.ad"
    _write_job_ad(job_ad, {k: v for k, v in SAMPLE_JOB_AD.items() if k != "ClusterId"})
    with patch.dict("os.environ", {"_CONDOR_JOB_AD": str(job_ad)}):
        fields = capture_job_ad_fields()

    assert "cluster_id" not in fields


def test_capture_job_ad_fields_custom_fields_file(tmp_path):
    job_ad = tmp_path / ".job.ad"
    _write_job_ad(job_ad, SAMPLE_JOB_AD)
    fields_file = tmp_path / "provenance_fields.yaml"
    fields_file.write_text("fields:\n  - RequestCpus\n")

    with patch.dict("os.environ", {"_CONDOR_JOB_AD": str(job_ad)}):
        fields = capture_job_ad_fields(fields_file)

    # cluster_id is always included even when the configured mapping is
    # narrowed down to just RequestCpus.
    assert fields == {"request_cpus": 4, "cluster_id": 12345}


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
