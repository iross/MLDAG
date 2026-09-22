import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from mldag.provenance import autocapture


def _write_job_ad(path: Path, attrs: dict) -> None:
    lines = []
    for k, v in attrs.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    path.write_text("\n".join(lines) + "\n")


SAMPLE_JOB_AD = {
    "ClusterId": 555,
    "ProcId": 0,
    "Args": "pretrain_local.sh 30 run-abc123 42",
    "RequestCpus": 4,
    "Environment": "PROVENANCE_RUN_ID=run-abc123",
}


# --- detect_condor_job ---


def test_detect_condor_job_true_when_env_set(monkeypatch):
    monkeypatch.setenv("_CONDOR_JOB_AD", "/some/path.ad")
    assert autocapture.detect_condor_job() is True


def test_detect_condor_job_false_when_unset(monkeypatch):
    monkeypatch.delenv("_CONDOR_JOB_AD", raising=False)
    assert autocapture.detect_condor_job() is False


# --- default_gpu_info ---


def test_default_gpu_info_uses_nvidia_smi(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Tesla T4, GPU-abc123\n", stderr=""
        )

    monkeypatch.setattr(autocapture.subprocess, "run", fake_run)

    info, source = autocapture.default_gpu_info()

    assert info == {"gpu_count": 1, "gpu_model": "Tesla T4", "gpu_id": "GPU-abc123"}
    assert source == "nvidia_smi"


def test_default_gpu_info_counts_multiple_gpus(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Tesla T4, GPU-aaa\nTesla T4, GPU-bbb\n", stderr=""
        )

    monkeypatch.setattr(autocapture.subprocess, "run", fake_run)

    info, source = autocapture.default_gpu_info()

    assert info["gpu_count"] == 2
    assert source == "nvidia_smi"


def test_default_gpu_info_falls_back_to_proc_when_nvidia_smi_absent(
    monkeypatch, tmp_path
):
    def fake_run(cmd, **kwargs):
        raise OSError("nvidia-smi not found")

    monkeypatch.setattr(autocapture.subprocess, "run", fake_run)

    proc_file = tmp_path / "0" / "information"
    proc_file.parent.mkdir(parents=True)
    proc_file.write_text("Model: \t\t Tesla T4\nGPU UUID: \t\t GPU-proc-fallback\n")
    monkeypatch.setattr(
        autocapture.glob,
        "glob",
        lambda pattern: [str(proc_file)] if "nvidia" in pattern else [],
    )

    info, source = autocapture.default_gpu_info()

    assert info == {
        "gpu_count": 1,
        "gpu_model": "Tesla T4",
        "gpu_id": "GPU-proc-fallback",
    }
    assert source == "proc_driver_nvidia"


def test_default_gpu_info_no_gpu_anywhere(monkeypatch):
    monkeypatch.setattr(
        autocapture.subprocess,
        "run",
        lambda cmd, **kwargs: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(autocapture.glob, "glob", lambda pattern: [])

    info, source = autocapture.default_gpu_info()

    assert info == {"gpu_count": 0, "gpu_model": "none", "gpu_id": "none"}
    assert source == "none"


# --- gather_site_info ---


def test_gather_site_info_uses_override(monkeypatch):
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "node01.example.edu")
    monkeypatch.setenv("_CONDOR_SLOT", "slot1_1")

    site_info, source = autocapture.gather_site_info(
        {"gpu_model": "A100", "gpu_count": 1, "gpu_id": "GPU-torch"}
    )

    assert site_info == {
        "hostname": "node01.example.edu",
        "slot": "slot1_1",
        "gpu_model": "A100",
        "gpu_count": 1,
        "gpu_id": "GPU-torch",
    }
    assert source == "torch_cuda"


def test_gather_site_info_fills_defaults_for_partial_override(monkeypatch):
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "host")
    monkeypatch.setenv("_CONDOR_SLOT", "unknown")

    site_info, _ = autocapture.gather_site_info({"gpu_model": "A100"})

    assert site_info["gpu_count"] == 0
    assert site_info["gpu_id"] == "none"


def test_gather_site_info_auto_detects_when_no_override(monkeypatch):
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "host")
    monkeypatch.setattr(
        autocapture,
        "default_gpu_info",
        lambda: ({"gpu_count": 0, "gpu_model": "none", "gpu_id": "none"}, "none"),
    )

    site_info, source = autocapture.gather_site_info(None)

    assert source == "none"
    assert site_info["gpu_model"] == "none"


# --- gather_env_info ---


def test_gather_env_info_reads_mldag_version_from_env(monkeypatch):
    monkeypatch.setenv("MLDAG_VERSION", "0.1.0rc26")
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "")

    env_info = autocapture.gather_env_info()

    assert env_info["mldag_version"] == "0.1.0rc26"
    assert env_info["cuda"] == "unknown"
    assert env_info["code_commit"] == "unknown"


def test_gather_env_info_falls_back_to_installed_package_version(monkeypatch):
    monkeypatch.delenv("MLDAG_VERSION", raising=False)
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "")

    env_info = autocapture.gather_env_info()

    assert env_info["mldag_version"] != "unknown"


def test_gather_env_info_parses_cuda_version_from_nvidia_smi(monkeypatch):
    monkeypatch.setenv("MLDAG_VERSION", "0.1.0rc26")

    def fake_run(cmd):
        if cmd[0] == "nvidia-smi":
            return "... CUDA Version: 12.4 ..."
        return ""

    monkeypatch.setattr(autocapture, "_run", fake_run)

    assert autocapture.gather_env_info()["cuda"] == "12.4"


# --- capture_and_emit ---


def test_capture_and_emit_noop_outside_condor_job(monkeypatch, tmp_path):
    monkeypatch.delenv("_CONDOR_JOB_AD", raising=False)
    monkeypatch.chdir(tmp_path)

    assert autocapture.capture_and_emit() is None
    assert not (tmp_path / "site_info.json").exists()


def test_capture_and_emit_noop_when_already_captured(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("_CONDOR_JOB_AD", str(tmp_path / "job.ad"))
    (tmp_path / autocapture._MARKER_FILENAME).write_text("")

    assert autocapture.capture_and_emit() is None
    assert not (tmp_path / "site_info.json").exists()


def test_capture_and_emit_writes_site_info_and_emits_event(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    job_ad = tmp_path / "job.ad"
    _write_job_ad(job_ad, SAMPLE_JOB_AD)
    monkeypatch.setenv("_CONDOR_JOB_AD", str(job_ad))
    monkeypatch.setenv("PROVENANCE_RUN_ID", "run-abc123")
    monkeypatch.setenv("PROVENANCE_LOG_DIR", str(tmp_path / "output" / "provenance"))
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "node01")
    monkeypatch.setattr(
        autocapture,
        "default_gpu_info",
        lambda: ({"gpu_count": 0, "gpu_model": "none", "gpu_id": "none"}, "none"),
    )

    payload = autocapture.capture_and_emit()

    assert payload["arguments"] == "pretrain_local.sh 30 run-abc123 42"
    assert payload["cluster_id"] == 555
    assert payload["hostname"] == "node01"

    site_info = json.loads((tmp_path / "site_info.json").read_text())
    assert site_info["arguments"] == "pretrain_local.sh 30 run-abc123 42"
    assert site_info["hostname"] == "node01"
    assert (
        "source" not in site_info
    )  # site_info.json never carries the event envelope fields

    ndjson_path = tmp_path / "output" / "provenance" / "run-abc123.ndjson"
    events = [json.loads(line) for line in ndjson_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["type"] == "job.assigned"
    assert events[0]["run_id"] == "run-abc123"
    assert events[0]["source"] == "execute_node_autocapture"
    assert events[0]["site_info_source"] == "none"
    assert events[0]["cluster_id"] == 555
    assert events[0]["arguments"] == "pretrain_local.sh 30 run-abc123 42"

    assert (tmp_path / autocapture._MARKER_FILENAME).exists()


def test_capture_and_emit_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    job_ad = tmp_path / "job.ad"
    _write_job_ad(job_ad, SAMPLE_JOB_AD)
    monkeypatch.setenv("_CONDOR_JOB_AD", str(job_ad))
    monkeypatch.setenv("PROVENANCE_RUN_ID", "run-abc123")
    monkeypatch.setenv("PROVENANCE_LOG_DIR", str(tmp_path / "output" / "provenance"))
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "node01")
    monkeypatch.setattr(
        autocapture,
        "default_gpu_info",
        lambda: ({"gpu_count": 0, "gpu_model": "none", "gpu_id": "none"}, "none"),
    )

    autocapture.capture_and_emit()
    second = autocapture.capture_and_emit()

    assert second is None
    ndjson_path = tmp_path / "output" / "provenance" / "run-abc123.ndjson"
    assert len(ndjson_path.read_text().splitlines()) == 1


def test_capture_and_emit_never_raises_on_internal_failure(monkeypatch, tmp_path):
    """AC#7: any failure inside capture (e.g. an unreadable/corrupt ClassAd
    file that makes capture_job_ad_fields blow up) must be swallowed, not
    propagated -- this is what fixes the old heredocs' `|| exit 1` footgun."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "_CONDOR_JOB_AD", str(tmp_path / "job.ad")
    )  # never written -- unreadable
    monkeypatch.setattr(
        autocapture,
        "capture_job_ad_fields",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert autocapture.capture_and_emit() is None
    assert not (tmp_path / autocapture._MARKER_FILENAME).exists()


def test_capture_and_emit_passes_through_fields_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    job_ad = tmp_path / "job.ad"
    _write_job_ad(job_ad, SAMPLE_JOB_AD)
    monkeypatch.setenv("_CONDOR_JOB_AD", str(job_ad))
    monkeypatch.setenv("PROVENANCE_RUN_ID", "run-abc123")
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "node01")
    monkeypatch.setattr(
        autocapture,
        "default_gpu_info",
        lambda: ({"gpu_count": 0, "gpu_model": "none", "gpu_id": "none"}, "none"),
    )
    fields_file = tmp_path / "provenance_fields.yaml"
    fields_file.write_text("fields:\n  - RequestCpus\n")

    payload = autocapture.capture_and_emit(fields_file=fields_file)

    assert payload == {
        "hostname": "node01",
        "slot": "unknown",
        "gpu_count": 0,
        "gpu_model": "none",
        "gpu_id": "none",
        "python": payload["python"],
        "cuda": "unknown",
        "code_commit": "node01",
        "mldag_version": payload["mldag_version"],
        "request_cpus": 4,
        "cluster_id": 555,
        "proc_id": 0,
    }


# --- main() CLI ---


def test_main_prints_empty_object_outside_condor_job(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("_CONDOR_JOB_AD", raising=False)

    with patch("sys.argv", ["autocapture"]):
        autocapture.main()

    assert capsys.readouterr().out.strip() == "{}"


def test_main_passes_gpu_overrides_through(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    job_ad = tmp_path / "job.ad"
    _write_job_ad(job_ad, SAMPLE_JOB_AD)
    monkeypatch.setenv("_CONDOR_JOB_AD", str(job_ad))
    monkeypatch.setenv("PROVENANCE_RUN_ID", "run-abc123")
    monkeypatch.setattr(autocapture, "_run", lambda cmd: "node01")

    with patch(
        "sys.argv",
        [
            "autocapture",
            "--gpu-model",
            "A100",
            "--gpu-count",
            "1",
            "--gpu-id",
            "GPU-torch",
        ],
    ):
        autocapture.main()

    out = json.loads(capsys.readouterr().out)
    assert out["gpu_model"] == "A100"
    assert out["gpu_count"] == 1
    assert out["gpu_id"] == "GPU-torch"
