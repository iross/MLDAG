"""Zero-config execute-node provenance auto-capture.

Detects that the current process is running inside an HTCondor job and, if
so, writes site_info.json (the contract mldag.provenance.watcher's
_load_site_info already expects) and emits a job.assigned event -- with no
call required from the training script itself. See mldag/provenance/_autorun.py
for the interpreter-startup hook that makes this actually zero-config (a .pth
file, installed by this package's build hook -- see hatch_build.py -- runs
_autorun at the start of every Python process in an environment where this
package is installed).

Detection reuses $_CONDOR_JOB_AD, the same env var mldag.provenance.jobad
already relies on to locate the job's ClassAd snapshot (see its module
docstring) -- one detection signal, not two independently-drifting ones.

Idempotency: HTCondor keeps one job pinned to one sandbox directory for its
whole lifetime, and a single job can start many Python interpreters (a
wrapper script's own `python3 --version` probe, the training process itself,
subprocesses it launches, ...). A marker file in the sandbox stops every
invocation after the first from re-emitting job.assigned.

Best-effort by design (AC#7): capture_and_emit() never raises. A missing
$_CONDOR_JOB_AD, an unreadable ClassAd, no nvidia-smi/no GPU, a read-only
sandbox -- none of it should ever be able to affect the training job's own
behavior, output, or exit code. Contrast this with the hand-rolled heredocs
this module replaces in pretrain_local.sh/pretrain.sh, which ran their whole
capture step as `_provenance_capture_and_emit() { ... } || exit 1` -- any
uncaught exception in there aborted the job outright.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

from mldag.provenance.events import _DEFAULT_LOG_DIR, emit_event
from mldag.provenance.jobad import capture_job_ad_fields

_MARKER_FILENAME = ".mldag_provenance_autocapture_done"

_GPU_INFO_DEFAULTS = {"gpu_count": 0, "gpu_model": "none", "gpu_id": "none"}


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def detect_condor_job() -> bool:
    """Best-effort HTCondor-job detection via $_CONDOR_JOB_AD presence."""
    return bool(os.environ.get("_CONDOR_JOB_AD"))


def default_gpu_info() -> tuple[dict, str]:
    """Best-effort, torch-free GPU model/count/id.

    Tries nvidia-smi first (a plain CLI call, no Python GPU library needed).
    Falls back to /proc/driver/nvidia/gpus/*/information -- the same file
    pretrain_local.sh's/pretrain.sh's heredocs already parsed as their own
    fallback for GPU UUID when torch lacked a `.uuid` attribute, generalized
    here to also recover the model name and to work as the primary path for
    any caller that doesn't have torch at all.

    Returns (gpu_info, source) where source records which mechanism found it
    (or "none" if neither did), for the event's site_info_source field.
    """
    smi_output = _run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"])
    lines = [line for line in smi_output.splitlines() if line.strip()]
    if lines:
        name, _, uuid = lines[0].partition(",")
        return (
            {
                "gpu_count": len(lines),
                "gpu_model": name.strip(),
                "gpu_id": uuid.strip(),
            },
            "nvidia_smi",
        )

    info_files = sorted(glob.glob("/proc/driver/nvidia/gpus/*/information"))
    if not info_files:
        return dict(_GPU_INFO_DEFAULTS), "none"
    try:
        text = Path(info_files[0]).read_text()
    except OSError:
        return dict(_GPU_INFO_DEFAULTS), "none"
    model_match = re.search(r"Model:\s+(.+)", text)
    uuid_match = re.search(r"GPU UUID:\s+(\S+)", text)
    return (
        {
            "gpu_count": len(info_files),
            "gpu_model": model_match.group(1).strip() if model_match else "unknown",
            "gpu_id": uuid_match.group(1) if uuid_match else "unknown",
        },
        "proc_driver_nvidia",
    )


def gather_site_info(
    gpu_info: dict | None = None, gpu_info_source: str = "torch_cuda"
) -> tuple[dict, str]:
    """Return (site_info, site_info_source).

    site_info is exactly the {hostname, slot, gpu_model, gpu_count, gpu_id}
    shape watcher.py's _load_site_info splits out of site_info.json. Pass
    gpu_info to supply caller-detected values (e.g. pretrain_local.sh's own
    torch.cuda introspection, more precise than this module's nvidia-smi/proc
    fallback) instead of auto-detecting.
    """
    hostname = _run(["hostname", "-f"]) or _run(["hostname"]) or "unknown"
    slot = os.environ.get("_CONDOR_SLOT", "unknown")
    if gpu_info is not None:
        gpu, source = {**_GPU_INFO_DEFAULTS, **gpu_info}, gpu_info_source
    else:
        gpu, source = default_gpu_info()
    return {"hostname": hostname, "slot": slot, **gpu}, source


def _mldag_version() -> str:
    # MLDAG_VERSION is baked into every job's environment by daggen.py at DAG
    # generation time -- prefer it so the recorded version matches exactly
    # what was submitted, even if a different mldag happens to be importable
    # in the job's Python environment. Falls back to the installed package's
    # own version for jobs not launched through a generated DAG.
    version = os.environ.get("MLDAG_VERSION")
    if version:
        return version
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as pkg_version

        return pkg_version("mldag")
    except PackageNotFoundError:
        return "unknown"


def gather_env_info() -> dict:
    smi_output = _run(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", smi_output)
    return {
        "python": sys.version.split()[0],
        "cuda": cuda_match.group(1) if cuda_match else "unknown",
        "code_commit": _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown",
        "mldag_version": _mldag_version(),
    }


def capture_and_emit(
    *, gpu_info: dict | None = None, fields_file: str | Path | None = None
) -> dict | None:
    """Best-effort: write site_info.json and emit job.assigned.

    Returns the emitted payload, or None if this isn't (detectably) an
    HTCondor job, this job's sandbox was already captured (idempotency
    marker), or capture failed for any reason -- callers don't need to check
    the return value to stay safe; it's provided for callers that want to
    inspect what was captured (e.g. tests, or a caller printing it for
    debugging).
    """
    try:
        if not detect_condor_job():
            return None
        marker = Path(_MARKER_FILENAME)
        if marker.exists():
            return None

        import json

        run_id = os.environ.get("PROVENANCE_RUN_ID", "unknown")
        log_dir = Path(os.environ.get("PROVENANCE_LOG_DIR", _DEFAULT_LOG_DIR))

        site_info, site_info_source = gather_site_info(gpu_info)
        env_info = gather_env_info()
        jobad_fields = capture_job_ad_fields(fields_file)

        payload = {**site_info, **env_info, **jobad_fields}
        Path("site_info.json").write_text(json.dumps(payload, indent=2))

        emit_event(
            "job.assigned",
            run_id,
            log_dir=log_dir,
            source="execute_node_autocapture",
            site_info_source=site_info_source,
            **payload,
        )
        marker.write_text("")
        return payload
    except Exception:  # noqa: BLE001 -- deliberately blind: AC#7's whole point
        # is that any failure here (best-effort) must never affect the
        # training job's own behavior, output, or exit code.
        return None


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Best-effort execute-node provenance auto-capture: write "
        "site_info.json and emit a job.assigned event. Safe to call "
        "unconditionally -- no-ops outside an HTCondor job or if this job's "
        "sandbox was already captured."
    )
    parser.add_argument("--gpu-model", help="Override auto-detected GPU model name")
    parser.add_argument(
        "--gpu-count", type=int, help="Override auto-detected GPU count"
    )
    parser.add_argument("--gpu-id", help="Override auto-detected GPU UUID")
    parser.add_argument(
        "--fields-file",
        default=None,
        help="YAML file mapping ClassAd attributes to provenance schema keys "
        "(see mldag.provenance.post.load_classad_field_mapping).",
    )
    args = parser.parse_args()

    gpu_info = None
    if any(v is not None for v in (args.gpu_model, args.gpu_count, args.gpu_id)):
        gpu_info = {
            "gpu_model": args.gpu_model if args.gpu_model is not None else "unknown",
            "gpu_count": args.gpu_count if args.gpu_count is not None else 0,
            "gpu_id": args.gpu_id if args.gpu_id is not None else "unknown",
        }

    payload = capture_and_emit(gpu_info=gpu_info, fields_file=args.fields_file)
    print(json.dumps(payload) if payload is not None else "{}")


if __name__ == "__main__":
    main()
