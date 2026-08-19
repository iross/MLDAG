import pytest

pytest.importorskip("htcondor2", reason="htcondor2 is a Linux-only dependency; daggen.py is untestable elsewhere")

from mldag.constants import DEFAULT_CLASSAD_FIELDS_FILE
from mldag.daggen import PROVENANCE_DIR, Job, get_ospool_submit_description, get_script, get_submit_description
from mldag.models.experiment import Experiment
from mldag.models.resource import Resource


def _job() -> Job:
    return Job(name="run0-train_epoch0", submit="default_pretrain.sub", epoch=1, run_uuid="run0", tr_id=0)


def test_get_script_bakes_in_default_fields_file():
    script = get_script(_job(), Resource(), config={})
    assert f"--fields-file {DEFAULT_CLASSAD_FIELDS_FILE}" in script


def test_get_script_bakes_in_custom_fields_file():
    script = get_script(_job(), Resource(), config={}, classad_fields_file="custom_fields.yaml")
    assert "--fields-file custom_fields.yaml" in script


def test_get_script_fields_file_precedes_post_hook():
    """--post-hook consumes argparse.REMAINDER, so --fields-file must come before it."""
    script = get_script(_job(), Resource(), config={}, post_hook="notify.sh done")
    post_line = next(line for line in script.splitlines() if line.startswith("SCRIPT POST"))
    assert post_line.index("--fields-file") < post_line.index("--post-hook")


def _experiment(submit_template: str) -> Experiment:
    return Experiment(submit_template=submit_template, vars={})


def test_get_submit_description_preserves_transfer_input_files_and_adds_keep_marker():
    """job_ad_file's target directory must exist before the starter writes it, or the
    write fails silently (see mldag_unknown_provenance_fix.md investigation) -- transferring
    a marker via a self-referencing transfer_input_files line materializes it ahead of time
    without clobbering whatever the experiment's own template already listed."""
    experiment = _experiment("transfer_input_files = pretrain.sh, data.tar.gz\nqueue\n")
    submit = get_submit_description(_job(), Resource(), config={}, experiment=experiment)
    assert "transfer_input_files = pretrain.sh, data.tar.gz\n" in submit
    assert f"transfer_input_files = $(transfer_input_files), {PROVENANCE_DIR}/.keep\n" in submit
    assert f"job_ad_file = {PROVENANCE_DIR}/$(ClusterId).ad\n" in submit


def test_get_submit_description_keep_marker_before_job_ad_file():
    experiment = _experiment("transfer_input_files = pretrain.sh\nqueue\n")
    submit = get_submit_description(_job(), Resource(), config={}, experiment=experiment)
    assert submit.index(".keep") < submit.index("job_ad_file")


def test_get_ospool_submit_description_preserves_transfer_input_files_and_adds_keep_marker():
    experiment = _experiment("transfer_input_files = pretrain.sh\nqueue\n")
    submit = get_ospool_submit_description(config={}, experiment=experiment)
    assert "transfer_input_files = pretrain.sh\n" in submit
    assert f"transfer_input_files = $(transfer_input_files), {PROVENANCE_DIR}/.keep\n" in submit
    assert f"job_ad_file = {PROVENANCE_DIR}/$(ClusterId).ad\n" in submit
