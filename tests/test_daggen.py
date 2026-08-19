import pytest

pytest.importorskip("htcondor2", reason="htcondor2 is a Linux-only dependency; daggen.py is untestable elsewhere")

from mldag.constants import DEFAULT_CLASSAD_FIELDS_FILE
from mldag.daggen import Job, get_ospool_submit_description, get_script, get_submit_description
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


def test_get_submit_description_does_not_add_job_ad_file():
    """job_ad_file is not a real HTCondor submit command -- condor_submit silently
    ignores it (confirmed against a live pool: "WARNING: the line 'job_ad_file = ...'
    was unused by condor_submit"). Resource usage now comes from log_monitor.py
    parsing the event log's 005 termination banner instead; daggen.py must not
    emit a job_ad_file line or the transfer_input_files marker that only existed
    to support it."""
    experiment = _experiment("transfer_input_files = pretrain.sh, data.tar.gz\nqueue\n")
    submit = get_submit_description(_job(), Resource(), config={}, experiment=experiment)
    assert "transfer_input_files = pretrain.sh, data.tar.gz\n" in submit
    assert "job_ad_file" not in submit
    assert ".keep" not in submit


def test_get_ospool_submit_description_does_not_add_job_ad_file():
    experiment = _experiment("transfer_input_files = pretrain.sh\nqueue\n")
    submit = get_ospool_submit_description(config={}, experiment=experiment)
    assert "transfer_input_files = pretrain.sh\n" in submit
    assert "job_ad_file" not in submit
    assert ".keep" not in submit
