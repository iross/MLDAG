import pytest

pytest.importorskip("htcondor2", reason="htcondor2 is a Linux-only dependency; daggen.py is untestable elsewhere")

from mldag.constants import DEFAULT_CLASSAD_FIELDS_FILE
from mldag.daggen import Job, get_script
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
