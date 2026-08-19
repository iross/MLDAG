"""Captures submit-time job attributes from $_CONDOR_JOB_AD, from within a running job.

HTCondor automatically writes a per-job ClassAd snapshot into the job's own
sandbox at the path named by the $_CONDOR_JOB_AD environment variable -- no
submit-file configuration needed. Per HTCondor's docs (env-of-job.rst): "The
job ad is current as of the start of the job, but is not updated during the
running of the job." So this is a source for static submit-time attributes
(Arguments, RequestCpus/RequestMemory/RequestGpus, GLIDEIN_ResourceName), not
final resource usage -- that comes from mldag.provenance.log_monitor parsing
the event log's 005 termination banner instead (see task-25).

This must be called from *within* the running job (e.g. from an experiment's
training wrapper script), not from post.py on the submit side: $_CONDOR_JOB_AD
points into the job's own sandbox, which isn't transferred back.
"""

from __future__ import annotations

import os
from pathlib import Path

from mldag.provenance.post import (
    load_classad_field_mapping,
    parse_classad,
    resource_fields_from_classad,
)


def capture_job_ad_fields(fields_file: str | Path | None = None) -> dict:
    """Return the configured subset of $_CONDOR_JOB_AD's fields.

    Reuses the same field-mapping file, default mapping, and sensitive-key
    blocklist as post.py's ClassAd handling (see load_classad_field_mapping).

    Returns {} if $_CONDOR_JOB_AD isn't set or unreadable (e.g. running
    outside HTCondor, or an HTCondor version that doesn't set it) -- this is
    best-effort enrichment, never a hard requirement.
    """
    job_ad_path = os.environ.get("_CONDOR_JOB_AD")
    if not job_ad_path:
        return {}
    ad = parse_classad(job_ad_path)
    if not ad:
        return {}
    mapping = load_classad_field_mapping(fields_file)
    return resource_fields_from_classad(ad, mapping)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Print $_CONDOR_JOB_AD's configured fields as JSON, for standalone use/debugging."
    )
    parser.add_argument(
        "--fields-file", default=None,
        help="YAML file mapping ClassAd attributes to provenance schema keys "
             "(see mldag.provenance.post.load_classad_field_mapping).",
    )
    args = parser.parse_args()
    print(json.dumps(capture_job_ad_fields(args.fields_file)))


if __name__ == "__main__":
    main()
