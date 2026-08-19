---
id: TASK-26
title: Capture submit-time job attributes from $_CONDOR_JOB_AD within the job
status: Done
assignee:
  - '@claude'
created_date: '2026-08-19 17:27'
updated_date: '2026-08-19 19:12'
labels: []
dependencies:
  - TASK-24
  - TASK-25
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-25 implemented resource-usage capture (wall_time_s, cpu_usage, peak_memory_mb, gpu_usage) via parsing the HTCondor event log's "005 Job terminated" banner. That covers usage numbers, but not submit-time attributes like Arguments (the original task-24 ask), RequestCpus/RequestMemory/RequestGpus, or GLIDEIN_ResourceName.

HTCondor automatically writes a real per-job ClassAd snapshot into every job's sandbox at $_CONDOR_JOB_AD (conventionally .job.ad) -- no submit-file configuration needed, confirmed against HTCondor's own docs (env-of-job.rst) and empirically on the live pool. Per the docs: "The job ad is current as of the start of the job, but is not updated during the running of the job" -- so this is a source for static, submit-time attributes only, not final resource usage (which task-25 already covers via the event log). Its format is the same as `condor_q -l` output, i.e. the same "attr = value" long-ClassAd text that mldag/provenance/post.py's parse_classad() already parses -- that function is reusable as-is, only the source of the text changes (read $_CONDOR_JOB_AD from within the job, rather than a submit-side file that never existed).

This needs to be read from WITHIN the running job (e.g. wired into pretrain_local.sh's existing _provenance_capture_and_emit-style step, via a new generic mldag entry point), not from post.py on the submit side, since the file lives in the job's own sandbox and doesn't get transferred back automatically.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A generic mldag entry point (not experiment-specific script code) reads $_CONDOR_JOB_AD from within a running job and extracts a configurable set of attributes, reusing parse_classad()/resource_fields_from_classad()/load_classad_field_mapping() from mldag/provenance/post.py rather than duplicating ClassAd-parsing logic
- [x] #2 Arguments, RequestCpus, RequestMemory, RequestGpus, and GLIDEIN_ResourceName are captured by default
- [x] #3 The sensitive-key blocklist from task-24 (starting with Environment) still applies to this capture path
- [x] #4 Missing $_CONDOR_JOB_AD (e.g. running outside HTCondor, or an older HTCondor version) degrades gracefully -- no exception, fields simply omitted
- [x] #5 Wired into pretrain_local.sh (or documented as the integration point for experiment repos) so the captured fields land in an NDJSON event correlated by run_id
- [x] #6 Tests cover parsing a real-shaped $_CONDOR_JOB_AD file and the missing-file degradation case
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented as a new mldag/provenance/jobad.py module: capture_job_ad_fields(fields_file=None) reads $_CONDOR_JOB_AD (returns {} gracefully if the env var is unset or the file can't be parsed) and reuses parse_classad()/resource_fields_from_classad()/load_classad_field_mapping() from post.py unchanged -- no ClassAd-parsing logic duplicated. Extended post.py's _DEFAULT_FIELD_MAPPING with RequestCpus/RequestMemory/RequestGpus (Arguments and GLIDEIN_ResourceName->resource_name were already present from task-24), so the same default mapping now serves both the (currently dead) post-completion path and this new within-job capture. Also corrected post.py's module docstring, which still claimed job_ad_file was a real mechanism.

Wired into pretrain_local.sh's existing _provenance_capture_and_emit() heredoc: imports capture_job_ad_fields, calls it once, and merges the result into the *existing* single job.assigned event (and the site_info.json debug dump) rather than emitting a second, separate event -- same timing (from within the job, near start), same emission call, no new event type.

Tests: tests/provenance/test_jobad.py covers env-var-unset, file-missing, a real-shaped job ad returning the expected default fields (and confirming ClusterId, not in the default mapping, does not leak through), a custom fields_file override, and that the sensitive-key blocklist from task-24 still applies through this path (proves reuse, not reimplementation). Verified pretrain_local.sh's heredoc still parses as valid Python (ast.parse) and the surrounding bash still parses (bash -n) after the edit. Full suite: 254 passed, 1 skipped. ruff clean on all touched files.

Not yet verified against a real live job (same environment limitation as task-25) -- recommend checking a run's job.assigned event for arguments/request_cpus/request_memory/request_gpus/resource_name once this ships.
<!-- SECTION:NOTES:END -->
