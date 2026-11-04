---
id: TASK-42
title: >-
  Push execute-node provenance back via Chirp instead of relying on output-file
  transfer
status: To Do
assignee: []
created_date: '2026-09-22 20:03'
labels: []
dependencies:
  - TASK-31
references:
  - mldag/provenance/autocapture.py
  - mldag/provenance/db.py
  - mldag/provenance/log_monitor.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
output/provenance/<run_id>.ndjson is written from two independent sides for the same job: log_monitor.py (task-38's predecessor architecture) runs as a DAGMan SERVICE on the AP side, continuously tailing metl.log and appending job.executing/transfer.*/job.resource_usage events there in real time while the job runs; separately, mldag.provenance.autocapture (task-38) writes its own copy of the same relative path inside the execute-node's sandbox, containing just job.assigned. transfer_output_files = output (or equivalent) copies that sandbox output/ tree back into the job's IWD at job completion, and HTCondor's output transfer is a plain file overwrite, not a merge -- so the execute-node's copy silently replaces whatever log_monitor.py had already accumulated AP-side for that run_id. This is not new in task-38; the old hand-rolled heredocs it replaced had the identical race. Confirmed real (not just theoretical) via manual review of daggen.py/Experiment.yaml's transfer_output_files config and log_monitor.py's SERVICE-node write path.

Chirp (HTCondor's job<->AP RPC channel, enabled via want_io_proxy = true in the submit description) sidesteps this entirely: a job can push data to the AP side the moment it's captured, without waiting for job exit or output-file transfer at all. Verified empirically this session via a minimal ulog() round-trip test (htchirp, the pure-Python PyPI package -- no compiled htcondor2 bindings needed, so it pip-installs cleanly the same way mldag itself does in pretrain_local.sh) -- confirmed working through both Docker and Singularity/Apptainer container-universe jobs (event code 008 landed in the job's own event log ~3s after the ulog() call).

Design: autocapture.capture_and_emit() keeps its existing local write unchanged (still the fallback), and additionally attempts a best-effort chirp put() of the same payload to a DISTINCTLY-named file on the AP side -- <run_id>.execute_node.ndjson, not <run_id>.ndjson -- so it can never collide with what log_monitor.py is writing to the shared name. db.py's build_database event-dir scan already globs *.ndjson (Path(d).glob("*.ndjson"), confirmed by reading db.py this session) rather than matching a specific filename pattern, so the distinctly-named file is picked up automatically with zero changes needed to db.py, log_monitor.py, or event_log_scan.py.

want_io_proxy is deliberately NOT being made a default in daggen.py's generated submit descriptions (explicit user decision) -- it stays opt-in per experiment/submit description. This task must therefore degrade cleanly (silent no-op, same best-effort contract as the rest of autocapture.py) whenever chirp isn't available: htchirp not installed, WantIOProxy not set on the job, connection failure, or any other error.

Blocked on task-31 (split mldag into provenance and DAG-generation-and-monitoring packages) by explicit request: this adds a new dependency (htchirp) that belongs on the provenance side only, and doing the split first keeps that dependency out of the heavier DAG-generation/monitoring package.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 autocapture.capture_and_emit() attempts a best-effort chirp put() of its payload to <run_id>.execute_node.ndjson on the AP side, immediately after (and independently of) the existing local site_info.json/NDJSON write, which is unchanged
- [ ] #2 Any chirp failure (htchirp not installed, WantIOProxy not set on the job, connection error, timeout, or anything else) is caught and silently no-ops -- never affects the training job's behavior, output, or exit code, matching autocapture.py's existing best-effort contract
- [ ] #3 No changes required to daggen.py -- want_io_proxy stays opt-in per submit description, not a default; the distinctly-named output file requires no changes to db.py/log_monitor.py/event_log_scan.py either (db.py's *.ndjson glob already picks it up)
- [ ] #4 Running mldag-query db build --event-dir against a provenance directory containing both log_monitor.py's AP-side <run_id>.ndjson and the chirped-back <run_id>.execute_node.ndjson ingests events from both files for the same run_id without conflict
- [ ] #5 Verified end-to-end on a real job (building on this session's confirmed ulog() round-trip): the chirped-back file actually appears on the AP side with the correct job.assigned payload, for both a Docker and a Singularity/Apptainer container-universe job
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add htchirp as a dependency (decide: hard runtime dependency vs. soft/optional import with graceful ImportError handling, matching how torch is already treated as optional in pretrain_local.sh/pretrain.sh)
2. Implement the chirp put() step in autocapture.capture_and_emit(), wrapped in the same best-effort try/except as the rest of the function, writing to <run_id>.execute_node.ndjson
3. Verify db build --event-dir ingests both the AP-side and chirped-back files for the same run_id cleanly (no duplicate-row or overwrite issues in the events table)
4. End-to-end test on a real job with want_io_proxy = true set manually in the submit description (Docker and Singularity/Apptainer), confirming the file lands AP-side without waiting for job completion/output transfer
5. End-to-end test WITHOUT want_io_proxy set, confirming clean silent no-op (no job failure, no error surfaced)
6. Document the want_io_proxy opt-in requirement somewhere a user would find it before wondering why chirp delivery isn't happening for their jobs (e.g. autocapture.py's module docstring, or backlog/docs/provenance_design.md)
<!-- SECTION:PLAN:END -->
