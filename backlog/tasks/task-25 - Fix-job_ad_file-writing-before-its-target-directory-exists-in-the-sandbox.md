---
id: TASK-25
title: Capture job resource usage via event-log parsing (job_ad_file is not real)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-19 16:10'
updated_date: '2026-08-19 17:27'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correction to this task's original diagnosis: live testing (condor_submit returned "WARNING: the line 'job_ad_file = ...' was unused by condor_submit. Is it a typo?") confirmed job_ad_file is not a real HTCondor submit command. condor_submit silently ignores unknown lines by default, which is why this went unnoticed since the mechanism was first added -- the directory-race theory this task originally shipped (a .keep marker + transfer_input_files append) was a real, well-reasoned fix for a real-looking bug, but for a mechanism that was never functional in the first place. That fix has been reverted (mldag/daggen.py no longer emits job_ad_file or the now-pointless .keep/transfer_input_files line).

The actual, verified mechanism: HTCondor's job event log already contains everything needed. The "005 Job terminated" event's body includes a multi-line "Partitionable Resources" usage banner (Cpus/GPUs/Memory Usage columns, TimeExecute (s)) -- confirmed against 6 real entries pulled from a live metl.log, including an edge case where the GPUs row's Usage column is blank on a 4-GPU job. mldag/provenance/log_monitor.py (the provenance NDJSON pipeline) previously didn't handle event code 005 at all; mldag/monitor/dagman.py (a separate, non-provenance tool) handles 005 but only extracts bytes-transferred, not the usage banner.

Separately, HTCondor does write a real per-job ClassAd snapshot automatically into every job's sandbox as $_CONDOR_JOB_AD (a.k.a. .job.ad) -- confirmed against HTCondor's docs and empirically. Per the docs, this snapshot is static as of job start and is not updated during the run, so it's a source for submit-time attributes (Arguments, Request*, GLIDEIN_ResourceName) but not final resource-usage numbers. Capturing that is a separate, not-yet-implemented piece of work (see follow-up task).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 mldag/daggen.py no longer emits a job_ad_file line or the transfer_input_files/.keep marker that only existed to support it
- [x] #2 mldag/provenance/log_monitor.py handles event code 005, accumulating its multi-line Partitionable Resources banner across lines and emitting one job.resource_usage event (wall_time_s, cpu_usage, peak_memory_mb, gpu_usage, gpu_ids) once the "..." event terminator confirms the banner is complete
- [x] #3 The GPUs row's Usage column being blank (observed on a real multi-GPU job) does not prevent capturing gpu_ids or other fields, and does not raise
- [x] #4 A banner that never matches any known row does not emit an empty job.resource_usage event
- [x] #5 State for an in-progress 005 accumulation persists correctly across poll() calls (mirrors the existing multiline_state pattern used for 000/DAGNodeName tracking)
- [x] #6 "job.resource_usage" is a valid event type in mldag/provenance/events.py's schema
- [x] #7 Tests validate the row-parsing regexes against real captured banner text (not synthetic approximations) for both the normal case and the blank-GPU-usage case
- [x] #8 Module docstring's ClusterId→run_id resolution-order documentation no longer claims the .ad-file tier is functional
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CORRECTION (supersedes the notes below, kept for history): the original diagnosis was wrong. job_ad_file is not a real HTCondor submit command -- confirmed by actually running condor_submit against a live pool and getting "WARNING: the line 'job_ad_file = some_dumb_test' was unused by condor_submit. Is it a typo?" condor_submit silently ignores unrecognized lines by default, which is exactly why the original .keep/transfer_input_files fix (shipped in rc16) never surfaced an error despite doing nothing useful.

Reverted from mldag/daggen.py: the job_ad_file line and the transfer_input_files/.keep marker in both get_submit_description() and get_ospool_submit_description(), plus the .keep file creation in main().

Implemented the real fix in mldag/provenance/log_monitor.py: added "005" to _CODES, and a new _accumulate_usage_field() that parses the Partitionable Resources banner rows (Cpus/GPUs/Memory Usage columns via regexes tolerant of a blank Usage column, TimeExecute (s) as a single value, GPU id list from the GPUs row's trailing quoted string). monitor_once()'s per-line loop accumulates these into multiline_state["usage_fields"] from the "005" header line until a "..." terminator line, at which point it resolves run_id via the existing _resolve_run_id() and emits a new "job.resource_usage" event type (added to events.py's VALID_EVENT_TYPES) via emit_event(). State persists across poll() calls the same way the existing 000/DAGNodeName multiline tracking already does.

All parsing regexes were verified against real banner text extracted directly from a live metl.log (6 real "005" events), including the discovered edge case where a 4-GPU job's GPUs row has a blank Usage column -- confirmed the regex still correctly parses Request/Allocated and the GPU id list in that case, and correctly omits gpu_usage rather than mis-parsing a wrong value into it.

Also corrected the module's docstring (ClusterId -> run_id resolution order) to no longer claim the .ad-file tier is functional -- it's dead code today (a future task could wire it to a real ClassAd source), which also explains why task-22's "unknown:<cluster_id>" investigation likely never found a root cause: tier 3 of that resolution chain has never worked either.

Separately confirmed (but not implemented -- see follow-up task): HTCondor does write a real per-job ClassAd snapshot to $_CONDOR_JOB_AD (.job.ad) in every job's sandbox automatically, no submit-file config needed. Per HTCondor's docs this snapshot is static as of job start (not updated during the run), so it covers Arguments/Request*/GLIDEIN_ResourceName but not final usage numbers -- those come from the 005 banner instead, which is what this task implements.

Tests: tests/provenance/test_log_monitor.py gained unit tests for _accumulate_usage_field (full banner, blank-GPU-usage, unrelated lines ignored) and monitor_once integration tests (005 emits job.resource_usage, blank GPU usage omits the field without raising, a banner with no matching rows emits nothing, and state persists correctly across a poll() boundary mid-banner). tests/test_daggen.py's task-25 tests were rewritten to assert job_ad_file/.keep no longer appear in generated submit descriptions. Full suite: 249 passed, 1 skipped (tests/test_daggen.py needs htcondor2, unavailable on macOS). ruff clean on all touched files.

Not yet verified against a real live job (same environment limitation as before) -- recommend checking a real run's provenance NDJSON for a job.resource_usage event once this ships.

--- Original (incorrect) notes, kept for history ---
Fixed in mldag/daggen.py: main() now touches a PROVENANCE_DIR/.keep marker file alongside the existing Path(PROVENANCE_DIR).mkdir(...) call. get_submit_description() and get_ospool_submit_description() both append `transfer_input_files = $(transfer_input_files), {PROVENANCE_DIR}/.keep` before the job_ad_file line -- the self-reference ($(transfer_input_files)) appends to whatever the experiment's own submit_template already set, rather than overwriting it. Since preserve_relative_paths = true is already set in the live Experiment.yaml, transferring output/provenance/.keep as an input file materializes that exact nested directory structure in the sandbox before the job starts. [Superseded above: job_ad_file itself was never a real submit command, so this fix -- while internally consistent -- addressed a mechanism that doesn't exist.]
<!-- SECTION:NOTES:END -->
