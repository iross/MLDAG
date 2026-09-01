---
id: TASK-37
title: >-
  Fix ClassAd attribute name mismatches in job-ad field capture; add machine
  field
status: Done
assignee:
  - '@claude'
created_date: '2026-09-01 16:03'
updated_date: '2026-09-01 16:11'
labels:
  - provenance
dependencies: []
modified_files:
  - mldag/provenance/post.py
  - mldag/provenance/history_enrich.py
  - mldag/provenance/db.py
  - tests/provenance/test_post.py
  - tests/provenance/test_jobad.py
  - tests/provenance/test_history_enrich.py
  - tests/provenance/test_log_monitor.py
  - tests/provenance/test_event_log_scan.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification against a real job (cluster_id 6141971) found that post.py's _DEFAULT_FIELD_MAPPING and history_enrich.py's _FIELD_MAPPING use ClassAd attribute names that don't match what HTCondor actually writes for this repo's submit descriptions, so arguments/request_gpus/resource_name have silently never been captured despite jobad.py/capture_job_ad_fields() running successfully (cluster_id, proc_id, request_cpus, request_memory, wall_time_s all come through fine).

Confirmed via `grep -i arg $_CONDOR_JOB_AD` and `grep -i resource`/`grep -i machine` on a live job:
- All submit descriptions here use unquoted old-syntax `arguments = ...`, which HTCondor names `Args` in the ClassAd, not `Arguments`.
- HTCondor's GPU request attribute is canonically `RequestGPUs` (capital GPUs), not `RequestGpus`.
- GLIDEIN_ResourceName is not a top-level job ad attribute; it's copied in via HTCondor's job_machine_attrs mechanism as `MachineAttrGLIDEIN_ResourceName0`.
- The same mechanism exposes the matched execute host as `MachineAttrMachine0`, which is not currently captured at all -- useful as an HTCondor-verified cross-check against pretrain_local.sh's self-reported `hostname -f` (relevant to task-22's unknown:&lt;cluster_id&gt; site-resolution investigation).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 post.py's _DEFAULT_FIELD_MAPPING and history_enrich.py's _FIELD_MAPPING map Args (not Arguments) to arguments
- [x] #2 Both mappings map RequestGPUs (not RequestGpus) to request_gpus
- [x] #3 Both mappings map MachineAttrGLIDEIN_ResourceName0 (not GLIDEIN_ResourceName) to resource_name
- [x] #4 A new machine field is captured end-to-end: MachineAttrMachine0 in the job ad -> jobad.py's capture_job_ad_fields() -> the job.assigned NDJSON event -> history_enrich.py's condor_history table (both the condor_history-query path and the jobad-mirror path)
- [x] #5 condor_history table in db.py gains a machine column, with schema-staleness detection so existing provenance.db files pick it up rather than erroring on missing column
- [x] #6 Existing tests using synthetic ClassAd fixtures are updated to use the real attribute names/casing confirmed against a live job, not placeholders
- [x] #7 Regression test covers machine flowing through the jobad-mirror path (job.assigned event -> condor_history row)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. post.py: fix _DEFAULT_FIELD_MAPPING keys (Args, RequestGPUs, MachineAttrGLIDEIN_ResourceName0), add MachineAttrMachine0 -> machine
2. history_enrich.py: mirror same fixes in _FIELD_MAPPING, add machine to _JOBAD_FIELD_MAPPING
3. db.py: add machine column to condor_history schema, extend staleness check
4. Update test fixtures (test_post.py, test_jobad.py, test_history_enrich.py) to real attribute names, add machine coverage
5. Run pytest + ruff + ty
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause, confirmed against a live job (cluster_id 6141971 on dgx-spark2.chtc.wisc.edu, mldag_version 0.1.0rc22): post.py's _DEFAULT_FIELD_MAPPING and history_enrich.py's _FIELD_MAPPING used ClassAd attribute names that don't match what HTCondor actually writes for this repo's submit descriptions, so capture_job_ad_fields() was silently dropping arguments/request_gpus/resource_name on every job while cluster_id/proc_id/request_cpus/request_memory/wall_time_s came through fine.

- Arguments -> Args: every submit description here uses unquoted old-syntax `arguments = ...`, which HTCondor names Args, not Arguments (Arguments is only used for quoted new-syntax arguments).
- RequestGpus -> RequestGPUs: HTCondor's canonical GPU request attribute capitalizes GPUs.
- GLIDEIN_ResourceName isn't a real top-level job-ad attribute at all. Pulling the full live ad surfaced two related attributes instead: MachineAttrGLIDEIN_ResourceName0 (HTCondor's job_machine_attrs echo of the matched slot's GLIDEIN_ResourceName -- literal ClassAd `undefined` for CHTC-direct/non-glidein resources) and JOBGLIDEIN_ResourceName (always populated: the real site name for glidein matches, "Local Job" for CHTC-direct resources). User chose to capture both: resource_name <- JOBGLIDEIN_ResourceName (always meaningful), new glidein_resource_name column <- MachineAttrGLIDEIN_ResourceName0 (null on CHTC-direct jobs, for callers that want to distinguish an actual glidein match).
- New machine field (MachineAttrMachine0) added end-to-end per user request, as an HTCondor-verified cross-check against pretrain_local.sh's self-reported `hostname -f`.
- parse_classad() previously had no handling of ClassAd's bare `undefined` literal -- it fell through to storing the Python string "undefined" rather than treating the attribute as absent. Fixed to skip such lines entirely, matching `ad_key in ad` checks used throughout. This was silently exercisable by more than just the resource_name investigation (e.g. JobCurrentReconnectAttempt = undefined also appears in real ads).

Also fixed the same GLIDEIN_ResourceName issue in mldag.provenance.event_log_scan._resolve_run_id and mldag.provenance.log_monitor's .ad-file resolution tier, which reuse post.py's resource_fields_from_classad/_DEFAULT_FIELD_MAPPING (both are documented as currently-dead in production per task-25, since nothing writes a `<cluster_id>.ad` file today, but still covered by tests).

db.py: condor_history gained `machine` and `glidein_resource_name` columns; _drop_stale_condor_history's staleness check extended to detect either missing column and drop+recreate (existing pattern for this table -- no incremental state to preserve, fully re-derivable via db enrich-history/enrich-jobad/scan --db).

Modified: post.py (_DEFAULT_FIELD_MAPPING, parse_classad undefined handling), history_enrich.py (_FIELD_MAPPING, _JOBAD_FIELD_MAPPING), db.py (condor_history schema + staleness check). Tests updated in test_post.py, test_jobad.py, test_history_enrich.py, test_log_monitor.py, test_event_log_scan.py to use real attribute names/casing confirmed against the live job, plus new coverage for machine, glidein_resource_name, and the undefined-literal parsing fix.

Full suite: 280 passed, 2 skipped (htcondor2 unavailable on macOS). ruff clean on all touched files. ty unavailable in this environment (not installed).

Not yet re-verified against a live job (same limitation noted on task-25/26) -- recommend checking a fresh run's job.assigned event and provenance.db condor_history row for arguments/request_gpus/resource_name/glidein_resource_name/machine once this ships.
<!-- SECTION:NOTES:END -->
