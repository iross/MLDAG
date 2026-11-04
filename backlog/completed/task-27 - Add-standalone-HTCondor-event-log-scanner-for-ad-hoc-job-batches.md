---
id: TASK-27
title: Add standalone HTCondor event-log scanner for ad hoc job batches
status: Done
assignee:
  - claude
created_date: '2026-08-24 14:53'
updated_date: '2026-08-24 14:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batches of jobs run without the full DAGMan PRE/POST provenance instrumentation (no NDJSON job.submitted events, no run_id) still produce a normal HTCondor event log. There's currently no way to get duration/site/resource-usage out of that log without the full pipeline. log_monitor.py already has the parsing logic (event 005 resource-usage banner, timestamps) but it's wired directly into DAGMan-specific run_id resolution and NDJSON emission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A function scans an arbitrary HTCondor event log and returns per-cluster_id duration/site/resource-usage with no NDJSON or run_id dependency
- [x] #2 Execute host/site is parsed from event 001's SlotName line (never captured anywhere today -- the .ad classad path is dead per task-25)
- [x] #3 Scanning opportunistically resolves run_id/job_name when a provenance_log_dir or classad log_dir is available and silently falls back to bare cluster_id keys otherwise
- [x] #4 Reuses log_monitor.py's existing regex/parsing helpers rather than duplicating event-log syntax
- [x] #5 A CLI command (mldag-query scan) exposes this
- [x] #6 Unit tests cover: full single-job lifecycle scan, glidein-style double-@ SlotName parsing, opportunistic enrichment via classad dir, opportunistic enrichment via provenance NDJSON dir, and pure fallback with neither available
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add _SLOTNAME_RE regex and a small site-from-slotname helper
2. Add scan_event_log() to a new mldag/provenance/event_log_scan.py, single-pass over the whole file, reusing log_monitor.py's _ANY_HEADER_RE/_DAGNODE_RE/_accumulate_usage_field/_resolve_run_id/_refresh_job_submitted_index
3. Opportunistic enrichment: job_name from 000/DAGNodeName always (free); run_id from provenance_log_dir (job_name index) and/or log_dir (classad/.run_id marker) only if those dirs are passed and contain matching data
4. Wire a 'scan' command into query.py's CLI
5. Unit tests: single-job lifecycle, glidein double-@ site parsing, both enrichment paths, and neither-available fallback
6. Run full test suite
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added mldag/provenance/event_log_scan.py:scan_event_log(), a single-pass parser reusing log_monitor.py's regexes/helpers (_ANY_HEADER_RE, _DAGNODE_RE, _accumulate_usage_field, _resolve_run_id, _refresh_job_submitted_index) with no NDJSON/run_id dependency. Added _SLOTNAME_RE + site_from_slotname() to log_monitor.py to parse execute host/site from event 001's SlotName body line -- a field nothing in the pipeline captured before (the .ad/GLIDEIN_ResourceName path is dead per task-25). Enrichment is opportunistic: job_name comes free from DAGNodeName in 000 blocks; run_id is resolved via provenance_log_dir (job_name index) and/or log_dir (.run_id marker or .ad classad) only when those directories exist and contain a match -- both default to output/provenance but missing/empty dirs are a silent no-op, never an error. Wired as 'mldag-query scan <log_file>'. Verified against real repo log files (metl_aws.log, metl.log) including the glidein double-@ SlotName format. 14 new tests in tests/provenance/test_event_log_scan.py; full suite (268 passed, 1 skipped) and ruff/ty clean on the new/touched files.
<!-- SECTION:NOTES:END -->
