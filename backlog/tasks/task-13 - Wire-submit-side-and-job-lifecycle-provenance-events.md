---
id: task-13
title: Wire submit-side and job lifecycle provenance events
status: Done
assignee: []
created_date: '2026-04-27 20:28'
updated_date: '2026-04-29 00:40'
labels: []
dependencies:
  - task-10
  - task-12
parent_task_id: task-16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect the event emitter (task-12) to the HTCondor job lifecycle at the two points the AP can observe: DAG generation time and job exit. daggen.py emits job.submitted when it generates the DAG. A SCRIPT POST emits job.completed or job.failed after the job exits, and captures runtime and resource usage from the HTCondor job ClassAd. The job.assigned event is handled on the execute node by task-10 and does not belong here. To make resource usage available to the POST script, daggen.py must include job_ad_file in each submit description so HTCondor writes the final ClassAd to disk when the job exits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 daggen.py emits a job.submitted event for each run at DAG generation time
- [x] #2 daggen.py includes `job_ad_file = <path>/<cluster>.ad` in each submit description so the final job ClassAd is written on exit
- [x] #3 provenance_post.sh reads the job ClassAd to extract: RemoteWallClockTime, CPUsUsage, MemoryUsage (peak MB), GPUsUsage
- [x] #4 provenance_post.sh emits job.completed (with final checkpoint hash and resource usage) or job.failed (with error class and partial runtime) based on job exit code
- [x] #5 POST script is wired as SCRIPT POST in daggen.py output
- [x] #6 All emission points write to the same per-run NDJSON file
- [x] #7 Resource usage fields in job.completed match the envelope schema: wall_time_s, cpu_usage, peak_memory_mb, gpu_usage
- [x] #8 Events are verifiable by replaying a real or synthetic DAG run
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created mldag/provenance/post.py with parse_classad(), run_id_from_classad(), resource_fields_from_classad(), and emit_post_event(). The module extracts run_id from the PROVENANCE_RUN_ID env var in the HTCondor ClassAd Environment attribute, maps RemoteWallClockTime/CPUsUsage/MemoryUsage/GPUsUsage to the provenance schema fields, and emits job.completed or job.failed. provenance_post.sh is a thin bash wrapper (set -euo pipefail, strips proc from  to get ClusterId) that calls python3 -m mldag.provenance.post. daggen.py updated to: add job_ad_file = output/provenance/$(ClusterId).ad to both submit description functions; wire SCRIPT POST in get_script(); emit job.submitted for each training job after DAG file is written. Added mldag-post console script entry point. 16 new tests all pass; full suite 42/42.
<!-- SECTION:NOTES:END -->
