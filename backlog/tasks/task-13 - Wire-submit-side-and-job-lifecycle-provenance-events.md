---
id: task-13
title: Wire submit-side and job lifecycle provenance events
status: To Do
assignee: []
created_date: '2026-04-27 20:28'
updated_date: '2026-04-28 20:45'
labels: []
dependencies:
  - task-10
  - task-12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect the event emitter (task-12) to the HTCondor job lifecycle at the two points the AP can observe: DAG generation time and job exit. daggen.py emits job.submitted when it generates the DAG. A SCRIPT POST emits job.completed or job.failed after the job exits, and captures runtime and resource usage from the HTCondor job ClassAd. The job.assigned event is handled on the execute node by task-10 and does not belong here. To make resource usage available to the POST script, daggen.py must include job_ad_file in each submit description so HTCondor writes the final ClassAd to disk when the job exits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] daggen.py emits a job.submitted event for each run at DAG generation time
- [ ] daggen.py includes `job_ad_file = <path>/<cluster>.ad` in each submit description so the final job ClassAd is written on exit
- [ ] provenance_post.sh reads the job ClassAd to extract: RemoteWallClockTime, CPUsUsage, MemoryUsage (peak MB), GPUsUsage
- [ ] provenance_post.sh emits job.completed (with final checkpoint hash and resource usage) or job.failed (with error class and partial runtime) based on job exit code
- [ ] POST script is wired as SCRIPT POST in daggen.py output
- [ ] All emission points write to the same per-run NDJSON file
- [ ] Resource usage fields in job.completed match the envelope schema: wall_time_s, cpu_usage, peak_memory_mb, gpu_usage
- [ ] Events are verifiable by replaying a real or synthetic DAG run
<!-- AC:END -->
