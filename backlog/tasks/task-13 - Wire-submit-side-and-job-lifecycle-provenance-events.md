---
id: task-13
title: Wire submit-side and job lifecycle provenance events
status: To Do
assignee: []
created_date: '2026-04-27 20:28'
labels: []
dependencies:
  - task-10
  - task-12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect the event emitter (task-12) to the HTCondor job lifecycle: daggen.py emits job.submitted at DAG generation time; provenance_pre.sh (task-10) emits job.assigned after capturing site info; a new provenance_post.sh emits job.completed or job.failed after the job exits. This gives an operational record of every job's lifecycle without touching the training code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 daggen.py emits a job.submitted event for each run at DAG generation time,provenance_pre.sh emits job.assigned with site and slot info after capturing the fingerprint,provenance_post.sh is created and emits job.completed (with final checkpoint hash) or job.failed (with error class) based on the job exit code,post script is wired as SCRIPT POST in daggen.py output,All three emission points write to the same per-run NDJSON file,Events are verifiable by replaying a real or synthetic DAG run
<!-- AC:END -->
