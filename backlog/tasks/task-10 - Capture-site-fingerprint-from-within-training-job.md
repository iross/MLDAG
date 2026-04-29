---
id: task-10
title: Capture site fingerprint from within training job
status: Done
assignee: []
created_date: '2026-04-27 20:24'
updated_date: '2026-04-29 00:35'
labels: []
dependencies: []
parent_task_id: task-16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The execution environment (hostname, GPU model, CUDA version, Python version, code commit) must be captured on the execute node where the job actually runs. This is done by a shell function or sourced script invoked at the start of pretrain.sh, before training begins. It writes a structured site_info.json to the job working directory, which is then available to the sidecar writer and event emitter. A DAGMan SCRIPT PRE cannot serve this purpose — it runs on the Access Point and has no visibility into the execute node.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 site_info.json is written at job start on the execute node before any training begins
- [x] #2 Captured fields: hostname, GPU model and count (from nvidia-smi), CUDA version, Python version, git commit of training code
- [x] #3 site_info.json schema matches the produced_at and environment fields in the sidecar design (backlog/docs/provenance_design.md)
- [x] #4 Capture code exits non-zero and aborts the job if nvidia-smi or other required tools are unavailable
- [x] #5 job.assigned event is emitted from the execute node using the event emitter (task-12) with site and slot info from site_info.json
- [x] #6 Capture is triggered from pretrain.sh — no DAGMan PRE script wiring required
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added _provenance_capture_and_emit() function to pretrain.sh as an inline Python heredoc. The function runs before dataset unpacking, captures hostname/GPU model+count (nvidia-smi), CUDA version, Python version, and git commit, then writes site_info.json to the job working directory and appends a job.assigned NDJSON event to output/provenance/<run_id>.ndjson (or PROVENANCE_LOG_DIR if set). Exits non-zero if nvidia-smi is unavailable, aborting the job. Also removed duplicate shebang and added set -euo pipefail.
<!-- SECTION:NOTES:END -->
