---
id: task-10
title: Capture site fingerprint from within training job
status: To Do
assignee: []
created_date: '2026-04-27 20:24'
updated_date: '2026-04-28 20:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The execution environment (hostname, GPU model, CUDA version, Python version, code commit) must be captured on the execute node where the job actually runs. This is done by a shell function or sourced script invoked at the start of pretrain.sh, before training begins. It writes a structured site_info.json to the job working directory, which is then available to the sidecar writer and event emitter. A DAGMan SCRIPT PRE cannot serve this purpose — it runs on the Access Point and has no visibility into the execute node.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] site_info.json is written at job start on the execute node before any training begins
- [ ] Captured fields: hostname, GPU model and count (from nvidia-smi), CUDA version, Python version, git commit of training code
- [ ] site_info.json schema matches the produced_at and environment fields in the sidecar design (backlog/docs/provenance_design.md)
- [ ] Capture code exits non-zero and aborts the job if nvidia-smi or other required tools are unavailable
- [ ] job.assigned event is emitted from the execute node using the event emitter (task-12) with site and slot info from site_info.json
- [ ] Capture is triggered from pretrain.sh — no DAGMan PRE script wiring required
<!-- AC:END -->
