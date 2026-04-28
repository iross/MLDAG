---
id: task-14
title: >-
  Wire epoch-level provenance events and sidecar generation into training
  workflow
status: To Do
assignee: []
created_date: '2026-04-27 20:30'
labels: []
parent_task_id: task-16
dependencies:
  - task-11
  - task-12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hook the sidecar writer (task-11) and event emitter (task-12) into the per-epoch checkpoint save. After each checkpoint is written, write_sidecar() records the checkpoint's hash and parent chain, and emit_event() records epoch.started and epoch.completed with loss and duration. If the training code cannot be modified directly, a file-watcher on the checkpoint output directory serves as a fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 epoch.started event is emitted at the beginning of each epoch with checkpoint_in_hash,epoch.completed event is emitted after checkpoint save with checkpoint_out_hash, loss, and duration_s,A sidecar .provenance.json is written alongside every checkpoint file,parent_hash in each sidecar correctly references the prior epoch's checkpoint hash,The chain is unbroken across a multi-epoch run,If the training script cannot be modified, a file-watcher alternative is documented and functional
<!-- AC:END -->
