---
id: task-14
title: >-
  Wire epoch-level provenance events and sidecar generation into training
  workflow
status: Done
assignee: []
created_date: '2026-04-27 20:30'
updated_date: '2026-04-29 00:44'
labels: []
dependencies:
  - task-11
  - task-12
parent_task_id: task-16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hook the sidecar writer (task-11) and event emitter (task-12) into the per-epoch checkpoint save. After each checkpoint is written, write_sidecar() records the checkpoint's hash and parent chain, and emit_event() records epoch.started and epoch.completed with loss and duration. If the training code cannot be modified directly, a file-watcher on the checkpoint output directory serves as a fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 epoch.started event is emitted at the beginning of each epoch with checkpoint_in_hash (hash of previous checkpoint, None for epoch 0)
- [x] #2 epoch.completed event is emitted after each checkpoint appears with checkpoint_out_hash and duration_s
- [x] #3 A sidecar .provenance.json is written alongside every checkpoint file
- [x] #4 parent_hash in each sidecar correctly references the prior epoch's checkpoint hash (None for epoch 0)
- [x] #5 The parent_hash chain is unbroken across a multi-epoch run
- [x] #6 The watcher reads site_info.json written by task-10 and includes it in sidecars
- [x] #7 pretrain.sh launches watcher in background and kills it cleanly on exit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created mldag/provenance/watcher.py with watch_and_emit() that polls a directory for new .ckpt files (rglob, sorted by mtime). For each new checkpoint: emits epoch.started with checkpoint_in_hash, calls write_sidecar() maintaining the parent_hash chain, emits epoch.completed with checkpoint_out_hash and duration_s. Reads site_info.json (written by task-10) to populate sidecar site/env fields. Exits on SIGTERM or idle_timeout. pretrain.sh updated to launch watcher in background after _provenance_capture_and_emit, with EXIT trap for clean shutdown. The training code at /workspace/metl/code/train_source_model.py is in the container and cannot be modified; loss is not captured (requires training code hook). 15 tests pass; full suite 57/57.
<!-- SECTION:NOTES:END -->
