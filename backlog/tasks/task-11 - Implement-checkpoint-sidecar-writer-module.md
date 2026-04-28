---
id: task-11
title: Implement checkpoint sidecar writer module
status: To Do
assignee: []
created_date: '2026-04-27 20:25'
labels: []
parent_task_id: task-16
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A self-contained Python module that writes a .provenance.json sidecar file alongside each checkpoint. The sidecar records the checkpoint's SHA-256 hash, the hash of its parent checkpoint, the run_id, epoch number, site/environment info, and training metrics. The parent_hash field creates a linked list across all checkpoints, making the full training lineage reconstructable from the files alone — no database required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 sha256_file(path) computes and returns the hex digest of a checkpoint file,write_sidecar() writes a .provenance.json next to the checkpoint matching the schema in provenance_design.md,parent_hash is None for epoch 0 and the previous checkpoint hash for all subsequent epochs,Schema version field is present in every sidecar,Module has no external dependencies (stdlib only),Unit tests cover: correct hash computation, correct parent chain, schema fields present, epoch-0 None parent
<!-- AC:END -->
