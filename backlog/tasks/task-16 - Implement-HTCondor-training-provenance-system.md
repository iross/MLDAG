---
id: task-16
title: Implement HTCondor training provenance system
status: To Do
assignee: []
created_date: '2026-04-28 20:49'
labels: []
dependencies:
  - task-9
  - task-10
  - task-11
  - task-12
  - task-13
  - task-14
  - task-15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
End-to-end lineage tracking for multi-site training jobs. Every training run should produce a complete, queryable record of where each epoch ran, what hardware and software environment was used, what checkpoint was produced, and what the resource consumption was. This record should be reconstructable from flat files alone — no shared database required — and should survive job migrations, retries, and partial failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A complete provenance record exists for any training run: run ID, submission metadata, per-epoch site and environment, checkpoint hashes, and resource usage
- [ ] The full training history of any checkpoint is reconstructable by walking the sidecar chain without consulting any external system
- [ ] All lifecycle events (submitted, assigned, epoch started/completed, completed/failed) are queryable by run ID from the NDJSON event log
- [ ] Resource usage (wall time, CPU, GPU, memory) is captured for every completed job via HTCondor ClassAds
- [ ] The system works across job migrations and retries — the run ID threads through all attempts
<!-- AC:END -->
