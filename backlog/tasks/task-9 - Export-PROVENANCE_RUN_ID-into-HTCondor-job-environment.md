---
id: task-9
title: Export PROVENANCE_RUN_ID into HTCondor job environment
status: To Do
assignee: []
created_date: '2026-04-27 20:18'
labels: []
parent_task_id: task-16
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The existing run_uuid DAG variable is already stable across all epochs of a run and used for directory naming, but it is passed only as a CLI argument to pretrain.sh. Lifecycle scripts (pre/post) and the training process need to read it without depending on argument parsing. Exporting it as an environment variable is the lowest-friction provenance step and unlocks every subsequent piece of provenance infrastructure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 run_uuid is exported as PROVENANCE_RUN_ID in the HTCondor job environment (via environment = in the .sub file or daggen.py),All epochs of a run share the same PROVENANCE_RUN_ID value,pretrain.sh and any pre/post scripts can read PROVENANCE_RUN_ID without argument changes,daggen.py generates the correct environment stanza for new experiments
<!-- AC:END -->
