---
id: task-9
title: Export PROVENANCE_RUN_ID into HTCondor job environment
status: Done
assignee: []
created_date: '2026-04-27 20:18'
updated_date: '2026-04-29 00:28'
labels: []
dependencies: []
parent_task_id: task-16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The existing run_uuid DAG variable is already stable across all epochs of a run and used for directory naming, but it is passed only as a CLI argument to pretrain.sh. Lifecycle scripts (pre/post) and the training process need to read it without depending on argument parsing. Exporting it as an environment variable is the lowest-friction provenance step and unlocks every subsequent piece of provenance infrastructure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 run_uuid is exported as PROVENANCE_RUN_ID in the HTCondor job environment (via environment = in the .sub file or daggen.py),All epochs of a run share the same PROVENANCE_RUN_ID value,pretrain.sh and any pre/post scripts can read PROVENANCE_RUN_ID without argument changes,daggen.py generates the correct environment stanza for new experiments
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added PROVENANCE_RUN_ID=$(run_uuid) to the HTCondor environment stanza in both get_submit_description() and get_ospool_submit_description() in mldag/daggen.py. The variable is set in VARS per job node so it is consistent across all epochs of a run. Coexists with WANDB_API_KEY when present.
<!-- SECTION:NOTES:END -->
