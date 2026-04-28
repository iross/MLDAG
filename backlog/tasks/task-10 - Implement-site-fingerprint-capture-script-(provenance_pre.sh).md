---
id: task-10
title: Implement site fingerprint capture script (provenance_pre.sh)
status: To Do
assignee: []
created_date: '2026-04-27 20:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A shell script run as a DAGMan SCRIPT PRE that captures the execution environment at job start — hostname, GPU model, CUDA version, Python version, and code commit — and writes a structured site_info JSON file. This gives every job a machine-readable record of where and in what environment it ran, without requiring any changes to the training code itself.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Script captures: hostname, GPU model (from nvidia-smi), CUDA version, Python version, git commit of training code,Output is written as site_info.json in a location readable by the job (e.g. alongside the job's output directory),Script is wired as SCRIPT PRE in the relevant .sub file or daggen.py output,Script exits non-zero on failure so DAGMan can retry or abort,site_info.json schema matches the produced_at and environment fields in the sidecar design
<!-- AC:END -->
