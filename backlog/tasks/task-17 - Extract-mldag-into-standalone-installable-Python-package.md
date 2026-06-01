---
id: TASK-17
title: Extract mldag into standalone installable Python package
status: In Progress
assignee:
  - claude
created_date: '2026-05-04 18:54'
updated_date: '2026-05-18 15:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The mldag framework (DAG generation, provenance tracking, log monitoring, reporting) is reusable across experiments, but is currently tangled with project-specific files. Splitting into a standalone package lets experiment repos stay minimal — Experiment.yaml, resources.yaml, the training script, and run artifacts — with mldag installed as a versioned dependency. This enables clean reproducibility: pin the mldag version used for a run and the experiment config is fully self-describing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `provenance_pre.sh` and `provenance_post.sh` removed — DAG calls `python -m mldag.provenance.pre/post` directly
- [x] `hourly_dashboard.py` promoted to `mldag-dashboard` CLI entry point registered in pyproject.toml
- [x] README documents how to bootstrap a new experiment repo against the package
- [ ] mldag package is installable via pip/uv from PyPI or git URL
- [ ] All CLI entry points (`mldag-gen`, `mldag-csv`, `mldag-report`, `mldag-dashboard`, provenance query) work after install
- [ ] A minimal experiment repo bootstraps correctly with only Experiment.yaml, resources.yaml, config.yaml, and run artifacts — no framework code
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Files that belong in the mldag package repo

**Python package (mldag/):** all of mldag/ as-is — daggen.py, models/, monitor/, provenance/, report/, annex/

**Tests:** tests/ directory as-is

**Package config:** pyproject.toml (already defines entry points), uv.lock

**Loose scripts → entry points:** hourly_dashboard.py should become mldag.dashboard CLI entry point in pyproject.toml, not a loose file

**Remove from experiment repos:** provenance_pre.sh and provenance_post.sh — the DAG already calls `python -m mldag.provenance.pre/post` directly; these shell wrappers are vestigial and should be deleted

**Justfile:** the common recipe skeleton (install, generate-csv, generate-report, daily-summary, recent-summary, monthly-report, hourly-site) belongs in the package repo as documentation/template; the experiment repo overrides the experiment-specific variables

---

## Files that belong in each experiment repo

**Required config:**
- Experiment.yaml — submit template, hyperparams, epoch/run counts
- resources.yaml — which compute sites to target (CHTC, OSPool, Annex names)
- config.yaml — runtime secrets and settings (W&B API key, etc.)
- .env — environment secrets (gitignored)

**Training executable (per experiment):**
- pretrain_local.sh (or pretrain.sh for OSPool) — the script that runs inside the container on the execute node. Currently mixes framework code (provenance capture, watcher call) with experiment-specific training invocation. After the split, this script should call package entry points to bracket the training command:
  1. `python -m mldag.provenance.capture_site` (replaces the inline _provenance_capture_and_emit function — emits job.assigned, writes cluster .run_id marker)
  2. Experiment-specific training command (e.g. python /workspace/metl/code/train_source_model.py ...)
  3. `python -m mldag.provenance.watcher . $PROVENANCE_RUN_ID --one-shot --start-time $_TRAIN_START`
  This means mldag must be installed in the container image.

**Annex resource requests (per experiment):**
- *.request files (anvil_annex.request, bridges2_annex.request, delta_annex.request, expanse_annex.request) — site-specific GPU allocation configs; vary per experiment

**Justfile (experiment-specific portions):**
- _refresh-* recipes: AP hostnames and remote paths are experiment-specific (e.g. ap40:/home/ian.ross/MLDAG_AWS/, iaross@ap2002.chtc.wisc.edu:/home/iaross/...)
- _csv-* recipes: the --dag-files and --metl-logs flags list specific filenames for this experiment

**Run artifacts (generated, not committed):**
- $run_uuid/ directories (stdout/stderr)
- output/ (training logs, provenance NDJSON, ClassAd files, checkpoints)
- metl.log, *.dag, *.dag.*, *.csv

---

## Design decisions to resolve before starting

1. **Container install:** mldag must be pip-installable and present in the training container for pretrain.sh to call its entry points. Options: bake it into the container image at build time, or pip-install at job start from a pinned git URL. Baking is faster; git URL is more flexible.

2. **mldag.provenance.capture_site entry point:** The _provenance_capture_and_emit inline Python block in pretrain_local.sh needs to become a proper module entry point (emits job.assigned, writes .run_id marker). This is a new addition to the package.

3. **output/ directory convention:** The package assumes output/provenance/ for NDJSON and ClassAds, and output/training_logs/$run_uuid/ for checkpoints. This path contract needs to be documented (or made configurable) before the split, since experiment repos will depend on it.

4. **justfile template vs experiment override:** Decide whether the package ships a justfile template that experiment repos copy-and-customize, or whether the experiment justfile simply calls uv run mldag-* commands directly (simpler, no template needed).

5. **site/ submodule (gh-pages dashboard):** Currently a submodule pointed at gh-pages branch of this repo. After the split, each experiment repo would push to its own gh-pages — the hourly-site recipe in the justfile already handles this.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed three of the key AC items: (1) deleted provenance_pre.sh and provenance_post.sh — vestigial wrappers since daggen.py already calls python -m mldag.provenance.pre/post directly; (2) moved hourly_dashboard.py into mldag/dashboard.py and registered mldag-dashboard entry point in pyproject.toml; (3) rewrote README with a bootstrap section documenting install, entry points table, and minimal justfile template. Remaining: container image install verification and PyPI publish.
<!-- SECTION:NOTES:END -->
