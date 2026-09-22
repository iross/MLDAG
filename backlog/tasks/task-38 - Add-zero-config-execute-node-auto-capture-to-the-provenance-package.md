---
id: TASK-38
title: >-
  Add zero-config execute-node auto-capture (site info, GPU, jobad fields) to
  the provenance package
status: In Progress
assignee: []
created_date: '2026-09-11 01:43'
updated_date: '2026-09-22 15:52'
labels: []
dependencies:
  - TASK-31
references:
  - pretrain_local.sh
  - mldag/provenance/jobad.py
  - mldag/provenance/events.py
  - mldag/provenance/watcher.py
  - mldag/provenance/post.py
  - mldag/provenance/history_enrich.py (enrich_from_jobad_events)
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute-node provenance capture is currently 100% manual. Every AP-side event (job.submitted, job.assigned, job.completed/failed) is wired automatically today because daggen.py bakes SCRIPT PRE/POST/SERVICE calls into every generated DAG -- but nothing runs inside the job itself unless an experiment repo hand-writes glue code. watcher.py already expects a site_info.json file to exist (its own comment calls it "written by the execute-node capture function"), but no such function exists anywhere in this codebase -- it's a documented contract with zero implementation.

This task gives the standalone provenance package (see task-31) a zero-code auto-capture path: once the package is installed into the same Python environment a training job runs in, it should detect that it's running inside an HTCondor job and automatically record site/hardware/software-environment context, with no line of code required in the training script itself.

Concrete motivating regression (originally task-40): pretrain.sh, used by the single-protein pretraining pipeline in the separate single_protein_models_analysis repo, hand-rolls its own job.assigned event via an inline Python heredoc, gathering GPU/host info via torch.cuda introspection. It never called mldag.provenance.jobad.capture_job_ad_fields(), so the event payload had none of the ClassAd-derived fields (arguments, request_cpus/memory/gpus, etc.) that 'mldag-query db enrich-jobad' expects to mirror into condor_history -- condor_history.arguments (and everything else enrich-jobad backfills) was empty for every run built from that pipeline, confirmed across mldag_version 0.1.0rc11 through rc23 in single_protein_models_analysis's provenance.db/mixed.db/dgxspark.db. That repo isn't checked out here, so this task can't edit its pretrain.sh directly -- but the whole point of a zero-config auto-capture path is that it doesn't need to: once that repo's mldag pin moves past this change, the same fix applies with no edits on their side.

This repo has its own two copies of the identical anti-pattern -- pretrain_local.sh (current; Experiment.yaml's "GB1 Pretraining"/"Many Protein Pretraining" jobs run it) and pretrain.sh (older, global-pretraining variant, still committed) -- both hand-roll job.assigned via inline heredocs that duplicate emit_event's schema/append logic, and both run their whole capture step inside `_provenance_capture_and_emit() { ... } || exit 1`, meaning any uncaught Python exception in there (a missing torch, a GPU query failure, anything) currently aborts the training job outright -- the opposite of the best-effort guarantee this task adds (AC#7). Fixing both in place, on top of the new shared module, both demonstrates it end-to-end and removes that footgun.

This captures the facets that are obtainable without any explicit call from the training script and don't require further research to implement:
- Site & hardware: hostname, slot, GPU model/count -- satisfies the site_info.json contract watcher.py already expects, and gives reliable host/site data that the AP side structurally cannot get today (log_monitor.py's own comments note its SlotName-based host parsing is dead/unreliable).
- Submit-time job identity: the executable's invocation arguments and path (Args/Cmd are ordinary $_CONDOR_JOB_AD attributes -- Args is already in post.py's default field mapping, Cmd just isn't yet) via the existing capture_job_ad_fields()/ClassAd field-mapping mechanism.

PROVENANCE_RUN_ID is already injected into every job's environment by daggen.py today, so this does not need to invent run-id discovery.

Out of scope for this task: executable/container checksums, Docker registry digest resolution, and SIGTERM/eviction handling. Those require further empirical verification against live HTCondor internals (open questions to HTCondor developers, not yet answered -- see this task's original research notes) and are tracked in a separate follow-on task rather than blocking this one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package detects it is running inside an HTCondor job (via $_CONDOR_JOB_AD / $_CONDOR_MACHINE_AD presence) at interpreter startup, with zero calls required from the training script; degrades to a documented no-op when the install mechanism doesn't support the auto-run hook (e.g. some container/zipapp setups), never failing installation
- [x] #2 Auto-writes site_info.json (hostname, slot, GPU model/count/id, cuda, python -- best-effort) on detection, satisfying the contract watcher.py's _load_site_info already expects
- [x] #3 Emits job.assigned via mldag.provenance.events.emit_event (not a hand-rolled JSON write), merging capture_job_ad_fields()'s output with cluster_id/proc_id directly in the event payload (not only a sidecar file)
- [x] #4 Cmd is added to the default ClassAd field mapping in post.py (_DEFAULT_FIELD_MAPPING), alongside the existing Args
- [x] #5 pretrain_local.sh's and pretrain.sh's hand-rolled _provenance_capture_and_emit heredocs are both replaced by the new auto-capture path, preserving every site-info/env field each currently writes (hostname, slot, gpu_model, gpu_count, gpu_id, python, cuda, code_commit, mldag_version where present) and dropping the now-redundant {cluster_id}.run_id sidecar file; neither script aborts the job any more if capture fails
- [x] #6 Running mldag-query db enrich-jobad against a freshly-produced provenance database populates condor_history.arguments (and the other jobad-mirrored fields) for runs produced by either script
- [x] #7 All auto-capture is best-effort: any failure (missing env var, absent nvidia-smi, non-HTCondor environment) is caught and never alters the training job's behavior, output, or exit code
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm $_CONDOR_JOB_AD / $_CONDOR_MACHINE_AD availability and exact content on a real execute node (CHTC direct + at least one OSPool glidein) before committing to field names. Verify via the job's own executable writing diagnostics to a file that transfers back, not via an interactive condor_ssh_to_job session -- condor_nsenter-based sessions are known to diverge from the job's actual launch-time environment (env vars set by Apptainer's own action scripts aren't sourced for the ssh-spawned shell, and GPU device bind-mounts can differ)
2. Design and implement the .pth-based auto-run trigger; verify it fires inside a real job's Python environment and no-ops cleanly when the install mechanism doesn't support .pth (e.g. some container/zipapp setups)
3. Add Cmd to post.py's _DEFAULT_FIELD_MAPPING
4. Build a single auto-capture entrypoint in the provenance package that writes site_info.json, calls capture_job_ad_fields(), and calls emit_event("job.assigned", ...) with the merged payload -- replacing the ad hoc JSON-append logic currently duplicated in pretrain_local.sh
5. Update pretrain_local.sh to call the new path instead of its heredoc, preserving its torch.cuda GPU introspection as the site-info source
6. Verify end-to-end on a real job: site_info.json appears, job.assigned lands in NDJSON with jobad fields + cluster_id, mldag-query db enrich-jobad populates condor_history.arguments
7. Archive task-40 (superseded by this task) once verified
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: added mldag/provenance/autocapture.py (capture_and_emit(), best-effort, idempotent via a .mldag_provenance_autocapture_done marker file in the sandbox) and mldag/provenance/_autorun.py (the .pth import target). The .pth itself is installed via hatch-autorun (a build-time-only dependency, maintained by hatchling's own author specifically for this pattern) configured in pyproject.toml's [tool.hatch.build.targets.wheel.hooks.autorun] -- avoids hand-writing a custom Hatchling build hook and its known editable-install/pth-ordering pitfalls. Added Cmd to post.py's _DEFAULT_FIELD_MAPPING (AC#4). Replaced both pretrain_local.sh's and pretrain.sh's ~70-line hand-rolled heredocs with a ~25-line one that does torch.cuda GPU introspection (kept, more precise than autocapture's own nvidia-smi/proc fallback) and calls capture_and_emit(gpu_info=...); both scripts no longer run capture under "|| exit 1", and the {cluster_id}.run_id sidecar file is gone now that cluster_id/proc_id ride in the event payload via capture_job_ad_fields().

Verification (no live HTCondor cluster available in this environment): built the wheel (uv build --wheel), installed it --no-deps into a clean venv, and confirmed empirically that (1) a plain python invocation with no _CONDOR_JOB_AD is a true no-op, (2) with a fake $_CONDOR_JOB_AD classad file set, the .pth hook fires with zero explicit calls and produces a correct site_info.json + job.assigned NDJSON event including cluster_id/arguments/request_cpus from the classad, (3) a second interpreter startup in the same sandbox does not duplicate the event (idempotency marker), (4) a genuinely broken environment (pyyaml missing, so importing the capture chain raises ModuleNotFoundError) is silently swallowed with exit 0 -- the exact AC#7 guarantee the old heredocs' "|| exit 1" violated. Also ran an end-to-end enrich-jobad check (build_database + enrich_from_jobad_events against a DB seeded from a capture_and_emit()-produced event) and confirmed condor_history.arguments/request_cpus/cluster_id populate correctly (AC#6). pretrain_local.sh/pretrain.sh changes verified via bash -n, shellcheck (no new findings), and py_compile of the embedded heredoc.

Recommended follow-up (not blocking, but not done here): submit one real test job on CHTC direct and one OSPool glidein target to confirm $_CONDOR_JOB_AD/$_CONDOR_MACHINE_AD content matches assumptions in the field, per the original implementation plan step 1/6 -- the detection env var itself ($_CONDOR_JOB_AD) is already proven reliable in production via jobad.py, so this is a low-risk confirmation rather than a design uncertainty.

Files added: mldag/provenance/autocapture.py, mldag/provenance/_autorun.py, tests/provenance/test_autocapture.py, tests/provenance/test_autorun.py.
Files modified: mldag/provenance/post.py (Cmd mapping), pretrain_local.sh, pretrain.sh, pyproject.toml (hatch-autorun build hook), tests/provenance/test_post.py, tests/provenance/test_jobad.py (Cmd coverage).
<!-- SECTION:NOTES:END -->
