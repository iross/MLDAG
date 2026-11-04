---
id: TASK-31
title: Split mldag into provenance and DAG-generation-and-monitoring packages
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-24 17:39'
updated_date: '2026-11-04 19:58'
labels: []
dependencies: []
references:
  - mldag/daggen.py
  - mldag/constants.py
  - pyproject.toml
  - pretrain.sh
  - pretrain_local.sh
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The mldag repo currently bundles two nearly-independent subsystems: mldag/provenance/* (job/artifact provenance capture and querying) and everything else -- daggen.py, dashboard.py, monitor/, report/, models/, annex/ (DAG generation and monitoring, mldag's namesake). Provenance-side code only depends on pyyaml, htcondor2 and typer; everything heavier (pandas, numpy, matplotlib, seaborn, plotly, pydantic) belongs to the other side. Splitting into two independently versioned/installed packages -- provenance as a new, separately-named, lightweight package; mldag slimmed to just DAG generation and monitoring, depending on the new provenance package -- lets provenance be used on its own (as already established for ad hoc event-log scanning) while mldag keeps instrumenting jobs the same way it does today.

Coupling re-verified 2026-09-22 (the original 2026-08-24 analysis is otherwise stale, predating task-38):
- Exactly one Python import crosses the boundary in either direction: post.py's `from mldag.constants import DEFAULT_CLASSAD_FIELDS_FILE`. daggen.py never imports provenance.
- BUT mldag/constants.py is genuinely shared, not provenance-only: daggen.py:14 and tests/test_daggen.py:5 import the same constant. It moves to the provenance package and both mldag-side importers get rewritten to pull it from there (mldag depends on provenance, so that direction is fine).
- daggen.py bakes three module paths as strings: `-m mldag.provenance.pre` and `-m mldag.provenance.post` (daggen.py:130-131, via `python = sys.executable`) and `-m mldag.provenance.log_monitor` in the SERVICE submit description (daggen.py:141, via bare `python3` off PATH).
- New since the original analysis (task-38): pyproject.toml's `[tool.hatch.build.targets.wheel.hooks.autorun]` hardcodes `code = "import mldag.provenance._autorun"`; pretrain.sh / pretrain_local.sh / pretrain_local_remote.sh each import `mldag.provenance.autocapture` in their heredocs and pip-install the provenance code via `git+https://github.com/iross/MLDAG@v${MLDAG_VERSION}`; autocapture.py, _autorun.py and tests/provenance/test_autocapture.py, test_autorun.py are new files to move.
- The job-side install path is the biggest practical consequence and was not covered by the original plan: daggen.py bakes `MLDAG_VERSION` into every job's environment (daggen.py:165, 193) purely so the job can pip-install the provenance code at a matching tag (pretrain_local.sh:19) and stamp a version into the provenance record (autocapture.py:134). Post-split the job has no use for mldag at all, so `MLDAG_VERSION` is *replaced* by an htprovenance version in the job environment, not supplemented by one.
- Knock-on data-schema decision: autocapture.py records that version under the payload key `mldag_version`, which flows into site_info.json, the job.assigned event, checkpoint sidecars (via watcher.py's env_info passthrough) and db.py's `checkpoints.mldag_version` column. Post-split that key holds an htprovenance version, so the name becomes actively wrong and should be renamed to `htprovenance_version`. Consequence to accept: historical sidecars/events/rows keep the old key, so anything querying `mldag_version` across the split sees both spellings.

Decisions settled 2026-09-22:
- Package name: `htprovenance` -- same string as both distribution name and import name (`pip install htprovenance` / `import htprovenance`), confirmed available on PyPI. Baked strings read as `python -m htprovenance.post`.
- Separate repo (not a monorepo), keeping provenance independently usable and independently versioned.
- Provenance CLI entry points are renamed to match the new package (`htprovenance-pre` etc.), not kept under the `mldag-` prefix and not aliased -- per this repo's "replace, don't deprecate" rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A new repo exists containing the provenance package as top-level `htprovenance/` (all of mldag/provenance/*.py moved 1:1, including autocapture.py and _autorun.py), with tests/provenance/* moved to its tests/, and its own pyproject.toml declaring only pyyaml, htcondor (linux-only) and typer
- [x] #2 The hatch-autorun build hook moves to htprovenance's pyproject.toml with `code = "import htprovenance._autorun"`, and is removed from mldag's; the installed wheel still drops a working .pth file
- [x] #3 DEFAULT_CLASSAD_FIELDS_FILE lives in htprovenance/constants.py; mldag/constants.py is deleted; daggen.py and tests/test_daggen.py import it from htprovenance
- [x] #4 daggen.py's three string-baked module paths (SCRIPT PRE, SCRIPT POST, and the SERVICE arguments line) reference htprovenance.* instead of mldag.provenance.*
- [x] #5 daggen.py bakes HTPROVENANCE_VERSION into each job's environment in place of MLDAG_VERSION (jobs no longer reference mldag at all), and pretrain.sh / pretrain_local.sh / pretrain_local_remote.sh install htprovenance at that version and import htprovenance.autocapture
- [x] #6 The version stamped into provenance records is renamed from mldag_version to htprovenance_version across autocapture.py's payload, db.py's checkpoints column, and the test fixtures asserting on it
- [x] #7 iross/MLDAG is slimmed to daggen.py, dashboard.py, monitor/, report/, models/, annex/, keeping the mldag package name; its pyproject.toml depends on htprovenance, drops the five provenance entry points, gains the currently-undeclared rich/htcondor_cli/classad2 dependencies actually required by annex/monitor, and drops the unused polars dependency
- [x] #8 Provenance entry points are renamed to htprovenance-pre/-post/-log-monitor/-query/-repair; both packages install cleanly and independently in a fresh venv (mldag pulling htprovenance transitively) and every entry point on both sides resolves and shows --help
- [ ] #9 A DAG generated by mldag-gen produces a working SCRIPT PRE/POST/SERVICE trio end-to-end against the new package, confirming the string-based coupling still resolves across the package boundary
- [x] #10 The zero-config .pth auto-capture path still works when installed from the htprovenance wheel (re-run task-38's verification: fake $_CONDOR_JOB_AD in a clean venv, confirm site_info.json + job.assigned appear, and that a non-HTCondor python invocation is still a silent no-op)
- [x] #11 Each repo's README documents the dependency relationship and install instructions for the other, including the in-flight-DAG upgrade hazard below
- [x] #12 mldag/query.py is explicitly left in place on the mldag side, untouched
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Phase 1 -- stand up htprovenance (new repo, nothing removed from MLDAG yet)
1. Create the repo with top-level `htprovenance/`; `git mv`-equivalent all of mldag/provenance/*.py into it and rewrite every intra-package import (`mldag.provenance.X` -> `htprovenance.X`) -- there are ~20 such imports plus docstring/CLI-help references (see the coupling notes above; `rg -n "mldag\.provenance"` finds them all).
2. Add htprovenance/constants.py holding DEFAULT_CLASSAD_FIELDS_FILE; point post.py at it.
3. Move tests/provenance/* into the new repo's tests/ and rewrite their imports the same way. These are the only tests that exist for this code and they must stay green.
4. Write pyproject.toml: hatchling build, deps pyyaml + typer + htcondor (linux-only marker, as today), entry points htprovenance-pre/-post/-log-monitor/-query/-repair, and the hatch-autorun hook with `code = "import htprovenance._autorun"`.
5. Copy .github/workflows/ci.yml and release.yml across, adapting paths (release.yml triggers on `v*` tags and publishes rc tags to TestPyPI -- same flow this repo already uses).
6. Verify before touching MLDAG at all: full test suite green, `uv build` produces a wheel containing `htprovenance/` plus a root-level .pth, and the task-38 clean-venv auto-capture check passes against that wheel.

Phase 2 -- slim MLDAG
7. Delete mldag/provenance/ and tests/provenance/; delete mldag/constants.py.
8. Rewrite daggen.py: the constants import (line 14) to come from htprovenance, and the three baked module paths (lines 130, 131, 141).
9. Replace the version plumbed into jobs: daggen.py's `_mldag_version()` helper and the `mldag_version` parameter threaded through get_submit_description / get_ospool_submit_description (daggen.py:30, 153, 180, 215, 253, 258) become an htprovenance lookup (`importlib.metadata.version("htprovenance")`, same fallback), and the env_vars lists (daggen.py:165, 193) emit `HTPROVENANCE_VERSION=...` instead of `MLDAG_VERSION=...`. Jobs have no remaining use for mldag's own version, so it is dropped rather than kept alongside.
9a. Rename the recorded field to match: autocapture.py's `_mldag_version()` reads `HTPROVENANCE_VERSION` (falling back to `importlib.metadata.version("htprovenance")`) and emits it under the `htprovenance_version` key; rename db.py's `checkpoints.mldag_version` column and update the fixtures in tests/provenance/test_db.py and test_autocapture.py that assert on the old name. Historical sidecars/events/rows keep the old spelling -- accepted, not migrated.
10. Update tests/test_daggen.py's constants import and any assertions that match on the baked `mldag.provenance` strings.
11. pyproject.toml: add htprovenance dependency, remove the five provenance entry points and the autorun build-hook block, add rich/htcondor_cli/classad2, drop polars.

Phase 3 -- the job-side scripts in this repo
12. pretrain.sh, pretrain_local.sh, pretrain_local_remote.sh: change the heredoc import to `htprovenance.autocapture` (and the AttributeError fallback's `default_gpu_info` import), the `python3 -m mldag.provenance.watcher` invocation to `htprovenance.watcher`, and the pip install line from `git+...MLDAG@v${MLDAG_VERSION}` to installing htprovenance at `${HTPROVENANCE_VERSION}` (including the `:?` guard, so a job whose environment predates this change fails loudly at the install step rather than silently running without provenance). `MLDAG_VERSION` should have no remaining readers in any of the three scripts afterwards -- grep to confirm.

Phase 4 -- verification
13. Fresh venv: `pip install mldag` pulls htprovenance transitively; every entry point on both sides resolves and `--help`s.
14. Both repos' full test suites green.
15. Generate a DAG with mldag-gen and confirm the SCRIPT PRE/POST/SERVICE lines name htprovenance modules and actually execute (this is the real test of the string-based coupling -- it is not type-checked or import-checked anywhere).
16. One real end-to-end job through the pretrain_local.sh path, since that is the only thing that exercises the job-side pip install + auto-capture + watcher chain together.

Phase 5 -- docs
17. READMEs on both sides: dependency relationship, install instructions, and the migration hazard below.

Migration hazard to document and respect during rollout:
SCRIPT PRE/POST run AP-side using `sys.executable` as captured at DAG-generation time, and the SERVICE node runs bare `python3` off PATH -- neither is version-pinned. So any DAG generated before the split keeps saying `-m mldag.provenance.post`, and those nodes break the moment that AP environment's mldag is upgraded past the split. Drain in-flight pre-split DAGs before upgrading mldag on the AP (or install the new mldag into a separate venv). Generated .dag files in the working tree are artifacts -- regenerate them, do not hand-edit. Note this is specifically an AP-side hazard: the job-side install is version-pinned via the baked MLDAG_VERSION tag, so already-running jobs keep resolving the old vendored provenance code from the old tag.

Out of scope / explicitly left alone:
- mldag/query.py (top-level, distinct from the provenance query CLI) stays on the mldag side untouched. The original plan called it dead; it is not -- analysis/compare_transfer_attempts.py:36 imports from it, and its fate is tied to the deferred ES-enrichment design question, not to this split.
- Experiment-repo-owned files outside this repo (e.g. the single_protein_models_analysis copy of pretrain.sh) need the same import/install updates as Phase 3 but are tracked separately.

Related: task-17 (original package extraction into mldag) established the CLI-entry-point verification pattern AC #7 mirrors. dashboard.py/monitor/*/report/*/models/*/annex/* have zero test coverage -- a real risk when moving/repackaging that code, flagged here rather than silently absorbed. task-42 (chirp-based provenance delivery) is blocked on this task and will add htchirp to htprovenance's dependencies once it lands.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 1 (htprovenance repo, v0.1.0rc1 tag, TestPyPI) was already done; 301 tests pass there.

MLDAG (branch feat/use-htprovenance, mldag 0.1.0rc28):
- Deleted mldag/provenance/, tests/provenance/, mldag/constants.py.
- daggen.py: htprovenance.constants import; baked -m htprovenance.pre/post/log_monitor; HTPROVENANCE_VERSION (importlib.metadata) replaces MLDAG_VERSION.
- pretrain.sh / pretrain_local.sh: pip-install git+https://github.com/iross/htprovenance@v${HTPROVENANCE_VERSION} --no-deps into .htprovenance and import htprovenance.*. pretrain.sh previously relied on the container image having mldag; it now installs like pretrain_local.sh.
- pyproject: provenance entry points and autorun hook removed; rich declared. Deviation from AC #7: htcondor_cli and classad2 ship inside the htcondor wheel (verified against htcondor 25.14.1), which is already a dependency, so they are not declared separately. polars moved to the dev group rather than dropped, since analysis/*.py still use it.
- README: provenance docs moved to htprovenance README (branch docs/field-mapping-and-scan there); mldag README gains a Provenance section with the in-flight upgrade hazard.
- Tests: test_baked_module_paths_resolve (mutation-checked: reverting one path fails it) and test_submit_description_bakes_htprovenance_version.

Verification (Linux container, python 3.13): all 27 mldag tests pass incl. daggen; all 10 entry points --help on both sides; mldag-gen on the current Experiment.yaml produced htprovenance PRE/POST/SERVICE lines and HTPROVENANCE_VERSION=0.1.0rc1; running those PRE/POST commands emitted job.submitted/job.completed; SERVICE stayed running; .pth auto-capture from the built wheel wrote site_info.json + job.assigned under a fake _CONDOR_JOB_AD and was a no-op otherwise.

Still open (AC #9): run a real DAG under DAGMan and one real pretrain_local.sh job (plan step 16).

Follow-ups noticed, not fixed here: the .pth hook imports autocapture -> jobad -> post at every interpreter start, so python -m htprovenance.post prints a runpy RuntimeWarning; numpy timedelta DeprecationWarning in mldag/report/experiment.py:132 (16 test warnings); 12 pre-existing ty errors in daggen.py.
<!-- SECTION:NOTES:END -->
