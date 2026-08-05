---
id: TASK-22
title: 'Investigate and fix persistent unknown:<cluster_id> provenance files'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-05 16:18'
updated_date: '2026-08-05 16:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-17's PR #3/#4 (2026-06-16) fixed the DAGNodeName regex mismatch and added durable .run_id marker files, intended to eliminate unknown:<cluster_id>.ndjson fallback files. Despite that fix being merged and present on main, ~1770 unknown:<cluster_id>.ndjson files exist in provenance/ and output/provenance/ today, with event timestamps as recent as 2026-08-05 -- weeks after the fix landed. This pollution blocks reliable cross-run querying (see task-21) and misattributes real job lifecycle events to a placeholder ID instead of the true run. The June fix addressed a real bug but did not stop the underlying pattern; the actual persisting cause(s) need to be found and closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause(s) for continued unknown:<cluster_id>.ndjson creation after the June 2026 fixes are identified and documented with supporting evidence
- [x] #2 A fix is implemented such that new DAG runs no longer produce persistent unknown:<cluster_id> events under normal operation
- [x] #3 The divergence between the provenance/ and output/provenance/ directories is either reconciled or documented with a clear explanation and remediation
- [x] #4 A repair pass resolves, or clearly reports as unresolvable, the unknown:<cluster_id>.ndjson files already on disk
- [x] #5 Regression tests are added that would have caught the persisting failure mode(s)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Evidence gathered so far

- `provenance/` has 1769 `unknown:*.ndjson` files, `output/provenance/` has 1771 — the two directories are near-identical, and for every filename checked the content is **byte-identical** (same md5, same sub-millisecond timestamps). This looks like one directory was copied/rsynced from the other at some point, not two independently-diverging outputs.
- The `run_id` field *inside* the JSON lines is itself `"unknown:<cluster_id>"` (not just the filename) — e.g. `provenance/unknown:5772633.ndjson` has 5 lines (`transfer.input.started/completed`, `job.executing`, `transfer.output.started/completed`) all carrying `"run_id":"unknown:5772633"`, with `condor_event_ts` of 2026-07-23 — five weeks after the regex/`.run_id`-marker fix (commit `eee7579`, merged 2026-06-16) landed on `main`. So the fix is present and the bug still fires on recent runs.
- Only two of the many `.dag` files in this checkout declare the `SERVICE provenance_monitor` block at all (`gb1_pretraining.dag`, `many_protein_pretraining_with_ospool.dag`); both hardcode `arguments = -m mldag.provenance.log_monitor --log-file metl.log --classad-dir output/provenance` (from `daggen.py:120`, `get_service()`). The other `.dag` files (bigger_global_pretraining, global_pretraining*, ospool_pretraining, trainingrun*, experiment_devices) predate the log_monitor SERVICE and never run it, so all `unknown:*` files with `job.executing`/`transfer.*` event types must come from one of those two DAGs.
- Reviewed `_resolve_run_id` and `monitor_once` (`mldag/provenance/log_monitor.py`) in full. The `pending_lookups` retry mechanism (added in `8a12e85`, persisted across restarts in `a362ad0`) retries every cluster_id → job_name mapping on every poll indefinitely, so a **transient** race between `SCRIPT PRE` writing `job.submitted` and `log_monitor` seeing the event-000 `DAGNodeName` block should self-heal. For a cluster_id to end up **permanently** `unknown:`, its `job.submitted` NDJSON record must never be found by `_job_name_to_run_id`, ever — pointing at a structural cause, not a timing fluke.

### Two candidate root causes (not yet confirmed against a live AP — both are directly supported by the code, either could be contributing)

**H1 — `provenance_log_dir` is resolved independently, per-process, from an environment variable that isn't reliably propagated.**

Four separate entry points each independently do `os.environ.get("PROVENANCE_LOG_DIR", "output/provenance")`: `pre.py` (writes `job.submitted`), `post.py` (writes `job.completed`/`job.failed`), `watcher.py` (writes epoch events), and `log_monitor.py`'s `provenance_log_dir` (used both to *search* for `job.submitted` via `_job_name_to_run_id` and to *write* every event it emits, including the `unknown:` fallback ones). `SCRIPT PRE`/`SCRIPT POST` are exec'd directly by `condor_dagman` and inherit its environment (i.e. whatever the shell that ran `condor_submit_dag` had set). The `provenance_monitor.sub` SUBMIT-DESCRIPTION is `universe = local` with **no `environment =` and no `getenv = true`** — a local-universe job does not inherit the submitting shell's environment by default, so if `PROVENANCE_LOG_DIR` is set to anything on the AP shell (even correctly, for a good reason), the SERVICE job's `log_monitor` process can silently resolve a *different* `provenance_log_dir` than `pre.py` used, and `_job_name_to_run_id` will search the wrong directory forever. Separately, `--classad-dir` (used for `.ad`/`.run_id`/offset/cache files) is hardcoded at DAG-generation time in `daggen.py`, so it can never agree with an env-var-driven `provenance_log_dir` unless both happen to default to the same string.
  - *Verify*: on the AP that ran the `many_protein_pretraining_with_ospool.dag` submission, check whether `PROVENANCE_LOG_DIR` was set in the shell/profile that called `condor_submit_dag`, and confirm the `job.submitted` record for a stuck cluster_id's job_name exists in a directory `log_monitor`'s own resolved `provenance_log_dir` was NOT searching.
  - *Fix direction*: stop deriving the log/event directory from independently-read environment variables in four different scripts. `daggen.py` already knows the intended directory at generation time (same way it already embeds `sys.executable` and `run_uuid` into `pre_args`/`post_args`) — bake an explicit `--log-dir <dir>` argument into the generated `SCRIPT PRE`/`SCRIPT POST`/`SERVICE` command lines instead, so every component agrees by construction rather than by environment inheritance.

**H2 — `_job_name_to_run_id` rescans and re-parses every `.ndjson` file on every lookup, and gets slower as the provenance directory grows.**

`_job_name_to_run_id` (`log_monitor.py:104`) does `for ndjson_path in provenance_log_dir.glob("*.ndjson")`, reading and JSON-decoding every line of every file, on **every** pending-lookup retry and every fresh DAGNodeName resolution. At current volume that's ~13,000 lines across ~1,800 files per single lookup call — and it's called once per pending cluster_id per poll cycle. As a long-running multi-hundred-job OSPool DAG accumulates more events, each lookup gets slower, which can make resolution lag behind job turnover: if a cluster_id's `.ad` ClassAd is cleaned up before `_resolve_run_id`'s method 3 fallback gets to it, and the job.submitted lookup never catches up either, the entry is stuck in `pending_lookups` and everything that already fired for it (which was emitted eagerly, not held back) stays `unknown:` in the NDJSON permanently.
  - *Verify*: time `_job_name_to_run_id` at current file counts (13k lines / 1.8k files); check whether `log_monitor`'s poll loop falls behind wall-clock time during a live run with many concurrent jobs.
  - *Fix direction*: build a `job_name -> run_id` index incrementally (update it as `job.submitted` events are observed, rather than re-scanning history every call), or cache per-file parse results keyed by mtime.

### Directory divergence (`provenance/` vs `output/provenance/`)

Byte-identical overlapping content across two directories, both currently untracked in git, strongly suggests a manual `cp`/`rsync` at some point (e.g. while debugging "does changing `PROVENANCE_LOG_DIR` fix it") rather than two live-diverging outputs — this needs confirming against actual shell/operational history on the AP (not visible from this checkout) rather than assumed. Document whichever explanation is confirmed, then retire the stale directory and make sure exactly one directory is the source of truth going forward (this task's H1 fix — baking `--log-dir` in at generation time — should also prevent recreating the split).

### Repair pass for existing data

Adapt the recovery approach already written up in `mldag_unknown_provenance_fix.md` (cross-reference `metl.log` event-000 `DAGNodeName` blocks against `job.submitted` records), but search `job.submitted` events across **both** `provenance/` and `output/provenance/` (not just DAG VARS, since these newer DAGs emit real `job.submitted` events via `pre.py`), and merge resolved events into the correct `<run_id>.ndjson`, consistent with how `provenance-db`'s loader (task-21) needs to treat this data.

### Testing

- Extend `tests/provenance/test_log_monitor_integration.py` with a case where `job.submitted` lives in a directory `log_monitor` is not configured to search, asserting the cluster_id stays correctly pending (not silently `unknown:`) and resolves once pointed at the right directory — locks in the H1 fix.
- Add a test asserting `_job_name_to_run_id`/its replacement does not re-read files it has already indexed on a subsequent call — locks in the H2 fix.

### Non-goals

- Fixing the separate epoch-duplication/overwrite behavior tracked in task-20 (different bug: repeated `epoch.started`/`epoch.completed` pairs for the same epoch number within one correctly-resolved run, not run_id resolution).
- Redesigning the resolution-order/fallback chain itself (cache → `.run_id` → `.ad` → `unknown:`) — the design is sound per the retry/persistence logic already in place; the goal here is making sure the inputs it depends on are actually reachable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### Update: real metl.log obtained, repair executed for real

The user supplied a current `metl.log` (2026-07-20 through 2026-08-05, 1770 DAGNodeName blocks) after the dry-run above. Re-running against real data surfaced a gap in the repair tool: `build_job_to_run_id` only harvested `job.submitted` events, missing `job.completed`/`job.failed` events that `post.py` writes with an equally-trustworthy job_name/run_id pair (resolved straight from the job's own ClassAd, no cluster_id ambiguity). Fixed to use the file's own name as run_id and accept a job_name from *any* event type -- every event in a properly-named `<run_id>.ndjson` file belongs to that run by construction (`events.py`'s `event_log_path`). Added `test_build_job_to_run_id_from_job_completed_event` to lock this in; suite is now 211 tests, all passing.

With the fresh log:

```
merged=956 duplicates_removed=0 skipped_no_dagnode=0 skipped_no_run_id=814 skipped_unparseable_name=1
```

Ran for real (not dry-run) with the user's confirmation. Verified: `unknown:*.ndjson` count in `output/provenance/` dropped from 1771 to 815 (814 genuinely unresolvable + 1 unparseable `unknown.ndjson`), and spot-checked that merged events landed in the correct `<run_id>.ndjson` with `run_id` correctly rewritten.

The remaining 814 are confirmed **not** a tool limitation: their job names (e.g. `run56-train_epoch12`, `run66-train_epoch19`) don't appear in the current `many_protein_pretraining_with_ospool.dag`, any `.dag.rescue*`/`.dag.bk` file, or any `job.submitted`/`job.completed`/`job.failed` NDJSON record anywhere in `output/provenance/` -- the specific `.dag` generation that submitted them has been overwritten by later `mldag-gen` runs and is not recoverable from local data.

This result narrows down which hypothesis was more likely dominant, though it doesn't prove it outright: 956 of 1770 (54%) were resolvable via a **batch, post-hoc** cross-reference once a complete `metl.log` existed, using `cluster_id -> job_name` and `job_name -> run_id` data that was, in the end, discoverable somewhere in local files. That's consistent with H2 (the O(n) rescan) rather than H1 (directory mismatch): if `pre.py`/`post.py` and `log_monitor` had simply been writing to two different directories the whole time (H1), the job_name/run_id records this repair pass found would not have existed in `output/provenance/` at all -- but they did. A more likely explanation is that `log_monitor`'s per-lookup full-directory rescan fell behind during the live run (job turnover on OSPool is fast relative to an O(total history) scan repeated every poll), so by the time a cluster_id's mapping was finally resolved, several of that cluster's early events had already been emitted with the `unknown:` fallback -- explaining both why the data was recoverable after the fact and why it wasn't resolved live. This is inference from the repair results, not a direct observation of the live monitor process; confirming it with certainty would need instrumenting `log_monitor` during an actual run. Both the H1 and H2 code fixes remain justified regardless -- they were found by direct code review of real structural weaknesses, not dependent on which one dominated historically -- and the H2 fix in particular directly addresses the mechanism this evidence points to.

Cleanup performed with explicit user confirmation: deleted the stale `provenance/` directory (byte-identical subset of `output/provenance/`) and the superseded root-level `recover_unknown_provenance.py`.
<!-- SECTION:NOTES:END -->
