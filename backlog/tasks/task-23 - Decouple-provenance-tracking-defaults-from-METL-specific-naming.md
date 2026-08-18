---
id: TASK-23
title: Decouple provenance tracking defaults from METL-specific naming
status: To Do
assignee: []
created_date: '2026-08-17 15:09'
updated_date: '2026-08-18 13:49'
labels: []
dependencies:
  - TASK-21
  - TASK-22
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The provenance subsystem (mldag/provenance/) is already HTCondor-generic in its core logic (events, checkpoint sidecars, SQLite db, ClassAd parsing, event-code handling) — nothing there is METL-specific. But a handful of hardcoded defaults and one domain-specific parser still tie it to this repo's current METL/PyTorch-Lightning setup, blocking reuse by a different experiment repo: the event-log filename ("metl.log") is hardcoded in three independent places instead of flowing from Experiment.yaml, the repair CLI's flag is literally named --metl-log, and the checkpoint watcher's epoch/metric extraction assumes a PyTorch Lightning checkpoint-naming convention with no way to plug in a different format.

This task makes the provenance subsystem fully general: config-driven log/output paths with one source of truth, generically-named CLI flags, and a pluggable checkpoint-metadata-extraction interface. Companion task task-3 covers the heavier, separate METL coupling in mldag/monitor/dagman.py and the reporting/dashboard layer — the two should agree on the same shared default filename via a new mldag/constants.py module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single shared constant module (mldag/constants.py) defines the default event-log filename; mldag/daggen.py, mldag/provenance/log_monitor.py, and mldag/provenance/repair.py all import it — no independent "metl.log" literals remain in mldag/provenance/ or mldag/daggen.py
- [ ] #2 Experiment.yaml gains provenance_dir and event_log_filename fields, and submit_template supports an {event_log} substitution variable, so the event-log filename and provenance output directory are fully configurable end-to-end from Experiment.yaml through the generated .dag file to the provenance CLI invocations
- [ ] #3 repair.py's --metl-log CLI flag and metl_log parameter names are renamed to a generic equivalent (e.g. --event-log/event_log), with no backward-compatible alias
- [ ] #4 mldag/provenance/watcher.py's checkpoint epoch/metric extraction is exposed via a CheckpointMetadataExtractor interface, with the existing PyTorch Lightning filename/metrics.csv behavior preserved as the default implementation and selectable via a --checkpoint-extractor flag
- [ ] #5 Existing and updated provenance tests pass (tests/provenance/test_repair.py, tests/provenance/test_watcher.py), plus a new tests/test_daggen.py covering that generated .dag output uses the configured filename/dir rather than a literal
- [ ] #6 This repo's own Experiment.yaml and README.md are updated to reflect the new config fields, the {event_log} template variable, and the renamed CLI flag
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. mldag/constants.py (new): DEFAULT_EVENT_LOG_NAME = "condor_events.log". Dependency-free (no htcondor2/typer imports) so mldag/monitor/ and mldag/report/ (task-3) can import it too without pulling in daggen.py's deps.
2. mldag/models/experiment.py: add Optional fields `provenance_dir: Optional[str] = "output/provenance"` and `event_log_filename: Optional[str] = None` (None -> DEFAULT_EVENT_LOG_NAME) to the Experiment model.
3. mldag/daggen.py: remove the module-level PROVENANCE_DIR constant; read provenance_dir/event_log_filename off the Experiment object in main(); thread both as explicit params into get_script(), get_service() (replaces the literal `--log-file metl.log`), get_submit_description(), and get_ospool_submit_description(); extend the submit_template .format() calls to also pass `event_log=event_log_filename`. Add tests/test_daggen.py (does not exist today) asserting generated output uses the configured values, not literals.
4. mldag/provenance/log_monitor.py: change the --log-file CLI default from "metl.log" to DEFAULT_EVENT_LOG_NAME; update docstring.
5. mldag/provenance/repair.py: rename --metl-log -> --event-log, metl_log params -> event_log, throughout build_cluster_to_job()/recover()/docstrings/messages; update tests/provenance/test_repair.py accordingly (pure rename, no behavior change).
6. mldag/provenance/watcher.py: introduce CheckpointMetadataExtractor protocol (metric_name, parse_epoch, extract_metric, read_metrics) and a LightningCheckpointExtractor implementing today's exact behavior (current _parse_epoch/_parse_val_loss/_read_metrics_csv) as the default. scan_once()/watch_and_emit() take an `extractor` param and use extractor.metric_name as the emitted event's dict key instead of the hardcoded "val_loss" string. main() gains --checkpoint-extractor MODULE:CLASSNAME (importlib-based, no registry). Preserve the existing None-on-no-match graceful degradation. Update tests/provenance/test_watcher.py to test LightningCheckpointExtractor's methods instead of the old private functions.
7. README.md and this repo's Experiment.yaml: document the two new fields and the {event_log} template variable; update Experiment.yaml's `log = metl.log` line to `log = {event_log}`.

Suggested order: 1 -> 2 -> 3 (land early, currently untested) -> 4 -> 5 -> 6 -> 7. Coordinate step 1 with task-3, which imports the same constant into mldag/monitor/dagman.py and mldag/report/csv.py — only one of the two tasks should actually create mldag/constants.py; whichever lands first creates it, the other just imports it.
<!-- SECTION:PLAN:END -->
