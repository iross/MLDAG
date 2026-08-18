---
id: task-3
title: Refactor to generalize the dag monitoring
status: To Do
assignee: []
created_date: '2025-07-07'
updated_date: '2026-08-18 13:49'
labels: []
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The dag monitoring, CSV/report, and dashboard code (mldag/monitor/dagman.py, mldag/report/csv.py, mldag/report/experiment.py, mldag/dashboard.py) is still heavily specialized for METL, this repo's original training workload. mldag/monitor/dagman.py alone has ~160 case-insensitive "metl" occurrences across ~20 identifiers (self.metl_log, metl_job_timing, parse_metl_log_timing, _parse_metl_log_line, apply_metl_data_to_all_jobs, etc.) — these are baked into method/attribute names, not just default values, so a different experiment repo can't use this code without also being named/shaped like METL. The reporting layer additionally has hardcoded, already-mutually-drifted compute-site and GPU display-color maps in report/experiment.py and dashboard.py (they disagree with each other on colors for the same GPU model today).

The log parsing and monitoring pieces should be generally useful, including information about the dag nodes, and the overall structure of the dag shouldn't matter to the "base mode" (reading from the dag, associated logs, and the job log only). External specifics (training runs, unique uuids, epochs) should keep existing for this use case but shouldn't be part of the primary framework.

Companion task task-23 covers the separate, already-more-generic provenance subsystem (mldag/provenance/) and introduces a shared mldag/constants.py for the default event-log filename — this task's dagman.py/csv.py changes should import that same constant rather than defining their own.

## Other notes
- This work should be done on a new feature git branch.
- The "base mode" should only read from the dag, associated logs, and the job log.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 mldag/monitor/dagman.py contains no METL-specific identifiers (metl_log, metl_job_timing, parse_metl_log_timing, _parse_metl_log_line, apply_metl_data_to_all_jobs, follow_metl_log, etc.) — renamed to a generic event-log vocabulary (event_log, job_timing, parse_event_log_timing, ...)
- [ ] #2 DAGStatusMonitor's event-log path is configurable via an --event-log CLI flag / constructor param and resolved consistently everywhere it's used — fixes the existing bug where 4 call sites (around _select_log_file, _get_cluster_timing_from_metl, _get_cluster_timing_details, monitor_live_tail) independently reconstruct Path("metl.log") relative to CWD instead of using the instance's resolved path, breaking when CWD != DAG directory
- [ ] #3 mldag/report/csv.py's --metl-logs CLI flag, metl_logs default, and the metl_*.log standalone-directory glob convention are renamed to generic equivalents sourced from the same shared default filename task-23 introduces
- [ ] #4 Compute-site and GPU display colors are sourced from a new shared, config-driven module (mldag/report/display_config.py) with a deterministic built-in fallback palette plus an optional display.yaml override, instead of the current hardcoded and mutually-drifted maps in report/experiment.py (format_resource_name_with_glidein, get_resource_colors, get_gpu_colors, the ospool_colors/nairr_colors/other_colors maps) and dashboard.py (RESOURCE_COLORS, GPU_COLORS)
- [ ] #5 Minimal smoke tests exist for DAGStatusMonitor and SimpleCSVGenerator event-log path resolution and for display_config.py's fallback/override behavior — none of mldag/monitor/dagman.py, mldag/report/csv.py, mldag/report/experiment.py, or mldag/dashboard.py have any test coverage today
- [ ] #6 `rg -i "metl"` under mldag/monitor/, mldag/report/, and mldag/dashboard.py returns no matches
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Depends on mldag/constants.py existing (DEFAULT_EVENT_LOG_NAME) — created by whichever of task-3/task-23 lands first; this task imports it into mldag/monitor/dagman.py and mldag/report/csv.py.
2. mldag/monitor/dagman.py: one scripted rename pass (longest-identifier-first to avoid partial-match collisions) covering DAGStatusMonitor.__init__ (self.metl_log -> self.event_log, self.last_metl_log_position -> self.last_event_log_position, self.metl_job_timing -> self.job_timing, new event_log: Optional[str] = None constructor param defaulting to dag_file.parent / DEFAULT_EVENT_LOG_NAME), the CLI (add --event-log), and the parsing methods as a block: parse_metl_log_timing -> parse_event_log_timing, _parse_gpu_info_from_metl -> _parse_gpu_info_from_event_log, _parse_glidein_resource_from_metl -> _parse_glidein_resource_from_event_log, apply_metl_data_to_all_jobs -> apply_event_log_data_to_all_jobs, _get_cluster_timing_from_metl -> _get_cluster_timing_from_event_log, _process_metl_line -> _process_event_log_line, _parse_metl_log_line -> _parse_event_log_line, follow_metl_log -> follow_event_log, plus local variable renames in the same pass. Bundle the CWD-vs-dag-dir bug fix (replace 4 independent Path(\"metl.log\") constructions with self.event_log) into this pass and call it out explicitly as a behavior change, not just a rename. Manual diff review grouped by method after the scripted pass, not blind acceptance.
3. mldag/report/csv.py: rename metl_logs -> event_logs (param, attribute, CLI flag --metl-logs -> --event-logs), parse_metl_log -> parse_event_log, extract_job_name_from_metl -> extract_job_name_from_event_log, and the metl_*.log standalone-dir glob convention -> derived from DEFAULT_EVENT_LOG_NAME's stem. Update README.md's --metl-logs example.
4. mldag/report/display_config.py (new): deterministic built-in fallback palette (seeded with today's existing color values) plus stable hash-based assignment for unconfigured names; optional display.yaml loader (default path, overridable via --display-config) with resources/glidein_aliases/gpu_colors sections; exposes format_resource_name(), resource_color(), gpu_color().
5. mldag/report/experiment.py and mldag/dashboard.py: remove the hardcoded/drifted color maps and GLIDEIN alias table, replace call sites with display_config.py imports, add --display-config CLI flag to both main()s. Move the current METL-specific color/alias values into an example display.yaml that ships with the METL experiment repo (per task-17's package/experiment split), not inside the mldag package.
6. Add smoke tests for DAGStatusMonitor/SimpleCSVGenerator event-log resolution and display_config.py fallback/override behavior, since none of these files have test coverage today.

Suggested order: after task-23's mldag/constants.py lands (or land it here first if task-23 hasn't) -> dagman.py rename+fix -> csv.py rename -> display_config.py -> experiment.py/dashboard.py consolidation together (they share the target module) -> smoke tests alongside each step, not after.
<!-- SECTION:PLAN:END -->
