---
id: TASK-25
title: Fix job_ad_file writing before its target directory exists in the sandbox
status: Done
assignee:
  - '@claude'
created_date: '2026-08-19 16:10'
updated_date: '2026-08-19 16:12'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live-system investigation (checking output/provenance/*.ndjson and provenance.db against the actual running METL DAGs) found that every job.completed/job.failed provenance event is missing all ClassAd-derived fields (wall_time_s, cpu_usage, peak_memory_mb, gpu_usage, resource_name, and now arguments) -- across 72 completed and 9 failed events, none have them. Zero output/provenance/*.ad files exist anywhere, despite job_ad_file being set in the submit description.

Root cause: job_ad_file = output/provenance/$(ClusterId).ad (mldag/daggen.py) is written by HTCondor's starter near job start, but output/provenance/ inside the job's sandbox is only created later by the training script itself (pretrain_local.sh:66, log_dir.mkdir(parents=True, exist_ok=True), which runs after pip-installing mldag). The starter's write to a nonexistent parent directory fails, silently -- job_ad_file failures aren't surfaced to the job or DAGMan -- so post.py's parse_classad() always gets a FileNotFoundError and resource_fields_from_classad() always returns {}, with no visible error anywhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 output/provenance/ (or whatever PROVENANCE_DIR resolves to) exists in the job sandbox before HTCondor's starter writes job_ad_file, without depending on the training script's own mkdir timing
- [x] #2 The fix does not clobber an experiment's own transfer_input_files list from its submit_template
- [x] #3 get_submit_description() and get_ospool_submit_description() both apply the fix identically
- [x] #4 tests/test_daggen.py covers that the generated submit description preserves an experiment's existing transfer_input_files value and adds the directory-materializing marker before job_ad_file
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in mldag/daggen.py: main() now touches a PROVENANCE_DIR/.keep marker file alongside the existing Path(PROVENANCE_DIR).mkdir(...) call. get_submit_description() and get_ospool_submit_description() both append `transfer_input_files = $(transfer_input_files), {PROVENANCE_DIR}/.keep` before the job_ad_file line -- the self-reference ($(transfer_input_files)) appends to whatever the experiment's own submit_template already set, rather than overwriting it (this pattern already existed elsewhere in this repo's Experiment.yaml template for the continue_from_checkpoint case, so it's a proven-working idiom here). Since preserve_relative_paths = true is already set in the live Experiment.yaml, transferring output/provenance/.keep as an input file materializes that exact nested directory structure in the sandbox before the job starts -- before HTCondor's starter writes job_ad_file, and well before pretrain_local.sh's own mkdir call. No changes needed to job_ad_file's path or to transfer_output_files, since output/provenance/$(ClusterId).ad still lands inside the output/ directory the experiment already transfers back.

Verified the .keep marker won't confuse anything else scanning PROVENANCE_DIR: log_monitor.py and watcher.py only glob for specific patterns (*.ndjson, metrics.csv, checkpoint patterns), never a blanket directory listing.

Could not run the new tests directly (htcondor2, a hard import in daggen.py, has no macOS wheel -- same pre-existing limitation as task-24's daggen tests, guarded the same way with pytest.importorskip). Verified the exact string-building logic in isolation (extracted the same textwrap.indent/rstrip("queue")/append sequence into a standalone script and asserted against it) to confirm the test assertions are correct against actual runtime behavior, not just plausible-looking. Full suite: 241 passed, 1 skipped (the daggen module). ruff check clean on both changed files (mldag/daggen.py, tests/test_daggen.py) -- the 2 findings ruff reports in daggen.py (unused `random` import, one-line `if...: continue`) are pre-existing, unrelated to this change.

Not yet verified against a real live job -- the fix addresses the mechanism (directory must exist before the starter writes job_ad_file) with high confidence given the evidence (0/81 completed+failed events have any ClassAd field, 0 .ad files anywhere in output/provenance/), but confirming it end-to-end requires regenerating a DAG with mldag-gen and running an actual job through HTCondor, which is out of reach in this environment. Recommend spot-checking the next real run's output/provenance/<cluster_id>.ad existence and the resulting job.completed event's fields.
<!-- SECTION:NOTES:END -->
