---
id: TASK-39
title: Add opt-in one/two-line hooks for real-time checkpoint and failure capture
status: To Do
assignee: []
created_date: '2026-09-11 01:43'
labels: []
dependencies:
  - TASK-31
  - TASK-38
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Even with zero-config execute-node auto-capture (task-38), the provenance package still cannot know anything about training progress -- checkpoints, epochs, loss -- or the real Python failure reason, without the training script telling it. Today the only path for any of this is watcher.py's post-hoc heuristic file scan, which guesses epoch number from checkpoint filenames and duration from file mtimes, and AP-side job.failed only ever sees an exit code and HTCondor's hold reason, never the actual exception.

This task adds opt-in integration points that an experiment's training script can adopt with one or two lines, trading a small amount of explicit wiring for real-time, exact provenance instead of after-the-fact guessing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single call (e.g. checkpoint_saved(path)) is available to invoke immediately after each checkpoint write, emitting exact real-time epoch/duration/parent-hash sidecars and epoch.started/epoch.completed events, replacing watcher.py's mtime/filename-guessing heuristics for scripts that adopt it
- [ ] #2 A ready-made PyTorch Lightning callback (matching this repo's existing epoch=N-step=M-val_loss=V.ckpt / metrics.csv convention) can be added with one line (e.g. trainer.callbacks.append(ProvenanceCallback())) and requires no other code changes
- [ ] #3 A context manager or decorator around the training entrypoint captures the real Python exception (class, message, traceback) on failure, which the AP-side job.failed cannot see today
- [ ] #4 watcher.py's existing post-hoc scan remains available as the fallback path for training scripts that adopt none of these hooks
<!-- AC:END -->
