---
id: m-0
title: "Generalize artifact provenance"
---

## Description

Job-level provenance now converges cleanly: condor_history unifies three
independent sources (condor_history query, event-log scan, in-job jobad
capture) into one merge-safe, per-job table, regardless of how a job was
run or instrumented.

Artifact-level provenance hasn't had the same treatment. The checkpoints
table is narrow by construction: it's built from a fixed sidecar shape
(*.ckpt.provenance.json) with columns tailored specifically to ML training
checkpoints (val_loss, epoch, parent_hash lineage, hostname/slot/gpu info),
and walk_lineage()/query.py's lineage command only ever walk checkpoint-to-
checkpoint parent_hash chains. Any other kind of output artifact an
experiment produces -- an eval report, an exported/converted model, a
generated dataset, a plot, a packaged results bundle -- has no equivalent
capture path today.

Goal: generalize the same way the job side was just generalized -- one
artifact-provenance mechanism flexible enough to cover checkpoints and
whatever else a pipeline produces, without losing what already works well
(sidecar-file capture living next to the artifact, lineage via parent
hashes, a curated set of queryable columns backed by a flexible overflow
blob for anything sparse/experiment-specific -- checkpoints' own
training_json is already precedent for that hybrid).

Open questions for whoever scopes the first task here:
- What's the right unit -- a generic `artifacts` table (type + flexible
  payload + a small set of promoted/indexed common columns), with
  checkpoints becoming one `type` value instead of an entirely separate
  table? Or a family of type-specific tables sharing common conventions?
- Lineage is checkpoint-to-checkpoint only right now (parent_hash chain,
  walk_lineage). Generalizing likely means artifact-to-artifact edges that
  aren't restricted to one type -- e.g. an eval report's parent could be a
  checkpoint.
- Should the existing `<artifact>.provenance.json` sidecar convention just
  extend to non-checkpoint artifacts as-is, or does a generic artifact need
  a different capture trigger (there's no equivalent of the training loop's
  epoch-boundary sidecar write for, say, a one-off eval script)?
- Backward compatibility: existing checkpoint sidecars/schema and every
  query against `checkpoints` (CLI, dashboards, reports) need to keep
  working through whatever the migration looks like.

Scope this into concrete tasks only once these are answered -- this
milestone exists to hold the vision and constraints, not to prescribe the
schema up front.
