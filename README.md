# mldag

HTCondor ML training DAG generation, monitoring, provenance tracking, and reporting.

## Installing

```bash
# From PyPI (once published)
uv add mldag

# From a pinned git commit (current approach)
uv add "mldag @ git+https://github.com/iross/MLDAG.git@<commit-or-tag>"
```

## Bootstrapping a new experiment repo

An experiment repo needs only these files — no framework code:

```
Experiment.yaml     # submit template, hyperparams, epoch/run counts
resources.yaml      # compute sites to target (CHTC, OSPool, Annex names)
config.yaml         # runtime settings (W&B API key, etc.)
.env                # secrets (gitignored)
pretrain_local.sh   # training script, calls mldag entry points to bracket training
justfile            # experiment-specific recipes (refresh, csv, report paths)
```

### Configuring which ClassAd fields land in provenance

`mldag-post` captures a curated subset of each job's HTCondor ClassAd (from
`job_ad_file`) into the `job.completed`/`job.failed` provenance events. The
default mapping (used when no file is configured, or the configured file
doesn't exist) is:

| ClassAd attribute | Provenance field |
|---|---|
| `RemoteWallClockTime` | `wall_time_s` |
| `CPUsUsage` | `cpu_usage` |
| `MemoryUsage` | `peak_memory_mb` |
| `GPUsUsage` | `gpu_usage` |
| `GLIDEIN_ResourceName` | `resource_name` |
| `Arguments` | `arguments` |

To capture different or additional fields, add a `classad_fields_file` entry
to `Experiment.yaml` pointing at a YAML file (conventionally
`provenance_fields.yaml`) listing the attributes to extract:

```yaml
fields:
  RequestCpus: num_cpus_requested   # explicit rename
  Cmd                                # bare entry -> auto snake_case ("cmd")
```

The path is baked into the generated DAG at `mldag-gen` time (like
`--log-dir`), so it can't drift between runs — but you can edit the file's
*contents* at any time without regenerating the DAG. Some ClassAd attributes
(currently just `Environment`, which can carry secrets like a W&B API key)
are blocked outright; requesting one raises an error when the file is loaded.

### Inspecting jobs without the full provenance pipeline

Two commands work on jobs that were never instrumented with the PRE/POST
provenance scripts (a one-off batch, a hand-submitted `.dag`) — no run_id or
NDJSON events required. Both enrich opportunistically when related data is
available and silently fall back to bare cluster_id/no-op when it isn't:

```bash
# Duration, execute host/site, and resource usage straight from an event log
mldag-query scan metl.log

# Filter to specific clusters (repeatable; every proc of a matched cluster is kept)
mldag-query scan metl.log --cluster-id 12345 --cluster-id 12399

# Dump results into provenance.db's condor_history table (source='event_log')
# instead of/alongside printing them
mldag-query scan metl.log --db provenance.db

# Backfill provenance.db's condor_history table from HTCondor job history
# (final host, exit code, hold reasons, requested vs. used resources)
mldag-query db enrich-history --schedd <name>
```

`scan` parses any HTCondor event log directly; if `--log-dir`/
`--provenance-log-dir` happen to point at a DAGMan provenance run's classad or
NDJSON directories, matching jobs are enriched with `run_id`/`job_name` too.
Jobs are keyed by `(cluster_id, proc_id)`, not `cluster_id` alone, since a
`queue N` job array puts many procs under one cluster.

`condor_history` holds rows from either source — `db enrich-history` (queried
from HTCondor) or `scan --db` (parsed from a raw event log) — distinguished by
its `source` column, since the two can disagree and neither is definitively
more current than the other. Writing from both for the same `(cluster_id,
proc_id)` never produces duplicate rows or a clobber: a write merges
column-by-column into any existing row (a column the new write doesn't know
about keeps its previous value), and `source` becomes `condor_history,event_log`
once both have contributed.

`db enrich-history` queries `condor_history` via the HTCondor Python bindings
(not the CLI) for every cluster_id already in `provenance.db`'s `events`
table, and is safe to re-run — already-enriched cluster_ids are skipped
unless `--full-rescan` is passed. Requires `htcondor2` (Linux only).

### Entry points after install

| Command | Purpose |
|---|---|
| `mldag-gen` | Generate DAG from Experiment.yaml |
| `mldag-csv` | Build metrics CSV from DAG files and training logs |
| `mldag-report` | Generate experiment report from CSV |
| `mldag-monitor` | HTCondor job monitor |
| `mldag-dashboard` | Generate interactive HTML dashboard |
| `mldag-query` | Query provenance records, scan raw event logs, build/enrich the SQLite db |
| `mldag-pre` / `mldag-post` | DAGMan pre/post scripts (provenance capture) |
| `mldag-log-monitor` | Provenance log monitor |

### Minimal justfile for a new experiment repo

```just
# Experiment-specific paths — override these
AP_HOST := "ap40"
AP_PATH := "/home/user/MY_EXPERIMENT"

_refresh:
    scp {{ AP_HOST }}:"{{ AP_PATH }}/metl.log" .
    scp {{ AP_HOST }}:"{{ AP_PATH }}/*.dag*" .

generate-csv:
    uv run mldag-csv --dag-files *.dag --metl-logs metl.log --output full.csv

generate-report:
    uv run mldag-report full.csv

hourly-site hours="24":
    just generate-csv
    uv run mldag-dashboard full.csv --output-dir site --hours {{ hours }}
    git -C site push --force https://github.com/user/MY_EXPERIMENT.git HEAD:gh-pages
```

