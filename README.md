# mldag

HTCondor ML training DAG generation, monitoring, and reporting.

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
pretrain_local.sh   # training script; installs htprovenance to bracket training
justfile            # experiment-specific recipes (refresh, csv, report paths)
```

## Provenance (htprovenance)

Provenance capture and querying live in the separate
[htprovenance](https://github.com/iross/htprovenance) package, which mldag
depends on and installs automatically. Its README covers the ClassAd field
mapping, `htprovenance-query`, and event-log scanning.

mldag uses it in two places:

- **Submit side.** `mldag-gen` writes `<python> -m htprovenance.pre` /
  `htprovenance.post` as each node's `SCRIPT PRE`/`POST`, and
  `python3 -m htprovenance.log_monitor` as a DAGMan `SERVICE`.
- **Execute side.** `mldag-gen` sets `HTPROVENANCE_VERSION` in each job's
  environment; `pretrain.sh` / `pretrain_local.sh` pip-install htprovenance
  from its git tag at that version.

**Upgrading on an access point with DAGs in flight.** The `SCRIPT PRE`/`POST`
and `SERVICE` lines are plain strings in the generated `.dag`, run with
whatever mldag/htprovenance is installed in that Python environment at the
time. DAGs generated before mldag 0.1.0rc28 still say
`-m mldag.provenance.*`, and fail at their next PRE/POST once that environment
is upgraded. Let those DAGs finish first, or install the new mldag into a
separate venv for new DAGs. Jobs themselves are unaffected: each installs the
version pinned in its own environment. Regenerate `.dag` files rather than
hand-editing them.

## Entry points after install

| Command | Purpose |
|---|---|
| `mldag-gen` | Generate DAG from Experiment.yaml |
| `mldag-csv` | Build metrics CSV from DAG files and training logs |
| `mldag-report` | Generate experiment report from CSV |
| `mldag-monitor` | HTCondor job monitor |
| `mldag-dashboard` | Generate interactive HTML dashboard |

## Minimal justfile for a new experiment repo

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

