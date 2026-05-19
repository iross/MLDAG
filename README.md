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

### Entry points after install

| Command | Purpose |
|---|---|
| `mldag-gen` | Generate DAG from Experiment.yaml |
| `mldag-csv` | Build metrics CSV from DAG files and training logs |
| `mldag-report` | Generate experiment report from CSV |
| `mldag-monitor` | HTCondor job monitor |
| `mldag-dashboard` | Generate interactive HTML dashboard |
| `mldag-query` | Query provenance records |
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

