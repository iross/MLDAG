MONTH := env_var_or_default("MONTH", "11")

# Install / sync dependencies
install:
    uv sync

# Private per-pool implementations (hidden from just --list)
_refresh-ospool:
    scp ap40:"/home/ian.ross/MLDAG_AWS/metl.log" metl_aws.log
    scp ap40:"/home/ian.ross/MLDAG_AWS/global_pretraining_in_aws.dag*" .
    scp ap40:"/home/ian.ross/MLDAG_fixed_global/global_pretraining.dag*" .
    scp ap40:"/home/ian.ross/MLDAG_fixed_global/bigger_global_pretraining.dag*" .
    scp ap40:"/home/ian.ross/MLDAG_fixed_global/ospool_pretraining.dag*" .
    scp ap40:/home/ian.ross/MLDAG_fixed_global/metl.log .

_refresh-chtc:
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/nairr_config_in_chtc/metl.log" metl_control.log
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/nairr_config_in_chtc/trainingrun*.dag*" .
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/path_supplement_march_runs/metl.log" metl_experiment_devices.log
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/path_supplement_march_runs/experiment_devices.dag*" .

_csv-ospool:
    uv run mldag-csv \
        --dag-files bigger_global_pretraining.dag global_pretraining.dag \
                    global_pretraining_in_aws.dag ospool_pretraining.dag \
        --metl-logs metl.log metl_aws.log \
        --output full_ospool.csv

_csv-chtc:
    uv run mldag-csv \
        --dag-files experiment_devices.dag* trainingrun*.dag* \
        --metl-logs metl_control.log metl_experiment_devices.log \
        --output full_chtc.csv

# Refresh data from remote (pool=ospool or pool=chtc)
refresh-data pool="ospool":
    just _refresh-{{ pool }}

# Generate CSV report from DAG files for the given pool
generate-csv pool="ospool":
    just _csv-{{ pool }}

# Generate experiment report from CSV
generate-report pool="ospool":
    uv run mldag-report full_{{ pool }}.csv

# Generate experiment report with dated output directory
generate-report-dated pool="ospool":
    uv run mldag-report full_{{ pool }}.csv --output-dir `date +"%Y-%m-%d"`

# Complete workflow: generate CSV and dated report
full-report pool="ospool":
    just _csv-{{ pool }}
    uv run mldag-report full_{{ pool }}.csv --output-dir `date +"%Y-%m-%d"`

# Summarize the last 24 hours of job activity
daily-summary pool="ospool":
    just _csv-{{ pool }}
    uv run mldag-report full_{{ pool }}.csv --hours 24 --output-dir daily_summary

# Summarize the last N hours of job activity (e.g. just recent-summary 48)
recent-summary hours pool="ospool":
    just _csv-{{ pool }}
    uv run mldag-report full_{{ pool }}.csv --hours {{ hours }} --output-dir recent_{{ hours }}h_summary

# Generate interactive HTML dashboard for the last N hours and push to GitHub Pages (ospool only)
hourly-site hours="24":
    just _csv-ospool
    uv run hourly_dashboard.py full_ospool.csv --output-dir site --hours {{ hours }}
    rm -rf site/.git
    git -C site init
    git -C site add -A
    git -C site commit -m "Update dashboard $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git -C site push --force https://github.com/iross/MLDAG.git HEAD:gh-pages

# Monthly report (e.g. just monthly-report chtc 10)
monthly-report pool="ospool" month=MONTH:
    just _csv-{{ pool }}
    uv run mldag-report full_{{ pool }}.csv --month {{ month }} --output-dir month_{{ month }}_reports
