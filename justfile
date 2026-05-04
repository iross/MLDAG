# Install / sync dependencies
install:
    uv sync

# Refresh data from remote
# pool=ospool pulls from ap40; pool=chtc pulls from ap2002.chtc.wisc.edu
refresh-data pool="ospool":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ pool }}" = "ospool" ]; then
        scp ap40:"/home/ian.ross/MLDAG_AWS/metl.log" metl_aws.log
        scp ap40:"/home/ian.ross/MLDAG_AWS/global_pretraining_in_aws.dag*" .
        scp ap40:"/home/ian.ross/MLDAG_fixed_global/global_pretraining.dag*" .
        scp ap40:"/home/ian.ross/MLDAG_fixed_global/bigger_global_pretraining.dag*" .
        scp ap40:"/home/ian.ross/MLDAG_fixed_global/ospool_pretraining.dag*" .
        scp ap40:/home/ian.ross/MLDAG_fixed_global/metl.log .
    elif [ "{{ pool }}" = "chtc" ]; then
        scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/nairr_config_in_chtc/metl.log" metl_control.log
        scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/nairr_config_in_chtc/trainingrun*.dag*" .
        scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/path_supplement_march_runs/metl.log" metl_experiment_devices.log
        scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/path_supplement_march_runs/experiment_devices.dag*" .
    else
        echo "Unknown pool: {{ pool }}. Use 'ospool' or 'chtc'." && exit 1
    fi

# Generate CSV report from DAG files for the given pool
generate-csv pool="ospool":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ pool }}" = "ospool" ]; then
        uv run mldag-csv \
            --dag-files bigger_global_pretraining.dag global_pretraining.dag \
                        global_pretraining_in_aws.dag ospool_pretraining.dag \
            --metl-logs metl.log metl_aws.log \
            --output full_ospool.csv
    elif [ "{{ pool }}" = "chtc" ]; then
        uv run mldag-csv \
            --dag-files experiment_devices.dag* trainingrun*.dag* \
            --metl-logs metl_control.log metl_experiment_devices.log \
            --output full_chtc.csv
    else
        echo "Unknown pool: {{ pool }}. Use 'ospool' or 'chtc'." && exit 1
    fi

# Generate experiment report from CSV
generate-report pool="ospool":
    uv run mldag-report full_{{ pool }}.csv

# Generate experiment report with dated output directory
generate-report-dated pool="ospool":
    uv run mldag-report full_{{ pool }}.csv --output-dir `date +"%Y-%m-%d"`

# Complete workflow: generate CSV and dated report
full-report pool="ospool": (generate-csv pool) (generate-report-dated pool)

# Summarize the last 24 hours of job activity
daily-summary pool="ospool": (generate-csv pool)
    uv run mldag-report full_{{ pool }}.csv --hours 24 --output-dir daily_summary

# Summarize the last N hours of job activity (e.g. just recent-summary 48)
recent-summary hours pool="ospool": (generate-csv pool)
    uv run mldag-report full_{{ pool }}.csv --hours {{ hours }} --output-dir recent_{{ hours }}h_summary

# Generate interactive HTML dashboard for the last N hours and push to GitHub Pages (ospool only)
hourly-site hours="24": (generate-csv "ospool")
    uv run hourly_dashboard.py full_ospool.csv --output-dir site --hours {{ hours }}
    rm -rf site/.git
    git -C site init
    git -C site add -A
    git -C site commit -m "Update dashboard $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git -C site push --force https://github.com/iross/MLDAG.git HEAD:gh-pages

# Monthly report (e.g. just monthly-report month=10 pool=chtc)
monthly-report month="${MONTH:-11}" pool="ospool":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ pool }}" = "ospool" ]; then
        uv run mldag-csv \
            --dag-files bigger_global_pretraining.dag global_pretraining.dag global_pretraining_in_aws.dag \
            --metl-logs metl.log metl_aws.log \
            --output full_ospool.csv
    elif [ "{{ pool }}" = "chtc" ]; then
        uv run mldag-csv \
            --dag-files experiment_devices.dag* trainingrun*.dag* \
            --metl-logs metl_control.log metl_experiment_devices.log \
            --output full_chtc.csv
    else
        echo "Unknown pool: {{ pool }}. Use 'ospool' or 'chtc'." && exit 1
    fi
    uv run mldag-report full_{{ pool }}.csv --month {{ month }} --output-dir month_{{ month }}_reports
