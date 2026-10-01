MONTH := env_var_or_default("MONTH", "11")

# Install / sync dependencies
install:
    uv sync

# Remove generated DAG files, run UUID directories, logs, and provenance output
clean:
    rm -f *.dag *.dag.* metl.log nodes.dag.status
    find . -maxdepth 1 -type d -name '????????' -exec rm -rf {} +
    rm -rf output/

# Private per-pool implementations (hidden from just --list)
_refresh-ospool:
    scp ap40:"/home/ian.ross/MLDAG_fixed_global/global_pretraining.dag*" .
    scp ap40:"/home/ian.ross/MLDAG_fixed_global/bigger_global_pretraining.dag*" .
    scp ap40:"/home/ian.ross/MLDAG_fixed_global/ospool_pretraining.dag*" .
    scp ap40:"/home/ian.ross/single_proteins/many_protein_pretraining_with_delta.dag*" .
    scp ap40:/home/ian.ross/MLDAG_fixed_global/metl.log .
    scp ap40:"/home/ian.ross/single_proteins/metl.log" metl_delta.log

_refresh-ospool-grand-total:
    scp ap40:/ospool/ap40/data/ian.ross/MLDAG/global_pretraining_with_ospool.dag global_pretraining_with_ospool_misconfigured.dag
    scp ap40:/ospool/ap40/data/ian.ross/MLDAG/global_pretraining_with_ospool_w_anvil.dag global_pretraining_with_ospool_w_anvil_misconfigured.dag
    scp ap40:/ospool/ap40/data/ian.ross/MLDAG/metl.log metl_misconfigured.log

_refresh-chtc:
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/nairr_config_in_chtc/metl.log" metl_control.log
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/nairr_config_in_chtc/trainingrun*.dag*" .
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/path_supplement_march_runs/metl.log" metl_experiment_devices.log
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/path_supplement_march_runs/experiment_devices.dag*" .
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/single_protein_models_gpu_device_constrained/metl.log" metl_single_protein_models_gpu_device_constrained.log
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/single_protein_models_gpu_device_constrained/many_protein_pretraining_with_ospool_device_constrained_runs.dag*" .
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/single_protein_models_gpu_device_constrained/many_protein_pretraining_with_ospool_device_constrained_runs_run2.dag*" .
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/single_protein_models_with_ospool/metl.log" metl_single_protein_models_with_ospool.log
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/single_protein_models_with_ospool/many_protein_pretraining_with_ospool.dag*" .
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/single_protein_models_with_ospool/many_protein_pretraining_with_ospool_run2.dag*" .
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/single_protein_models_metl_updates/metl.log" metl_single_protein_models_metl_updates.log
    scp iaross@ap2002.chtc.wisc.edu:"/home/iaross/single_protein_models_metl_updates/many_protein_pretraining_updated_metl.dag*" .

_db-build-chtc dir db checkpoint_dir:
    ssh iaross@ap2002.chtc.wisc.edu 'cd /home/iaross/{{ dir }} && ~/.local/bin/uv run mldag-query db build --db {{ db }} --checkpoint-dir {{ checkpoint_dir }}'

# Build each CHTC run's provenance DB on ap2002 from its /staging checkpoints
db-build-chtc:
    just _db-build-chtc single_protein_models_with_ospool with_ospool.db /staging/i/iaross/single_protein_checkpoints_with_ospool
    just _db-build-chtc single_protein_models_gpu_device_constrained gpu_device_constrained.db /staging/i/iaross/single_protein_models_gpu_device_constrained
    just _db-build-chtc single_protein_models_dgxspark dgxspark.db /staging/i/iaross/single_protein_models_dgxspark
    just _db-build-chtc single_protein_models_metl_updates metl_updates.db /staging/i/iaross/single_protein_models_metl_updates

# Copy the CHTC provenance DBs built by db-build-chtc into chtc_dbs/
fetch-dbs-chtc:
    mkdir -p chtc_dbs
    scp iaross@ap2002.chtc.wisc.edu:/home/iaross/single_protein_models_with_ospool/with_ospool.db chtc_dbs/
    scp iaross@ap2002.chtc.wisc.edu:/home/iaross/single_protein_models_gpu_device_constrained/gpu_device_constrained.db chtc_dbs/
    scp iaross@ap2002.chtc.wisc.edu:/home/iaross/single_protein_models_dgxspark/dgxspark.db chtc_dbs/
    scp iaross@ap2002.chtc.wisc.edu:/home/iaross/single_protein_models_metl_updates/metl_updates.db chtc_dbs/

_csv-ospool:
    uv run mldag-csv \
        --dag-files bigger_global_pretraining.dag global_pretraining.dag \
                    ospool_pretraining.dag many_protein_pretraining_with_delta.dag \
        --metl-logs metl.log metl_delta.log \
        --output full_ospool.csv

_csv-ospool-grand-total:
    uv run mldag-csv \
        --dag-files bigger_global_pretraining.dag global_pretraining.dag \
                    ospool_pretraining.dag \
                    global_pretraining_with_ospool_misconfigured.dag \
                    global_pretraining_with_ospool_w_anvil_misconfigured.dag \
        --metl-logs metl.log metl_misconfigured.log \
        --output full_ospool-grand-total.csv

_csv-chtc:
    uv run mldag-csv \
        --dag-files experiment_devices.dag* trainingrun*.dag* many_protein_pretraining_with_ospool.dag* many_protein_pretraining_with_ospool_device_constrained_runs.dag* many_protein_pretraining_with_ospool_run2.dag* many_protein_pretraining_with_ospool_device_constrained_runs_run2.dag* \
                    many_protein_pretraining_updated_metl.dag* \
        --metl-logs metl_control.log metl_experiment_devices.log metl_single_protein_models_gpu_device_constrained.log \
            metl_single_protein_models_with_ospool.log metl_single_protein_models_metl_updates.log \
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
    uv run mldag-report full_{{ pool }}.csv --output-dir {{ pool }}_`date +"%Y-%m-%d"`

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
    uv run mldag-dashboard full_ospool.csv --output-dir site --hours {{ hours }}
    rm -rf site/.git
    git -C site init
    git -C site add -A
    git -C site commit -m "Update dashboard $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git -C site push --force https://github.com/iross/MLDAG.git HEAD:gh-pages

# Monthly report (e.g. just monthly-report chtc 2025-10, or a bare 10 for the most recent October)
monthly-report pool="ospool" month=MONTH:
    just _csv-{{ pool }}
    uv run mldag-report full_{{ pool }}.csv --month {{ month }} --output-dir {{ pool }}_month_{{ month }}_reports

# Report for a custom date range (e.g. just date-range-report 2025-10-01 2026-09-30 ospool)
date-range-report start end pool="ospool":
    just _csv-{{ pool }}
    uv run mldag-report full_{{ pool }}.csv --start-date {{ start }} --end-date {{ end }} --output-dir {{ pool }}_{{ start }}_to_{{ end }}_reports

_csv-global-pretraining:
    uv run mldag-csv \
        --dag-files report_data/global_pretraining/bigger_global_pretraining.dag \
                    report_data/global_pretraining/global_pretraining.dag \
                    report_data/global_pretraining/ospool_pretraining.dag \
        --metl-logs report_data/global_pretraining/metl.log \
        --output full_global_pretraining.csv

_csv-ospool-misconfigured:
    uv run mldag-csv \
        --dag-files report_data/global_pretraining_with_ospool_misconfigured/global_pretraining_with_ospool.dag \
                    report_data/global_pretraining_with_ospool_misconfigured/global_pretraining_with_ospool_w_anvil.dag \
        --metl-logs report_data/global_pretraining_with_ospool_misconfigured/metl.log \
        --include-standalone \
        --standalone-dir report_data/global_pretraining_with_ospool_misconfigured/standalone \
        --output full_ospool_misconfigured.csv

# Full report for report_data/global_pretraining
full-report-global-pretraining:
    just _csv-global-pretraining
    uv run mldag-report full_global_pretraining.csv --output-dir final_report/global_pretraining

# Full report for report_data/global_pretraining_with_ospool_misconfigured
full-report-ospool-misconfigured:
    just _csv-ospool-misconfigured
    uv run mldag-report full_ospool_misconfigured.csv --output-dir final_report/ospool_misconfigured
