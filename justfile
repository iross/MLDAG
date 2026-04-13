# Install Python dependencies using uv
install:
    just --list

# Refresh data from remote
refresh-data:
    scp ap40:"/home/ian.ross/MLDAG_AWS/metl.log" metl_aws.log
    scp ap40:"/home/ian.ross/MLDAG_AWS/global_pretraining_in_aws.dag*" .
    scp ap40:"/home/ian.ross/MLDAG_fixed_global/global_pretraining.dag*" .
    scp ap40:"/home/ian.ross/MLDAG_fixed_global/bigger_global_pretraining.dag*" .
    scp ap40:/home/ian.ross/MLDAG_fixed_global/metl.log .

# Generate full CSV report from DAG files
generate-csv:
    uv run post_experiment_csv.py --dag-files bigger_global_pretraining.dag global_pretraining.dag --output full.csv

# Generate CSV and report from AWS DAG files only
generate-report-only-aws:
    uv run post_experiment_csv.py --dag-files global_pretraining_in_aws.dag --metl-logs metl_aws.log --output full_aws.csv
    uv run experiment_report.py full_aws.csv

# Generate full CSV report from all DAG files including AWS
generate-csv-with-aws:
    uv run post_experiment_csv.py \
        --dag-files bigger_global_pretraining.dag global_pretraining.dag global_pretraining_in_aws.dag \
        --metl-logs metl.log metl_aws.log \
        --output full.csv

# Generate experiment report from CSV
generate-report:
    uv run experiment_report.py full.csv

# Generate experiment report with dated output directory
generate-report-dated:
    uv run experiment_report.py full.csv --output-dir `date +"%Y-%m-%d"`

# Complete workflow: generate CSV from all sources and dated report
full-report: generate-csv-with-aws generate-report-dated

# Summarize the last 24 hours of job activity across all DAG sources
daily-summary: generate-csv-with-aws
    uv run experiment_report.py full.csv --hours 24 --output-dir daily_summary

# Summarize the last N hours of job activity (e.g. just recent-summary 48)
recent-summary hours: generate-csv-with-aws
    uv run experiment_report.py full.csv --hours {{ hours }} --output-dir recent_{{ hours }}h_summary

# Monthly report (set MONTH env var, e.g., MONTH=10 just monthly-report)
monthly-report month="${MONTH:-11}":
    uv run post_experiment_csv.py \
        --dag-files bigger_global_pretraining.dag global_pretraining.dag global_pretraining_in_aws.dag \
        --metl-logs metl.log metl_aws.log \
        --output full.csv
    uv run experiment_report.py full.csv --month {{ month }} --output-dir month_{{ month }}_reports
