#!/bin/bash
# Rsync output/ to a destination, then prune checkpoints locally to last.ckpt
# and the single best (lowest val_loss) checkpoint per training run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"
TRAINING_LOGS="$OUTPUT_DIR/training_logs"

DRY_RUN=false
DEST=""

usage() {
    echo "Usage: $0 [--dry-run] <destination>"
    echo "  destination  rsync target (local path or user@host:/path)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) DEST="$1"; shift ;;
    esac
done

[[ -z "$DEST" ]] && usage

delete() {
    if $DRY_RUN; then
        echo "[dry-run] rm -rf $1"
    else
        rm -rf "$1"
    fi
}

# --- Step 1: rsync ---
echo "==> Syncing $OUTPUT_DIR/ to $DEST"
rsync_args=(-av --progress)
$DRY_RUN && rsync_args+=(--dry-run)
rsync "${rsync_args[@]}" "$OUTPUT_DIR/" "$DEST/"

# --- Step 2: prune checkpoints ---
echo "==> Pruning checkpoints under $TRAINING_LOGS"

for run_dir in "$TRAINING_LOGS"/*/; do
    [[ -d "$run_dir" ]] || continue
    ckpt_dir="$run_dir/checkpoints"
    [[ -d "$ckpt_dir" ]] || continue
    run_id=$(basename "$run_dir")

    # Delete time_checkpoints entirely — no val_loss, not useful locally
    if [[ -d "$ckpt_dir/time_checkpoints" ]]; then
        echo "  [$run_id] removing time_checkpoints/"
        delete "$ckpt_dir/time_checkpoints"
    fi

    # Find best checkpoint by lowest val_loss
    best=""
    best=$(find "$ckpt_dir" -maxdepth 1 -name "*val_loss=*.ckpt" \
        | awk -F'val_loss=' '{print $2, $0}' \
        | sort -g \
        | head -1 \
        | cut -d' ' -f2-)
    [[ -n "$best" ]] && echo "  [$run_id] best: $(basename "$best")"

    # Delete all val_loss checkpoints except best
    while IFS= read -r ckpt; do
        [[ -z "$ckpt" ]] && continue
        if [[ "$ckpt" == "$best" ]]; then
            echo "  [$run_id] keeping $(basename "$ckpt")"
        else
            echo "  [$run_id] deleting $(basename "$ckpt")"
            delete "$ckpt"
            [[ -f "$ckpt.provenance.json" ]] && delete "$ckpt.provenance.json"
        fi
    done < <(find "$ckpt_dir" -maxdepth 1 -name "*val_loss=*.ckpt")

    if [[ -f "$ckpt_dir/last.ckpt" ]]; then
        echo "  [$run_id] keeping last.ckpt"
    fi
done

echo "==> Done"
