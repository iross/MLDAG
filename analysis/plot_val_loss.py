#!/usr/bin/env python3
"""
Plot val_loss curves for training runs that completed 30 full epochs.
"""

import csv
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class EpochPoint:
    epoch: int
    val_loss: float
    resource: str


def load_val_loss_data(csv_path: str) -> dict[str, dict[int, EpochPoint]]:
    """Load per-run val_loss data.

    Returns:
        Dict of run_uuid -> {epoch: EpochPoint}
    """
    # Prefer completed attempts; if none, keep the last attempt
    STATUS_PRIORITY = {'completed': 0, 'checkpointed': 1}

    run_epochs: dict[str, dict[int, EpochPoint]] = defaultdict(dict)
    # Track best status seen per (uuid, epoch) to pick the right attempt
    best_status: dict[tuple[str, int], int] = {}

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_loss = row.get('Val Loss', '')
            run_uuid = row.get('Run UUID', '')
            job_name = row.get('Job Name', '')
            resource = row.get('Targeted Resource', '')
            status = row.get('Final Status', '')

            if not val_loss or not run_uuid:
                continue

            m = re.search(r'_epoch(\d+)$', job_name)
            if not m:
                continue

            epoch = int(m.group(1)) + 1
            priority = STATUS_PRIORITY.get(status, 99)
            key = (run_uuid, epoch)

            if key not in best_status or priority < best_status[key]:
                best_status[key] = priority
                run_epochs[run_uuid][epoch] = EpochPoint(epoch, float(val_loss), resource)

    return dict(run_epochs)


MARKERS = {'aws': 'o', 'bridges2': 's', 'delta': '^', 'expanse': 'D', 'ospool': 'v', 'unknown': 'x'}
COLORS = {'aws': '#1f77b4', 'bridges2': '#ff7f0e', 'delta': '#2ca02c', 'expanse': '#d62728', 'ospool': '#9467bd'}
DISPLAY_NAMES = {'aws': 'AWS', 'bridges2': 'Bridges2', 'delta': 'Delta', 'expanse': 'Expanse', 'ospool': 'OSPool'}
# Canonical legend order: alphabetical by key
LEGEND_ORDER = ['aws', 'bridges2', 'delta', 'expanse', 'ospool']


def plot_val_loss(data: dict[str, dict[int, EpochPoint]], output_path: str,
                  show_lines: bool = False, show_bars: bool = True):
    """Plot val_loss points colored by Targeted Resource with an optional stacked bar subplot."""
    if show_bars:
        fig, (ax, ax_bar) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                          gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})
    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax_bar = None

    # --- Top panel: scatter plot ---

    # Draw faint connecting lines per run UUID
    uuid_colors = plt.cm.tab20.colors
    if show_lines:
        for i, uuid in enumerate(sorted(data)):
            epochs = data[uuid]
            xs = sorted(epochs)
            ys = [epochs[e].val_loss for e in xs]
            ax.plot(xs, ys, color=uuid_colors[i % len(uuid_colors)], linewidth=0.7, alpha=0.3)

    # Collect all points grouped by resource
    resource_points: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for uuid in sorted(data):
        for epoch, point in data[uuid].items():
            xs, ys = resource_points[point.resource or 'unknown']
            xs.append(point.epoch)
            ys.append(point.val_loss)

    # Plot ospool first (lower z-order) so other resources paint on top
    plot_order = sorted(resource_points, key=lambda r: 0 if r == 'ospool' else 1)
    for resource in plot_order:
        xs, ys = resource_points[resource]
        z = 2 if resource == 'ospool' else 3
        ax.scatter(xs, ys, s=20, marker=MARKERS.get(resource, 'x'),
                   color=COLORS.get(resource, 'gray'),
                   label=DISPLAY_NAMES.get(resource, resource), zorder=z, alpha=0.6)

    ax.set_ylabel('Val Loss', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='x', which='minor', length=4, width=0.8)
    ax.tick_params(axis='x', which='major', length=7)
    total_points = sum(len(eps) for eps in data.values())
    ax.set_title(f'Validation Loss by Epoch (n={total_points}, global config)', fontsize=15)
    # Build legend in canonical order
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[DISPLAY_NAMES[r]] for r in LEGEND_ORDER if DISPLAY_NAMES[r] in label_to_handle]
    ordered_labels = [DISPLAY_NAMES[r] for r in LEGEND_ORDER if DISPLAY_NAMES[r] in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, title='Resource', fontsize=11, title_fontsize=12, loc='lower left')
    ax.grid(True, alpha=0.3)

    if ax_bar is not None:
        # --- Bottom panel: stacked bar chart ---

        # Count resources per epoch
        all_epochs = sorted(set(e for eps in data.values() for e in eps))
        epoch_resource_counts: dict[int, dict[str, int]] = {e: defaultdict(int) for e in all_epochs}
        for uuid in data:
            for epoch, point in data[uuid].items():
                epoch_resource_counts[epoch][point.resource or 'unknown'] += 1

        # Build stacked bars in canonical order
        bottom = [0] * len(all_epochs)
        for resource in LEGEND_ORDER:
            counts = [epoch_resource_counts[e].get(resource, 0) for e in all_epochs]
            ax_bar.bar(all_epochs, counts, bottom=bottom, width=0.8,
                       color=COLORS.get(resource, 'gray'), label=DISPLAY_NAMES.get(resource, resource))
            bottom = [b + c for b, c in zip(bottom, counts)]

        ax_bar.set_xlabel('Epoch', fontsize=16)
        ax_bar.set_ylabel('Count', fontsize=14)
        ax_bar.tick_params(axis='both', labelsize=14)
        ax_bar.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax_bar.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax_bar.tick_params(axis='x', which='minor', length=4, width=0.8)
        ax_bar.tick_params(axis='x', which='major', length=7)

        fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.08)
    else:
        ax.set_xlabel('Epoch', fontsize=16)
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot val_loss curves for completed training runs")
    parser.add_argument("--input", "-i", default="full_with_val_loss.csv", help="Input CSV (default: full_with_val_loss.csv)")
    parser.add_argument("--output", "-o", default="reports/val_loss_curves.png", help="Output plot path")
    parser.add_argument("--lines", action="store_true", help="Draw faint connecting lines per run UUID")
    parser.add_argument("--bars", action="store_true", default=True, help="Show stacked bar chart (default: on)")
    parser.add_argument("--no-bars", action="store_false", dest="bars", help="Hide stacked bar chart")

    args = parser.parse_args()

    data = load_val_loss_data(args.input)
    print(f"Found {len(data)} runs")
    for uuid in sorted(data):
        epochs = data[uuid]
        missing = [i for i in range(max(epochs) + 1) if i not in epochs]
        resources = sorted(set(p.resource for p in epochs.values()))
        print(f"  {uuid}: {len(epochs)} checkpoints, missing epochs: {missing or 'none'}, resources: {resources}")

    plot_val_loss(data, args.output, show_lines=args.lines, show_bars=args.bars)


if __name__ == "__main__":
    main()
