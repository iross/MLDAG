#!/usr/bin/env python3
"""
Hourly Dashboard Generator

Generates a self-contained interactive HTML dashboard from HPC experiment CSV data
and (optionally) pushes it to the gh-pages branch for GitHub Pages hosting.
"""

import argparse
import textwrap
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

RESOURCE_COLORS = {
    "ospool": "#2E8B57",
    "expanse": "#4169E1",
    "delta": "#FF6347",
    "bridges2": "#9370DB",
    "anvil": "#FF8C00",
    "aws": "#FFD700",
}
FALLBACK_COLOR = "#808080"

GPU_COLORS = {
    "NVIDIA A100-SXM4-40GB": "#e78ac3",
    "NVIDIA A100-SXM4-80GB": "#fc8d62",
    "NVIDIA H200": "#66c2a5",
    "NVIDIA A40": "#8da0cb",
    "NVIDIA Tesla V100-SXM2-32GB": "#a6d854",
    "NVIDIA RTX A5000": "#ffd92f",
}


def resource_color(name: str) -> str:
    return RESOURCE_COLORS.get(name.lower(), FALLBACK_COLOR)


def format_resource(name: str) -> str:
    if name.lower() == "ospool":
        return "OSPool"
    nairr = {"expanse", "delta", "bridges2", "anvil", "aws"}
    if name.lower() in nairr:
        suffix = "NAIRR" if name.lower() != "aws" else "NAIRR"
        return f"{name.title()} ({suffix})"
    return name.title()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_file: Path, hours: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    time_cols = ["Submit Time", "Start Time", "End Time", "Held Time",
                 "Released Time", "Evicted Time", "Aborted Time"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Epoch"] = df["Job Name"].str.extract(r"epoch(\d+)", expand=False).astype("Int64")
    df["Run"] = df["Job Name"].str.extract(r"run(\d+)", expand=False).astype("Int64")
    df["Final Status"] = df["Final Status"].fillna("unknown")
    df["Is Successful"] = df["Final Status"].isin({"completed", "checkpointed"})

    if hours is not None:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
        activity_cols = [c for c in ["Submit Time", "Start Time", "End Time"] if c in df.columns]
        mask = pd.Series(False, index=df.index)
        for col in activity_cols:
            mask |= df[col].notna() & (df[col] >= cutoff)
        df = df[mask].copy()

    return df


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_status_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["Final Status"].value_counts()
    status_colors = {
        "completed": "#2ecc71",
        "checkpointed": "#27ae60",
        "held": "#e74c3c",
        "evicted": "#e67e22",
        "aborted": "#c0392b",
        "running": "#3498db",
        "idle": "#95a5a6",
        "unknown": "#bdc3c7",
    }
    colors = [status_colors.get(s, FALLBACK_COLOR) for s in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=colors,
        textinfo="label+value+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Job Status Distribution", height=400)
    return fig


def chart_resource_breakdown(df: pd.DataFrame) -> go.Figure:
    successful = df[df["Is Successful"] & (df["Execution Duration (seconds)"] > 0)].copy()
    if successful.empty:
        return _empty_fig("Resource Breakdown (no data)")

    longest = successful.loc[
        successful.groupby(["DAG Source", "Job Name"])["Execution Duration (seconds)"].idxmax()
    ]
    grouped = longest.groupby("Targeted Resource").agg(
        epochs=("Is Successful", "count"),
        hours=("Execution Duration (seconds)", lambda s: s.sum() / 3600),
    ).reset_index()
    grouped["color"] = grouped["Targeted Resource"].map(resource_color)
    grouped["label"] = grouped["Targeted Resource"].map(format_resource)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Successful Jobs", "Computation Time (hrs)"))
    fig.add_trace(go.Bar(
        x=grouped["label"], y=grouped["epochs"],
        marker_color=grouped["color"], name="Jobs",
        hovertemplate="<b>%{x}</b><br>Jobs: %{y}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=grouped["label"], y=grouped["hours"].round(1),
        marker_color=grouped["color"], name="Hours",
        hovertemplate="<b>%{x}</b><br>Hours: %{y:.1f}<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(title="Resource Breakdown", height=450, showlegend=False)
    return fig


def chart_job_timeline(df: pd.DataFrame) -> go.Figure:
    timeline_df = df[df["Start Time"].notna() & df["End Time"].notna()].copy()
    if timeline_df.empty:
        timeline_df = df[df["Start Time"].notna()].copy()
        if timeline_df.empty:
            return _empty_fig("Job Timeline (no data)")
        timeline_df["End Time"] = timeline_df["Start Time"] + pd.Timedelta(hours=1)

    status_colors = {
        "completed": "#2ecc71", "checkpointed": "#27ae60",
        "held": "#e74c3c", "evicted": "#e67e22",
        "aborted": "#c0392b", "running": "#3498db",
        "idle": "#95a5a6", "unknown": "#bdc3c7",
    }

    fig = go.Figure()
    for status, grp in timeline_df.groupby("Final Status"):
        color = status_colors.get(status, FALLBACK_COLOR)
        for _, row in grp.iterrows():
            duration_h = (row["End Time"] - row["Start Time"]).total_seconds() / 3600
            fig.add_trace(go.Scatter(
                x=[row["Start Time"], row["End Time"]],
                y=[row["Job Name"], row["Job Name"]],
                mode="lines",
                line=dict(color=color, width=8),
                name=status,
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Job Name']}</b><br>"
                    f"Status: {status}<br>"
                    f"Resource: {row.get('Targeted Resource', 'unknown')}<br>"
                    f"Duration: {duration_h:.2f} hrs<extra></extra>"
                ),
            ))

    for status, color in status_colors.items():
        if status in timeline_df["Final Status"].values:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=color, width=4),
                name=status, showlegend=True,
            ))

    fig.update_layout(
        title="Job Timeline",
        height=max(400, 30 * len(timeline_df) + 100),
        xaxis_title="Time",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_gpu_distribution(df: pd.DataFrame) -> go.Figure:
    gpu_df = df[df["Number of GPUs"] > 0].copy()
    if gpu_df.empty or "GPU Device Name" not in df.columns:
        return _empty_fig("GPU Distribution (no GPU data)")

    counts = gpu_df["GPU Device Name"].value_counts()
    colors = [GPU_COLORS.get(g, FALLBACK_COLOR) for g in counts.index]
    short_names = [g.replace("NVIDIA ", "").replace("Tesla ", "") for g in counts.index]
    fig = go.Figure(go.Pie(
        labels=short_names, values=counts.values,
        marker_colors=colors, textinfo="label+value+percent",
        hovertemplate="<b>%{label}</b><br>Jobs: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="GPU Type Distribution", height=400)
    return fig


def chart_gpu_hours(df: pd.DataFrame) -> go.Figure:
    gpu_df = df[
        (df["Number of GPUs"] > 0) &
        (df["Execution Duration (seconds)"] > 0)
    ].copy()
    if gpu_df.empty:
        return _empty_fig("GPU Hours by Resource (no GPU data)")

    gpu_df["GPU Hours"] = (gpu_df["Execution Duration (seconds)"] / 3600) * gpu_df["Number of GPUs"]
    grouped = gpu_df.groupby("Targeted Resource")["GPU Hours"].sum().reset_index()
    grouped["color"] = grouped["Targeted Resource"].map(resource_color)
    grouped["label"] = grouped["Targeted Resource"].map(format_resource)

    fig = go.Figure(go.Bar(
        x=grouped["label"], y=grouped["GPU Hours"].round(1),
        marker_color=grouped["color"],
        hovertemplate="<b>%{x}</b><br>GPU Hours: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(title="GPU Hours by Resource", xaxis_title="Resource",
                      yaxis_title="GPU Hours", height=400)
    return fig


def chart_data_transfer(df: pd.DataFrame) -> go.Figure:
    tx_df = df[(df["Total Bytes Sent"] > 0) | (df["Total Bytes Received"] > 0)].copy()
    if tx_df.empty:
        return _empty_fig("Data Transfer (no transfer data)")

    tx_df["Sent (GB)"] = tx_df["Total Bytes Sent"] / 1024**3
    tx_df["Received (GB)"] = tx_df["Total Bytes Received"] / 1024**3
    grouped = tx_df.groupby("Targeted Resource")[["Sent (GB)", "Received (GB)"]].sum().reset_index()
    grouped["label"] = grouped["Targeted Resource"].map(format_resource)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Sent", x=grouped["label"], y=grouped["Sent (GB)"].round(2),
        hovertemplate="<b>%{x}</b><br>Sent: %{y:.2f} GB<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Received", x=grouped["label"], y=grouped["Received (GB)"].round(2),
        hovertemplate="<b>%{x}</b><br>Received: %{y:.2f} GB<extra></extra>",
    ))
    fig.update_layout(
        title="Data Transfer by Resource", barmode="group",
        xaxis_title="Resource", yaxis_title="GB", height=400,
    )
    return fig


def chart_transfer_timing(df: pd.DataFrame) -> go.Figure:
    col = "Transfer Input Duration (seconds)"
    if col not in df.columns:
        return _empty_fig("Transfer Timing (column not available)")

    tx_df = df[df[col].notna() & (df[col] > 0)].copy()
    if tx_df.empty:
        return _empty_fig("Transfer Timing (no data)")

    tx_df["Duration (min)"] = tx_df[col] / 60
    resources = tx_df["Targeted Resource"].unique()
    fig = go.Figure()
    for res in resources:
        grp = tx_df[tx_df["Targeted Resource"] == res]["Duration (min)"]
        fig.add_trace(go.Box(
            y=grp, name=format_resource(res),
            marker_color=resource_color(res), boxpoints="outliers",
            hovertemplate="<b>%{x}</b><br>Duration: %{y:.1f} min<extra></extra>",
        ))
    fig.update_layout(title="Input Transfer Duration by Resource",
                      yaxis_title="Duration (minutes)", height=400)
    return fig


def chart_epochs_over_time(df: pd.DataFrame) -> go.Figure:
    completed = df[
        df["Final Status"].isin(["completed", "checkpointed"]) &
        df["End Time"].notna() &
        df["Epochs Completed"].notna() &
        (df["Epochs Completed"] > 0)
    ].copy()
    if completed.empty:
        return _empty_fig("Epochs Over Time (no completed epoch data)")

    completed["End Date"] = completed["End Time"].dt.floor("h")
    hourly = completed.groupby(["End Date", "Targeted Resource"])["Epochs Completed"].sum().reset_index()
    resources = hourly["Targeted Resource"].unique()

    min_t = hourly["End Date"].min()
    max_t = hourly["End Date"].max()
    date_range = pd.date_range(start=min_t, end=max_t, freq="h")

    fig = go.Figure()
    total = pd.Series(0.0, index=date_range)
    for res in resources:
        s = pd.Series(0.0, index=date_range)
        for _, row in hourly[hourly["Targeted Resource"] == res].iterrows():
            if row["End Date"] in s.index:
                s[row["End Date"]] = row["Epochs Completed"]
        total += s
        cum = s.cumsum()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            name=format_resource(res),
            line=dict(color=resource_color(res), width=2),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate="<b>%{x}</b><br>Cumulative epochs: %{y}<extra></extra>",
        ))
    cum_total = total.cumsum()
    fig.add_trace(go.Scatter(
        x=cum_total.index, y=cum_total.values,
        name="Total", line=dict(color="black", width=3),
        mode="lines+markers", marker=dict(size=5, symbol="square"),
        hovertemplate="<b>%{x}</b><br>Total cumulative: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Cumulative Epochs Completed Over Time",
                      xaxis_title="Time", yaxis_title="Cumulative Epochs", height=450)
    return fig


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No data available", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(title=title, height=300)
    return fig


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def build_summary_html(df: pd.DataFrame, hours: Optional[int]) -> str:
    total_attempts = len(df)
    successful = df["Is Successful"].sum()
    success_rate = (successful / total_attempts * 100) if total_attempts else 0.0

    epoch_total = 0
    if "Epochs Completed" in df.columns:
        epoch_total = int(df[df["Is Successful"]]["Epochs Completed"].sum())

    gpu_hours = 0.0
    if "Number of GPUs" in df.columns and "Execution Duration (seconds)" in df.columns:
        gpu_df = df[(df["Number of GPUs"] > 0) & (df["Execution Duration (seconds)"] > 0)]
        gpu_hours = (gpu_df["Execution Duration (seconds)"] / 3600 * gpu_df["Number of GPUs"]).sum()

    compute_hours = 0.0
    if "Execution Duration (seconds)" in df.columns:
        compute_hours = df[df["Is Successful"]]["Execution Duration (seconds)"].sum() / 3600

    window_label = f"last {hours}h" if hours else "all time"
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return textwrap.dedent(f"""
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{total_attempts:,}</div>
            <div class="stat-label">Job Attempts ({window_label})</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{success_rate:.1f}%</div>
            <div class="stat-label">Success Rate</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{epoch_total:,}</div>
            <div class="stat-label">Epochs Completed</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{compute_hours:.1f}h</div>
            <div class="stat-label">Compute Time</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{gpu_hours:.1f}</div>
            <div class="stat-label">GPU Hours</div>
          </div>
          <div class="stat-card updated">
            <div class="stat-value" style="font-size:1rem">{updated}</div>
            <div class="stat-label">Last Updated</div>
          </div>
        </div>
    """)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; background: #f5f7fa; color: #333; }
header { background: #1a1a2e; color: #eee; padding: 1.5rem 2rem; }
header h1 { margin: 0 0 0.25rem; font-size: 1.6rem; }
header p  { margin: 0; opacity: 0.7; font-size: 0.9rem; }
main { max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem; }
.stats-grid { display: grid;
              grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 1rem; margin-bottom: 2rem; }
.stat-card { background: #fff; border-radius: 8px; padding: 1.25rem 1rem;
             box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }
.stat-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
.stat-label { font-size: 0.78rem; color: #666; margin-top: 0.25rem; text-transform: uppercase; }
.chart-card { background: #fff; border-radius: 8px; padding: 1rem;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 1.5rem; }
h2 { color: #1a1a2e; font-size: 1.1rem; margin: 0 0 0.75rem; }
"""

def assemble_html(summary_html: str, chart_htmls: list[str], hours: Optional[int]) -> str:
    window_label = f"Last {hours} Hours" if hours else "All Time"
    charts_block = "\n".join(f'<div class="chart-card">{c}</div>' for c in chart_htmls)
    # First chart gets the CDN plotly.js; rest reuse it
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <meta http-equiv="refresh" content="3600">
          <title>ML Training Dashboard — {window_label}</title>
          <style>{CSS}</style>
        </head>
        <body>
          <header>
            <h1>ML Training Dashboard</h1>
            <p>Window: {window_label} &nbsp;·&nbsp;
               Auto-refreshes every hour</p>
          </header>
          <main>
            {summary_html}
            {charts_block}
          </main>
        </body>
        </html>
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_dashboard(csv_file: Path, output_dir: Path, hours: Optional[int]) -> None:
    print(f"Loading data from {csv_file} (window: {'all' if hours is None else f'{hours}h'})")
    df = load_data(csv_file, hours)
    print(f"  {len(df)} job attempts after filtering")

    output_dir.mkdir(parents=True, exist_ok=True)

    charts = [
        chart_status_pie(df),
        chart_resource_breakdown(df),
        chart_job_timeline(df),
        chart_gpu_distribution(df),
        chart_gpu_hours(df),
        chart_epochs_over_time(df),
        chart_data_transfer(df),
        chart_transfer_timing(df),
    ]

    chart_htmls = []
    for i, fig in enumerate(charts):
        include_js = "cdn" if i == 0 else False
        chart_htmls.append(to_html(fig, include_plotlyjs=include_js, full_html=False))

    summary_html = build_summary_html(df, hours)
    html = assemble_html(summary_html, chart_htmls, hours)

    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard written to {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML dashboard from experiment CSV data"
    )
    parser.add_argument("input_file", nargs="?", default="full.csv",
                        help="Input CSV file (default: full.csv)")
    parser.add_argument("--output-dir", default="site",
                        help="Directory to write index.html into (default: site)")
    parser.add_argument("--hours", type=int, default=24,
                        help="Restrict to jobs active in the last N hours (default: 24)")
    args = parser.parse_args()

    generate_dashboard(
        csv_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        hours=args.hours,
    )


if __name__ == "__main__":
    main()
