#!/usr/bin/env python3
"""
Cache Performance Analysis

Analyzes transfer performance across different caches (endpoints) and sites over time
to identify performance issues and patterns.

By default, only performs slow transfer analysis by cache. Use --full for complete analysis.

Usage:
    python analyze_cache_performance.py [es_transfer_data.csv] [options]

Examples:
    # Basic slow transfer analysis (default threshold = 300s = 5 min)
    python analyze_cache_performance.py

    # Set slow transfer threshold to 10 minutes
    python analyze_cache_performance.py --slow-threshold 600

    # Full analysis with all visualizations and reports
    python analyze_cache_performance.py --full

    # Custom input and filter only very slow transfers
    python analyze_cache_performance.py es_transfer_data.csv --min-duration 600 --slow-threshold 900
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class CachePerformanceAnalyzer:
    """Analyze cache performance from Elasticsearch transfer data."""

    def __init__(self, csv_file: str, output_dir: str = "cache_reports", min_duration: float = 0, slow_threshold: float = 300):
        """Initialize analyzer.

        Args:
            csv_file: Path to ES transfer data CSV
            output_dir: Directory for output reports
            min_duration: Minimum transfer duration to include (seconds)
            slow_threshold: Threshold for identifying slow transfers (seconds)
        """
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.min_duration = min_duration
        self.slow_threshold = slow_threshold

        # Load data
        self.df = self.load_data()

        print(f"Loaded {len(self.df)} transfer records from {csv_file}")
        print(f"Output directory: {self.output_dir}")
        if min_duration > 0:
            print(f"Filtering for transfers > {min_duration}s ({min_duration/60:.1f} min)")
        print(f"Slow transfer threshold: {slow_threshold}s ({slow_threshold/60:.1f} min)")

    @staticmethod
    def _extract_domain(machine_name):
        """Extract domain from machine name (e.g., slot1@hostname.domain.edu -> domain.edu)."""
        if pd.isna(machine_name) or machine_name == '':
            return 'Unknown'

        # Extract hostname after @ symbol
        if '@' in str(machine_name):
            hostname = str(machine_name).split('@')[-1]
        else:
            hostname = str(machine_name)

        # Extract domain (last two parts of hostname)
        parts = hostname.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[1:])
        return hostname

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the ES transfer data."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        # Load data
        df = pd.read_csv(self.csv_file)

        print(f"\nOriginal columns: {list(df.columns)}")

        # Convert TransferStartTime from Unix timestamp to datetime
        if 'TransferStartTime' in df.columns:
            df['TransferStartTime'] = pd.to_datetime(df['TransferStartTime'], unit='s', errors='coerce')

        # Clean up column names (handle potential variations)
        column_mapping = {}
        for col in df.columns:
            if 'glidein_site' in col.lower() or 'machineattrglidein_site' in col.lower():
                column_mapping[col] = 'SiteRaw'
            elif 'machineattrname0' in col.lower():
                column_mapping[col] = 'MachineName'
            elif col == 'Endpoint':
                column_mapping[col] = 'Endpoint'
            elif col == 'AttemptTime':
                column_mapping[col] = 'AttemptTime'
            elif col == 'ClusterId':
                column_mapping[col] = 'ClusterId'

        df = df.rename(columns=column_mapping)

        # Create Site column: use domain from MachineName if SiteRaw is "2"
        if 'SiteRaw' in df.columns:
            df['Site'] = df['SiteRaw'].astype(str)

            if 'MachineName' in df.columns:
                # When SiteRaw is "2", extract domain from MachineName
                mask = df['SiteRaw'].astype(str) == '2'
                df.loc[mask, 'Site'] = df.loc[mask, 'MachineName'].apply(self._extract_domain)
        elif 'MachineName' in df.columns:
            # If no SiteRaw column, just extract from MachineName
            df['Site'] = df['MachineName'].apply(self._extract_domain)

        # Filter out records with missing critical data
        required_cols = ['TransferStartTime', 'AttemptTime']
        for col in required_cols:
            if col in df.columns:
                df = df[df[col].notna()].copy()

        # Remove duplicate records (same ClusterId, TransferStartTime, Endpoint, TransferUrl)
        # These are ES duplicates, not real retries
        dedup_cols = ['ClusterId', 'TransferStartTime', 'Endpoint']
        if 'TransferUrl' in df.columns:
            dedup_cols.append('TransferUrl')

        initial_count = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep='first').copy()
        duplicates_removed = initial_count - len(df)

        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed:,} duplicate records ({duplicates_removed/initial_count*100:.1f}%)")

        # Filter by minimum duration
        if self.min_duration > 0 and 'AttemptTime' in df.columns:
            df = df[df['AttemptTime'] > self.min_duration].copy()

        # Add derived columns
        if 'AttemptTime' in df.columns:
            df['AttemptTimeMinutes'] = df['AttemptTime'] / 60
            df['AttemptTimeHours'] = df['AttemptTime'] / 3600

        # Extract date and hour for time-based analysis
        if 'TransferStartTime' in df.columns:
            df['Date'] = df['TransferStartTime'].dt.date
            df['Hour'] = df['TransferStartTime'].dt.hour
            df['DayOfWeek'] = df['TransferStartTime'].dt.day_name()

        return df

    def save_figure(self, fig, filename: str):
        """Save figure to output directory."""
        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")

    def print_summary_statistics(self):
        """Print summary statistics about the data."""
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        print(f"\nTotal transfers: {len(self.df):,}")

        if 'TransferStartTime' in self.df.columns:
            print(f"Time range: {self.df['TransferStartTime'].min()} to {self.df['TransferStartTime'].max()}")
            duration = (self.df['TransferStartTime'].max() - self.df['TransferStartTime'].min())
            print(f"Duration: {duration.days} days, {duration.seconds // 3600} hours")

        if 'AttemptTime' in self.df.columns:
            print(f"\nTransfer Duration Statistics:")
            print(f"  Mean:   {self.df['AttemptTime'].mean():.1f}s ({self.df['AttemptTime'].mean()/60:.1f} min)")
            print(f"  Median: {self.df['AttemptTime'].median():.1f}s ({self.df['AttemptTime'].median()/60:.1f} min)")
            print(f"  Min:    {self.df['AttemptTime'].min():.1f}s")
            print(f"  Max:    {self.df['AttemptTime'].max():.1f}s ({self.df['AttemptTime'].max()/60:.1f} min)")
            print(f"  Std:    {self.df['AttemptTime'].std():.1f}s")

        if 'Endpoint' in self.df.columns:
            unique_endpoints = self.df['Endpoint'].nunique()
            print(f"\nUnique endpoints (caches): {unique_endpoints}")
            print(f"Top 5 endpoints by transfer count:")
            top_endpoints = self.df['Endpoint'].value_counts().head(5)
            for endpoint, count in top_endpoints.items():
                print(f"  {endpoint}: {count:,} transfers")

        if 'Site' in self.df.columns:
            unique_sites = self.df['Site'].nunique()
            print(f"\nUnique sites: {unique_sites}")
            print(f"Top 5 sites by transfer count:")
            top_sites = self.df['Site'].value_counts().head(5)
            for site, count in top_sites.items():
                print(f"  {site}: {count:,} transfers")

    def plot_cache_performance_over_time(self):
        """Plot cache performance over time showing transfer durations."""
        if 'Endpoint' not in self.df.columns or 'TransferStartTime' not in self.df.columns:
            print("Missing required columns for cache performance plot")
            return

        fig, ax = plt.subplots(figsize=(16, 10))

        # Get top caches by transfer count
        top_caches = self.df['Endpoint'].value_counts().head(10).index
        df_top = self.df[self.df['Endpoint'].isin(top_caches)].copy()

        # Plot scatter for each cache
        for i, cache in enumerate(top_caches):
            cache_data = df_top[df_top['Endpoint'] == cache]
            ax.scatter(cache_data['TransferStartTime'],
                      cache_data['AttemptTimeMinutes'],
                      label=cache,
                      alpha=0.6,
                      s=50,
                      edgecolors='black',
                      linewidth=0.3)

        ax.set_xlabel('Transfer Start Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Transfer Duration (minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Cache Performance Over Time (Top 10 Caches by Transfer Count)',
                    fontsize=14, fontweight='bold', pad=20)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df_top['Date'].unique())//10)))
        plt.xticks(rotation=45, ha='right')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        self.save_figure(fig, 'cache_performance_over_time')

    def plot_cache_performance_heatmap(self):
        """Create heatmap showing average transfer duration by cache and time period."""
        if 'Endpoint' not in self.df.columns or 'Date' not in self.df.columns:
            print("Missing required columns for heatmap")
            return

        # Get top caches
        top_caches = self.df['Endpoint'].value_counts().head(15).index
        df_top = self.df[self.df['Endpoint'].isin(top_caches)].copy()

        # Pivot: cache vs date with mean transfer time
        pivot = df_top.pivot_table(
            values='AttemptTimeMinutes',
            index='Endpoint',
            columns='Date',
            aggfunc='mean'
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(18, 10))

        # Create heatmap
        sns.heatmap(pivot,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Avg Transfer Duration (minutes)'},
                   linewidths=0.5,
                   linecolor='gray',
                   ax=ax)

        ax.set_title('Average Transfer Duration by Cache and Date', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cache Endpoint', fontsize=12, fontweight='bold')

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        self.save_figure(fig, 'cache_performance_heatmap')

    def plot_site_cache_performance(self):
        """Plot performance breakdown by site and cache."""
        if 'Site' not in self.df.columns or 'Endpoint' not in self.df.columns:
            print("Missing required columns for site-cache performance plot")
            return

        # Get top sites
        top_sites = self.df['Site'].value_counts().head(10).index
        df_top = self.df[self.df['Site'].isin(top_sites)].copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: Box plot of transfer times by site
        site_data = []
        site_labels = []
        for site in top_sites:
            site_transfers = df_top[df_top['Site'] == site]['AttemptTimeMinutes']
            if len(site_transfers) > 0:
                site_data.append(site_transfers)
                site_labels.append(f"{site}\n(n={len(site_transfers)})")

        bp1 = ax1.boxplot(site_data, labels=site_labels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax1.set_title('Transfer Duration Distribution by Site', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Site', fontsize=11)
        ax1.set_ylabel('Transfer Duration (minutes)', fontsize=11)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Average transfer time by site (bar chart)
        site_avg = df_top.groupby('Site')['AttemptTimeMinutes'].agg(['mean', 'count']).sort_values('mean', ascending=False)

        bars = ax2.barh(range(len(site_avg)), site_avg['mean'], color='coral', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(site_avg)))
        ax2.set_yticklabels([f"{site} (n={int(site_avg.loc[site, 'count'])})" for site in site_avg.index], fontsize=9)
        ax2.set_xlabel('Average Transfer Duration (minutes)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Transfer Duration by Site', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (idx, row) in enumerate(site_avg.iterrows()):
            ax2.text(row['mean'] + 0.5, i, f"{row['mean']:.1f}",
                    va='center', fontsize=9)

        plt.tight_layout()
        self.save_figure(fig, 'site_cache_performance')

    def plot_hourly_performance_pattern(self):
        """Plot performance patterns by hour of day."""
        if 'Hour' not in self.df.columns:
            print("Missing Hour column for hourly pattern plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Box plot by hour
        hourly_data = [self.df[self.df['Hour'] == h]['AttemptTimeMinutes'] for h in range(24)]
        bp = ax1.boxplot(hourly_data, labels=range(24), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)

        ax1.set_title('Transfer Duration Distribution by Hour of Day', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Hour of Day (UTC)', fontsize=11)
        ax1.set_ylabel('Transfer Duration (minutes)', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Transfer count and average duration by hour
        hourly_stats = self.df.groupby('Hour').agg({
            'AttemptTimeMinutes': ['mean', 'count']
        }).reset_index()
        hourly_stats.columns = ['Hour', 'AvgDuration', 'Count']

        ax2_twin = ax2.twinx()

        # Bar chart for count
        bars = ax2.bar(hourly_stats['Hour'], hourly_stats['Count'],
                      alpha=0.6, color='steelblue', label='Transfer Count')

        # Line chart for average duration
        line = ax2_twin.plot(hourly_stats['Hour'], hourly_stats['AvgDuration'],
                            color='red', marker='o', linewidth=2, markersize=6,
                            label='Avg Duration')

        ax2.set_xlabel('Hour of Day (UTC)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Transfer Count', fontsize=11, color='steelblue')
        ax2_twin.set_ylabel('Average Duration (minutes)', fontsize=11, color='red')
        ax2.set_title('Transfer Count and Average Duration by Hour', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        self.save_figure(fig, 'hourly_performance_pattern')

    def analyze_slow_transfers_by_cache(self, threshold_seconds=300):
        """Analyze slow transfers by cache showing where they occurred and when.

        Args:
            threshold_seconds: Duration threshold in seconds to consider a transfer slow
        """
        print("\n" + "=" * 100)
        print(f"SLOW TRANSFER ANALYSIS BY CACHE (Threshold: {threshold_seconds}s = {threshold_seconds/60:.1f} min)")
        print("=" * 100)

        # Filter for slow transfers
        slow_transfers = self.df[self.df['AttemptTime'] > threshold_seconds].copy()

        if len(slow_transfers) == 0:
            print(f"\nNo transfers found above threshold of {threshold_seconds}s")
            return

        print(f"\nTotal slow transfers: {len(slow_transfers):,} ({len(slow_transfers)/len(self.df)*100:.1f}% of all transfers)")
        print()

        # Group by cache endpoint
        if 'Endpoint' not in slow_transfers.columns:
            print("Missing Endpoint column")
            return

        cache_analysis = slow_transfers.groupby('Endpoint').agg({
            'AttemptTime': ['count', 'sum', 'mean', 'median', 'max'],
            'ClusterId': 'nunique'
        }).round(1)

        cache_analysis.columns = ['Count', 'Total (s)', 'Mean (s)', 'Median (s)', 'Max (s)', 'Unique Jobs']
        cache_analysis = cache_analysis.sort_values('Count', ascending=False)

        # Add total transfer count for each cache from full dataset
        total_transfers_by_cache = self.df.groupby('Endpoint').size()
        cache_analysis['Total Transfers'] = cache_analysis.index.map(total_transfers_by_cache)

        # Calculate percentage of transfers that are slow for each cache
        cache_analysis['% Slow'] = (cache_analysis['Count'] / cache_analysis['Total Transfers'] * 100).round(1)

        # Calculate percentage of all slow transfers
        cache_analysis['% of Total Slow'] = (cache_analysis['Count'] / len(slow_transfers) * 100).round(1)

        print("=" * 135)
        print("SLOW TRANSFERS BY CACHE")
        print("=" * 135)
        print(f"\n{'Cache Endpoint':<50} {'Slow':<8} {'Total':<8} {'% Slow':<9} {'% Total':<9} {'Mean (s)':<10} {'Max (s)':<10}")
        print("-" * 135)

        for endpoint, row in cache_analysis.head(20).iterrows():
            endpoint_short = endpoint[:47] + '...' if len(endpoint) > 50 else endpoint
            print(f"{endpoint_short:<50} {int(row['Count']):<8} {int(row['Total Transfers']):<8} {row['% Slow']:<9.1f} {row['% of Total Slow']:<9.1f} "
                  f"{row['Mean (s)']:<10.1f} "
                  f"{row['Max (s)']:<10.1f}")

        # Detailed analysis for top problematic caches
        print("\n" + "=" * 100)
        print("DETAILED ANALYSIS: TOP 10 CACHES WITH MOST SLOW TRANSFERS")
        print("=" * 100)

        top_caches = cache_analysis.head(10).index

        for cache in top_caches:
            cache_slow = slow_transfers[slow_transfers['Endpoint'] == cache].copy()

            print(f"\n{'─' * 100}")
            print(f"Cache: {cache}")
            print(f"{'─' * 100}")
            print(f"Slow transfers: {len(cache_slow):,} (threshold: >{threshold_seconds}s)")
            print(f"Average duration: {cache_slow['AttemptTime'].mean():.1f}s ({cache_slow['AttemptTime'].mean()/60:.1f} min)")

            # Time distribution
            if 'TransferStartTime' in cache_slow.columns:
                print(f"\nTime period: {cache_slow['TransferStartTime'].min()} to {cache_slow['TransferStartTime'].max()}")

                # Group by date
                if 'Date' in cache_slow.columns:
                    # date_counts = cache_slow.groupby('Date').size().sort_values(ascending=False)
                    date_counts = cache_slow.groupby('Date').size()
                    print(f"\nDates with slow transfers:")
                    for date, count in date_counts.head(10).items():
                        date_data = cache_slow[cache_slow['Date'] == date]
                        avg_duration = date_data['AttemptTime'].mean()
                        print(f"  {date}: {count:3d} transfers, avg {avg_duration:.1f}s ({avg_duration/60:.1f} min)")

            # Destination analysis (sites)
            if 'Site' in cache_slow.columns:
                site_counts = cache_slow.groupby('Site').agg({
                    'AttemptTime': ['count', 'mean', 'max']
                }).round(1)
                site_counts.columns = ['Count', 'Mean (s)', 'Max (s)']
                site_counts = site_counts.sort_values('Count', ascending=False)

                print(f"\nDestination sites (where transfers were going):")
                print(f"  {'Site':<30} {'Count':<8} {'Mean (s)':<10} {'Max (s)':<10}")
                print(f"  {'-'*58}")
                for site, row in site_counts.head(10).iterrows():
                    site_short = site[:27] + '...' if len(str(site)) > 30 else site
                    print(f"  {site_short:<30} {int(row['Count']):<8} {row['Mean (s)']:<10.1f} {row['Max (s)']:<10.1f}")

            # Show a few example transfers
            print(f"\nExample slow transfers (showing 5 slowest):")

            # Include TransferUrl and Attempt if available
            cols_to_show = ['TransferStartTime', 'AttemptTime', 'Site', 'ClusterId']
            if 'TransferUrl' in cache_slow.columns:
                cols_to_show.append('TransferUrl')
            if 'Attempt' in cache_slow.columns:
                cols_to_show.append('Attempt')

            examples = cache_slow.nlargest(5, 'AttemptTime')[cols_to_show].copy()
            examples['AttemptTimeMin'] = (examples['AttemptTime'] / 60).round(1)

            print(f"  {'Start Time':<20} {'Duration (min)':<15} {'Site':<25} {'Cluster ID':<12} {'Att':<5} {'File URL':<50}")
            print(f"  {'-'*127}")
            for _, row in examples.iterrows():
                start_time = row['TransferStartTime'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['TransferStartTime']) else 'N/A'
                site = str(row['Site'])[:22] + '...' if len(str(row['Site'])) > 25 else str(row['Site'])

                # Get attempt number
                attempt = str(int(row['Attempt'])) if 'Attempt' in row and pd.notna(row['Attempt']) else '-'

                # Extract filename from URL if available
                file_url = ''
                if 'TransferUrl' in row and pd.notna(row['TransferUrl']) and row['TransferUrl']:
                    url = str(row['TransferUrl'])
                    # Extract just the filename from the URL
                    filename = url.split('/')[-1]
                    file_url = filename[:47] + '...' if len(filename) > 50 else filename

                print(f"  {start_time:<20} {row['AttemptTimeMin']:<15.1f} {site:<25} {row['ClusterId']:<12} {attempt:<5} {file_url:<50}")

        # Summary by site (destination)
        if 'Site' in slow_transfers.columns:
            print("\n" + "=" * 100)
            print("SLOW TRANSFERS BY DESTINATION SITE")
            print("=" * 100)

            site_analysis = slow_transfers.groupby('Site').agg({
                'AttemptTime': ['count', 'mean', 'median', 'max'],
                'Endpoint': 'nunique'
            }).round(1)

            site_analysis.columns = ['Count', 'Mean (s)', 'Median (s)', 'Max (s)', 'Unique Caches']
            site_analysis = site_analysis.sort_values('Count', ascending=False)

            print(f"\n{'Site':<35} {'Count':<8} {'Mean (s)':<10} {'Median (s)':<12} {'Max (s)':<10} {'Caches':<8}")
            print("-" * 90)
            for site, row in site_analysis.head(15).iterrows():
                site_short = str(site)[:32] + '...' if len(str(site)) > 35 else str(site)
                print(f"{site_short:<35} {int(row['Count']):<8} {row['Mean (s)']:<10.1f} "
                      f"{row['Median (s)']:<12.1f} {row['Max (s)']:<10.1f} {int(row['Unique Caches']):<8}")

    def identify_problem_periods(self, threshold_multiplier=2.0):
        """Identify time periods with abnormally slow transfers.

        Args:
            threshold_multiplier: Multiple of median duration to flag as problematic
        """
        if 'TransferStartTime' not in self.df.columns or 'AttemptTime' not in self.df.columns:
            print("Missing required columns for problem period identification")
            return

        print("\n" + "=" * 80)
        print("PROBLEM PERIOD IDENTIFICATION")
        print("=" * 80)

        median_duration = self.df['AttemptTime'].median()
        threshold = median_duration * threshold_multiplier

        print(f"\nMedian transfer duration: {median_duration:.1f}s ({median_duration/60:.1f} min)")
        print(f"Problem threshold: {threshold:.1f}s ({threshold/60:.1f} min)")
        print(f"(>{threshold_multiplier}x median)")

        # Find problematic transfers
        problems = self.df[self.df['AttemptTime'] > threshold].copy()

        print(f"\nFound {len(problems)} problematic transfers ({len(problems)/len(self.df)*100:.1f}% of total)")

        if len(problems) == 0:
            return

        # Group by date and cache
        if 'Endpoint' in problems.columns:
            print("\nProblematic transfers by cache:")
            cache_problems = problems.groupby('Endpoint').agg({
                'AttemptTime': ['count', 'mean', 'median', 'max']
            }).round(1)
            cache_problems.columns = ['Count', 'Mean (s)', 'Median (s)', 'Max (s)']
            cache_problems = cache_problems.sort_values('Count', ascending=False).head(10)
            print(cache_problems.to_string())

        if 'Site' in problems.columns:
            print("\nProblematic transfers by site:")
            site_problems = problems.groupby('Site').agg({
                'AttemptTime': ['count', 'mean', 'median', 'max']
            }).round(1)
            site_problems.columns = ['Count', 'Mean (s)', 'Median (s)', 'Max (s)']
            site_problems = site_problems.sort_values('Count', ascending=False).head(10)
            print(site_problems.to_string())

        if 'Date' in problems.columns:
            print("\nProblematic transfers by date:")
            date_problems = problems.groupby('Date').agg({
                'AttemptTime': ['count', 'mean', 'median', 'max']
            }).round(1)
            date_problems.columns = ['Count', 'Mean (s)', 'Median (s)', 'Max (s)']
            date_problems = date_problems.sort_values('Count', ascending=False).head(10)
            print(date_problems.to_string())

    def generate_cache_report(self):
        """Generate detailed report for each cache."""
        if 'Endpoint' not in self.df.columns:
            print("Missing Endpoint column for cache report")
            return

        print("\n" + "=" * 80)
        print("DETAILED CACHE PERFORMANCE REPORT")
        print("=" * 80)

        # Get all caches
        caches = self.df['Endpoint'].value_counts()

        print(f"\nAnalyzing {len(caches)} caches...")

        for cache, count in caches.head(20).items():
            cache_data = self.df[self.df['Endpoint'] == cache]

            print(f"\n{'─' * 80}")
            print(f"Cache: {cache}")
            print(f"{'─' * 80}")
            print(f"Total transfers: {count:,}")

            if 'AttemptTime' in cache_data.columns:
                print(f"Duration statistics:")
                print(f"  Mean:   {cache_data['AttemptTime'].mean():.1f}s ({cache_data['AttemptTime'].mean()/60:.1f} min)")
                print(f"  Median: {cache_data['AttemptTime'].median():.1f}s ({cache_data['AttemptTime'].median()/60:.1f} min)")
                print(f"  Std:    {cache_data['AttemptTime'].std():.1f}s")
                print(f"  Min:    {cache_data['AttemptTime'].min():.1f}s")
                print(f"  Max:    {cache_data['AttemptTime'].max():.1f}s ({cache_data['AttemptTime'].max()/60:.1f} min)")

                # Percentiles
                p95 = cache_data['AttemptTime'].quantile(0.95)
                p99 = cache_data['AttemptTime'].quantile(0.99)
                print(f"  95th percentile: {p95:.1f}s ({p95/60:.1f} min)")
                print(f"  99th percentile: {p99:.1f}s ({p99/60:.1f} min)")

            if 'Site' in cache_data.columns:
                unique_sites = cache_data['Site'].nunique()
                print(f"Used by {unique_sites} unique sites")
                if unique_sites <= 5:
                    print(f"Sites: {', '.join(cache_data['Site'].unique())}")

    def generate_all_reports(self, full_analysis=False):
        """Generate analysis reports.

        Args:
            full_analysis: If True, generate all visualizations and detailed reports.
                          If False (default), only analyze slow transfers by cache.
        """
        print("\nGenerating cache performance analysis reports...")

        try:
            self.print_summary_statistics()

            print("\n1. Analyzing slow transfers by cache...")
            self.analyze_slow_transfers_by_cache(self.slow_threshold)

            if full_analysis:
                print("\n2. Generating cache performance over time plot...")
                self.plot_cache_performance_over_time()

                print("\n3. Generating cache performance heatmap...")
                self.plot_cache_performance_heatmap()

                print("\n4. Generating site-cache performance analysis...")
                self.plot_site_cache_performance()

                print("\n5. Generating hourly performance pattern analysis...")
                self.plot_hourly_performance_pattern()

                print("\n6. Identifying problem periods...")
                self.identify_problem_periods()

                print("\n7. Generating detailed cache reports...")
                self.generate_cache_report()

            print(f"\n✅ Analysis complete!")

        except Exception as e:
            print(f"❌ Error during report generation: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze cache performance from Elasticsearch transfer data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with defaults
  python analyze_cache_performance.py

  # Custom input file and output directory
  python analyze_cache_performance.py es_transfer_data.csv --output-dir reports/

  # Only analyze slow transfers (> 5 minutes)
  python analyze_cache_performance.py --min-duration 300
        """
    )

    parser.add_argument("input_file", nargs="?", default="es_transfer_data.csv",
                       help="Input CSV file path (default: es_transfer_data.csv)")
    parser.add_argument("--output-dir", default="cache_reports",
                       help="Output directory for reports (default: cache_reports)")
    parser.add_argument("--min-duration", type=float, default=0,
                       help="Minimum transfer duration to include in seconds (default: 0)")
    parser.add_argument("--slow-threshold", type=float, default=300,
                       help="Threshold in seconds for identifying slow transfers (default: 300 = 5 min)")
    parser.add_argument("--full", action="store_true",
                       help="Generate full analysis with all visualizations and reports (default: only slow transfer analysis)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    try:
        analyzer = CachePerformanceAnalyzer(args.input_file, args.output_dir, args.min_duration, args.slow_threshold)
        analyzer.generate_all_reports(full_analysis=args.full)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Make sure {args.input_file} exists and contains valid ES transfer data.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
