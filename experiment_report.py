#!/usr/bin/env python3
"""
Experiment Report Generation Tool

Generates comprehensive analysis and visualizations of HPC experiment results
from CSV data produced by post_experiment_csv.py.

Usage:
    python experiment_report.py [input.csv] [--output-dir reports/]
"""

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress future warnings from pandas/seaborn
warnings.filterwarnings('ignore', category=FutureWarning)

class ExperimentAnalyzer:
    """Main class for analyzing experiment data and generating reports."""

    def __init__(self, csv_file: str, output_dir: str = "reports"):
        """Initialize the analyzer with data file and output directory."""
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set up consistent styling
        self.setup_plotting_style()

        # Load and validate data
        self.df = self.load_data()
        self.validate_data()

        print(f"Loaded {len(self.df)} job attempts from {csv_file}")
        print(f"Output directory: {self.output_dir}")

    def setup_plotting_style(self):
        """Set up consistent styling for all plots."""
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # Set matplotlib parameters for consistency
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.format': 'png'
        })

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the CSV data."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        # Load data
        df = pd.read_csv(self.csv_file)

        # Convert timing columns to datetime
        time_columns = ['Submit Time', 'Start Time', 'End Time', 'Held Time', 'Released Time', 'Evicted Time', 'Aborted Time']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Extract epoch information from job names
        df['Epoch'] = df['Job Name'].str.extract(r'epoch(\d+)', expand=False).astype('Int64')
        df['Run'] = df['Job Name'].str.extract(r'run(\d+)', expand=False).astype('Int64')

        # Clean up status column
        df['Final Status'] = df['Final Status'].fillna('unknown')

        # Categorize success/failure
        successful_statuses = {'completed', 'checkpointed'}
        df['Is Successful'] = df['Final Status'].isin(successful_statuses)

        return df

    def validate_data(self):
        """Validate the loaded data and print summary."""
        required_columns = [
            'Job Name', 'Final Status', 'Targeted Resource',
            'Execution Duration (seconds)', 'Total Bytes Sent', 'Total Bytes Received'
        ]

        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"\nData Summary:")
        print(f"- Time range: {self.df['Submit Time'].min()} to {self.df['Submit Time'].max()}")
        resources = [r for r in self.df['Targeted Resource'].unique() if pd.notna(r)]
        print(f"- Resources: {sorted(resources)}")
        statuses = [s for s in self.df['Final Status'].unique() if pd.notna(s)]
        print(f"- Job statuses: {sorted(statuses)}")

        # Handle epochs more carefully for mixed data
        epoch_data = self.df['Epoch'].dropna()
        if len(epoch_data) > 0:
            print(f"- Epochs found: {epoch_data.min()}-{epoch_data.max()} ({len(epoch_data)} jobs with epochs)")
        else:
            print("- No epoch information found")

    def extract_epoch_stats(self) -> pd.DataFrame:
        """Extract epoch completion statistics by resource, using longest execution per job."""
        # Filter for jobs with successful status and execution time
        # Include both epoch-based jobs and standalone jobs (like UUID 861e7e66)
        epoch_data = self.df[
            (self.df['Is Successful']) &
            (self.df['Execution Duration (seconds)'] > 0)
        ].copy()

        # Separate epoch-based jobs from standalone jobs
        epoch_jobs = epoch_data[epoch_data['Epoch'].notna()].copy()
        standalone_jobs = epoch_data[epoch_data['Epoch'].isna()].copy()

        print(f"Found {len(epoch_jobs)} successful epoch-based attempts and {len(standalone_jobs)} standalone attempts")

        # Combine epoch jobs and standalone jobs for processing
        all_jobs = pd.concat([epoch_jobs, standalone_jobs], ignore_index=True)

        # For each unique job (DAG + Job Name), select the attempt with longest execution time
        # This represents where the real computational work was done
        longest_executions = all_jobs.loc[
            all_jobs.groupby(['DAG Source', 'Job Name'])['Execution Duration (seconds)'].idxmax()
        ].copy()

        print(f"Selected {len(longest_executions)} longest executions from {len(all_jobs)} successful attempts ({len(epoch_jobs)} epoch-based + {len(standalone_jobs)} standalone)")

        # Calculate summary statistics
        summary_stats = []
        for resource in longest_executions['Targeted Resource'].unique():
            resource_data = longest_executions[longest_executions['Targeted Resource'] == resource]

            total_jobs = len(resource_data)
            completed_jobs = len(resource_data[resource_data['Final Status'] == 'completed'])
            checkpointed_jobs = len(resource_data[resource_data['Final Status'] == 'checkpointed'])
            successful_jobs = total_jobs  # All are successful by definition

            total_time = resource_data['Execution Duration (seconds)'].sum()

            # Calculate all-time efficiency (successful time vs total wall-clock time including failed attempts)
            all_resource_data = self.df[self.df['Targeted Resource'] == resource]
            total_wall_time = all_resource_data['Total Duration (seconds)'].sum()

            summary_stats.append({
                'Resource': resource,
                'Total Jobs': total_jobs,
                'Completed Jobs': completed_jobs,
                'Checkpointed Jobs': checkpointed_jobs,
                'Successful Jobs': successful_jobs,
                'Success Rate': successful_jobs / total_jobs if total_jobs > 0 else 0,  # Always 1.0 by definition
                'Total Time (hours)': total_time / 3600,
                'Successful Time (hours)': total_time / 3600,  # Same as total time since all are successful
                'Time Efficiency': total_time / total_wall_time if total_wall_time > 0 else 0
            })

        return pd.DataFrame(summary_stats)

    def analyze_gpu_utilization(self) -> Dict:
        """Analyze GPU utilization patterns using longest execution per job."""
        gpu_data = self.df[
            (self.df['Number of GPUs'] > 0) &
            (self.df['Execution Duration (seconds)'] > 0)
        ].copy()

        # Select only the longest execution per unique job (where real work was done)
        longest_gpu_executions = gpu_data.loc[
            gpu_data.groupby(['DAG Source', 'Job Name'])['Execution Duration (seconds)'].idxmax()
        ].copy()

        # GPU type distribution by resource
        gpu_by_resource = longest_gpu_executions.groupby(['Targeted Resource', 'GPU Device Name']).agg({
            'HTCondor Cluster ID': 'count',
            'Execution Duration (seconds)': 'sum',
            'GPU Memory MB': 'first'
        }).rename(columns={'HTCondor Cluster ID': 'Job Count'})

        # GPU efficiency metrics - simplified approach
        gpu_efficiency = longest_gpu_executions.groupby(['GPU Device Name']).agg({
            'Execution Duration (seconds)': ['sum', 'mean', 'count'],
            'GPU Memory MB': 'first'
        })

        # Flatten column names
        gpu_efficiency.columns = ['Total Time', 'Avg Time', 'Total Jobs', 'Memory MB']

        # Since we're only looking at successful longest executions, success rate is 100%
        gpu_efficiency['Successful Jobs'] = gpu_efficiency['Total Jobs']
        gpu_efficiency['Success Rate'] = 1.0

        return {
            'by_resource': gpu_by_resource,
            'efficiency': gpu_efficiency,
            'total_gpu_hours': longest_gpu_executions['Execution Duration (seconds)'].sum() / 3600
        }

    def analyze_transfer_input_timing(self) -> Dict:
        """Analyze input file transfer timing patterns."""
        # Filter for jobs with transfer input timing data
        transfer_timing_data = self.df[
            (self.df['Transfer Input Duration (seconds)'].notna()) &
            (self.df['Transfer Input Duration (seconds)'] > 0)
        ].copy()

        if transfer_timing_data.empty:
            return {
                'by_resource': pd.DataFrame(),
                'summary': {},
                'has_data': False
            }

        # Convert to minutes for readability
        transfer_timing_data['Transfer Input Duration (minutes)'] = transfer_timing_data['Transfer Input Duration (seconds)'] / 60

        # Transfer timing by resource
        timing_by_resource = transfer_timing_data.groupby('Targeted Resource').agg({
            'Transfer Input Duration (minutes)': ['mean', 'median', 'std', 'min', 'max', 'count']
        })

        # Flatten column names
        timing_by_resource.columns = ['Mean (min)', 'Median (min)', 'Std Dev (min)', 'Min (min)', 'Max (min)', 'Job Count']

        # Overall summary statistics
        summary = {
            'total_jobs': len(transfer_timing_data),
            'mean_duration_min': transfer_timing_data['Transfer Input Duration (minutes)'].mean(),
            'median_duration_min': transfer_timing_data['Transfer Input Duration (minutes)'].median(),
            'total_transfer_time_hours': transfer_timing_data['Transfer Input Duration (seconds)'].sum() / 3600
        }

        return {
            'by_resource': timing_by_resource,
            'raw_data': transfer_timing_data,
            'summary': summary,
            'has_data': True
        }

    def analyze_data_transfer(self) -> Dict:
        """Analyze data transfer patterns."""
        # Filter out jobs with no transfer data
        transfer_data = self.df[(self.df['Total Bytes Sent'] > 0) | (self.df['Total Bytes Received'] > 0)].copy()

        # Convert bytes to GB for readability
        transfer_data['Data Sent (GB)'] = transfer_data['Total Bytes Sent'] / (1024**3)
        transfer_data['Data Received (GB)'] = transfer_data['Total Bytes Received'] / (1024**3)
        transfer_data['Total Transfer (GB)'] = transfer_data['Data Sent (GB)'] + transfer_data['Data Received (GB)']

        # Transfer by resource
        transfer_by_resource = transfer_data.groupby('Targeted Resource').agg({
            'Data Sent (GB)': 'sum',
            'Data Received (GB)': 'sum',
            'Total Transfer (GB)': 'sum',
            'HTCondor Cluster ID': 'count'
        }).rename(columns={'HTCondor Cluster ID': 'Job Count'})

        # Transfer efficiency (GB per hour)
        transfer_data['Transfer Rate (GB/hr)'] = transfer_data['Total Transfer (GB)'] / (transfer_data['Execution Duration (seconds)'] / 3600)
        transfer_data['Transfer Rate (GB/hr)'] = transfer_data['Transfer Rate (GB/hr)'].replace([np.inf, -np.inf], np.nan)

        efficiency_by_resource = transfer_data.groupby('Targeted Resource')['Transfer Rate (GB/hr)'].agg(['mean', 'median', 'std'])

        return {
            'by_resource': transfer_by_resource,
            'efficiency': efficiency_by_resource,
            'total_transfer_gb': transfer_data['Total Transfer (GB)'].sum()
        }

    def save_figure(self, fig, filename: str):
        """Save figure with consistent formatting."""
        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")

    def format_resource_name(self, resource_name):
        """Format resource names for display: ospool -> OSPool, others -> Title Case."""
        if resource_name.lower() == 'ospool':
            return 'OSPool'
        else:
            return resource_name.title()

    def get_resource_colors(self, resources):
        """Get consistent colors for resources regardless of which ones are present."""
        # Define consistent color mapping for each resource
        resource_color_map = {
            'ospool': '#2E8B57',    # Sea Green
            'expanse': '#4169E1',   # Royal Blue
            'delta': '#FF6347',     # Tomato
            'bridges2': '#9370DB',  # Medium Purple
            'anvil': '#FF8C00'      # Dark Orange
        }

        # Return colors for the resources present in the data
        colors = []
        for resource in resources:
            resource_key = resource.lower()
            if resource_key in resource_color_map:
                colors.append(resource_color_map[resource_key])
            else:
                # Fallback color for unknown resources
                colors.append('#808080')  # Gray

        return colors

    def get_gpu_colors(self, gpu_types):
        """Get consistent colors for GPU types regardless of which ones are present."""
        # Define consistent color mapping for each GPU type (nice colors from Set2 palette)
        gpu_color_map = {
            'NVIDIA A100-SXM4-40GB': '#e78ac3',     # Pink
            'NVIDIA A100-SXM4-80GB': '#fc8d62',     # Coral orange
            'NVIDIA A40': '#ffd92f',                 # Yellow
            'NVIDIA H200': '#66c2a5',                # Soft teal
            'Tesla V100-SXM2-32GB': '#a6d854',      # Light green
            'NVIDIA RTX A5000': '#8da0cb',          # Light purple
            # Add fallback patterns for partial matches
            'a100-sxm4-40gb': '#e78ac3',
            'a100-sxm4-80gb': '#fc8d62',
            'a40': '#ffd92f',
            'h200': '#e78ac3',
            'v100-sxm2-32gb': '#a6d854',
            'v100': '#a6d854',
            'a100': '#66c2a5',
            'h200': '#66c2a5',
            'rtx a5000': '#8da0cb',
            'a5000': '#8da0cb'
        }

        # Return colors for the GPU types present in the data
        colors = []
        for gpu_type in gpu_types:
            # Try exact match first
            if gpu_type in gpu_color_map:
                colors.append(gpu_color_map[gpu_type])
            else:
                # Try normalized match
                gpu_key = gpu_type.lower().replace('nvidia ', '').replace('tesla ', '').replace('geforce ', '').replace('quadro ', '')
                if gpu_key in gpu_color_map:
                    colors.append(gpu_color_map[gpu_key])
                else:
                    # Fallback color for unknown GPU types
                    colors.append('#808080')  # Gray

        return colors

    def plot_resource_breakdown(self):
        """Generate resource utilization breakdown plots."""
        epoch_stats = self.extract_epoch_stats()

        # Format resource names for display
        formatted_resources = [self.format_resource_name(resource) for resource in epoch_stats['Resource']]

        # Get consistent colors for each resource
        colors = self.get_resource_colors(epoch_stats['Resource'])

        # Plot 1: Epoch completion by resource
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Successful jobs by resource
        bars = ax.bar(formatted_resources, epoch_stats['Successful Jobs'], color=colors)
        ax.set_title('Successful Epochs by Resource', fontsize=14, pad=20)
        ax.set_xlabel('Resource', fontsize=12)
        ax.set_ylabel('Number of Successful Epochs', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        self.save_figure(fig, 'resource_epoch_breakdown')

        # Plot 2: Total computation time by resource (separate pie chart)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Use the same consistent colors for pie charts
        pie_colors = colors

        # Total computation time by resource (pie chart)
        def make_time_autopct(values):
            def my_autopct(pct):
                absolute = pct/100.*sum(values)
                return f'{round(absolute)}h\n({pct:.1f}%)'
            return my_autopct

        wedges1, texts1, autotexts1 = ax.pie(
            epoch_stats['Total Time (hours)'],
            labels=formatted_resources,
            autopct=make_time_autopct(epoch_stats['Total Time (hours)']),
            startangle=90,
            colors=pie_colors,
            textprops={'fontsize': 16},  # Much larger font for poster
            labeldistance=1.1,  # Revert to original distance
            pctdistance=0.85    # Place percentages closer to edge but inside
        )
        ax.set_title('Total Computation Time by Resource', fontsize=40, pad=30)

        # Improve text readability for pie chart percentages
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)  # Larger font for values/percentages

        # Adjust layout to prevent label cutoff
        plt.tight_layout(pad=4.0)  # Extra padding to accommodate larger text
        self.save_figure(fig, 'resource_computation_time_breakdown')

        # Plot 3: Successful epochs by resource (separate pie chart)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Successful jobs by resource (pie chart)
        def make_autopct(values):
            def my_autopct(pct):
                absolute = int(round(pct/100.*sum(values)))
                return f'{absolute}\n({pct:.1f}%)'
            return my_autopct

        wedges2, texts2, autotexts2 = ax.pie(
            epoch_stats['Successful Jobs'],
            labels=formatted_resources,
            autopct=make_autopct(epoch_stats['Successful Jobs']),
            startangle=90,
            colors=pie_colors,
            textprops={'fontsize': 16},  # Much larger font for poster
            labeldistance=1.1,  # Revert to original distance
            pctdistance=0.85    # Place percentages closer to edge but inside
        )
        ax.set_title('Successful Epochs by Resource', fontsize=40, pad=30)

        # Improve text readability for pie chart percentages
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)  # Larger font for values/percentages

        # Adjust layout to prevent label cutoff
        plt.tight_layout(pad=4.0)  # Extra padding to accommodate larger text
        self.save_figure(fig, 'resource_epochs_breakdown')

    def plot_gpu_analysis(self):
        """Generate GPU utilization analysis plots."""
        gpu_analysis = self.analyze_gpu_utilization()

        if gpu_analysis['efficiency'].empty:
            print("No GPU data found for analysis")
            return

        # Plot 1: GPU type distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        gpu_counts = gpu_analysis['efficiency']['Total Jobs']
        colors = self.get_gpu_colors(gpu_counts.index)

        # Create autopct function that shows both count and percentage
        def make_gpu_autopct(values):
            def my_autopct(pct):
                absolute = int(round(pct/100.*sum(values)))
                return f'{absolute}\n({pct:.1f}%)'
            return my_autopct

        ax.pie(gpu_counts.values, labels=gpu_counts.index,
               autopct=make_gpu_autopct(gpu_counts.values),
               startangle=90, colors=colors)
        ax.set_title('GPU Type Distribution (by Job Count)', fontsize=30, pad=30)

        plt.tight_layout()
        self.save_figure(fig, 'gpu_distribution_analysis')

        # Plot 2: GPU efficiency metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Average job time by GPU type
        gpu_avg_time = gpu_analysis['efficiency']['Avg Time'] / 3600  # Convert to hours
        colors = self.get_gpu_colors(gpu_avg_time.index)
        bars = ax1.bar(range(len(gpu_avg_time)), gpu_avg_time.values, color=colors)
        ax1.set_title('Average Job Duration by GPU Type')
        ax1.set_xlabel('GPU Type')
        ax1.set_ylabel('Average Duration (hours)')
        ax1.set_xticks(range(len(gpu_avg_time)))
        ax1.set_xticklabels([name.replace('NVIDIA ', '').replace('Tesla ', '') for name in gpu_avg_time.index], rotation=45, ha='right')

        # GPU memory distribution
        gpu_memory = gpu_analysis['efficiency']['Memory MB'] / 1024  # Convert to GB
        colors = self.get_gpu_colors(gpu_memory.index)
        bars = ax2.bar(range(len(gpu_memory)), gpu_memory.values, color=colors)
        ax2.set_title('GPU Memory by Type')
        ax2.set_xlabel('GPU Type')
        ax2.set_ylabel('Memory (GB)')
        ax2.set_xticks(range(len(gpu_memory)))
        ax2.set_xticklabels([name.replace('NVIDIA ', '').replace('Tesla ', '') for name in gpu_memory.index], rotation=45, ha='right')

        plt.tight_layout()
        self.save_figure(fig, 'gpu_efficiency_analysis')

    def plot_gpu_runtime_by_resource(self):
        """Generate box plot of execution times by GPU device and resource using longest execution per job."""
        # Filter for GPU jobs with execution time data
        gpu_runtime_data = self.df[
            (self.df['Number of GPUs'] > 0) &
            (self.df['GPU Device Name'].notna()) &
            (self.df['GPU Device Name'] != '') &
            (self.df['Execution Duration (seconds)'].notna()) &
            (self.df['Execution Duration (seconds)'] > 0)
        ].copy()

        # Select only the longest execution per unique job (where real work was done)
        longest_gpu_executions = gpu_runtime_data.loc[
            gpu_runtime_data.groupby(['DAG Source', 'Job Name'])['Execution Duration (seconds)'].idxmax()
        ].copy()

        gpu_runtime_data = longest_gpu_executions

        if gpu_runtime_data.empty:
            print("No GPU runtime data found for analysis")
            return

        # Convert execution time to hours for better readability
        gpu_runtime_data['Execution Time (hours)'] = gpu_runtime_data['Execution Duration (seconds)'] / 3600

        # Clean up GPU names for better display and create ordered categories
        gpu_runtime_data['GPU Type'] = gpu_runtime_data['GPU Device Name'].str.replace('NVIDIA ', '').str.replace('Tesla ', '')

        # Create figure with appropriate size
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))

        # Get unique GPU types and sort them for consistent ordering
        gpu_types = sorted(gpu_runtime_data['GPU Type'].unique())

        # Create box plot with more spacing
        box_plot = sns.boxplot(data=gpu_runtime_data,
                              x='GPU Type',
                              y='Execution Time (hours)',
                              hue='Targeted Resource',
                              ax=ax,
                              width=0.6)  # Make boxes narrower for better separation

        # Customize the plot
        ax.set_title('Execution Time Distribution by GPU Type and Resource', fontsize=18, pad=25, fontweight='bold')
        ax.set_xlabel('GPU Device Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Execution Time (hours)', fontsize=14, fontweight='bold')

        # Improve x-axis labels
        ax.tick_params(axis='x', rotation=45, labelsize=11, pad=8)
        ax.tick_params(axis='y', labelsize=11)

        # Add vertical lines between GPU types for clear separation
        for i in range(len(gpu_types) - 1):
            ax.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

        # Enhance the legend
        legend = ax.legend(title='Targeted Resource',
                          bbox_to_anchor=(1.02, 1),
                          loc='upper left',
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          fontsize=11)
        legend.get_title().set_fontsize(12)
        legend.get_title().set_fontweight('bold')

        # Add alternating background colors for GPU types
        for i, gpu_type in enumerate(gpu_types):
            if i % 2 == 0:  # Every other GPU type gets a light background
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='lightblue', zorder=0)

        # Enhance grid
        ax.grid(True, alpha=0.4, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Set y-axis to log scale if there's a wide range of values
        y_range = gpu_runtime_data['Execution Time (hours)'].max() / gpu_runtime_data['Execution Time (hours)'].min()
        if y_range > 100:  # If range spans more than 2 orders of magnitude
            ax.set_yscale('log')
            ax.set_ylabel('Execution Time (hours) - Log Scale', fontsize=14, fontweight='bold')

        # Add a subtle border around the plot area
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('darkgray')

        # Adjust layout to accommodate labels and legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend

        self.save_figure(fig, 'gpu_runtime_by_resource_boxplot')

        # Print summary statistics
        print("\nGPU Runtime Summary by Device and Resource:")
        summary_stats = gpu_runtime_data.groupby(['GPU Type', 'Targeted Resource'])['Execution Time (hours)'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        for (gpu_type, resource), stats in summary_stats.iterrows():
            if stats['count'] > 0:
                print(f"{gpu_type} on {resource}: {int(stats['count'])} jobs, "
                      f"median: {stats['median']:.1f}h, mean: {stats['mean']:.1f}h ±{stats['std']:.1f}")

    def plot_gpu_capability_heatmap(self):
        """Generate heatmap showing average epoch time by GPU capability and number of GPUs."""
        # Filter for GPU jobs with execution time data
        gpu_runtime_data = self.df[
            (self.df['Number of GPUs'] > 0) &
            (self.df['GPU Device Name'].notna()) &
            (self.df['GPU Device Name'] != '') &
            (self.df['Execution Duration (seconds)'].notna()) &
            (self.df['Execution Duration (seconds)'] > 0)
        ].copy()

        # Select only the longest execution per unique job (where real work was done)
        longest_gpu_executions = gpu_runtime_data.loc[
            gpu_runtime_data.groupby(['DAG Source', 'Job Name'])['Execution Duration (seconds)'].idxmax()
        ].copy()

        if longest_gpu_executions.empty:
            print("No GPU runtime data found for heatmap analysis")
            return

        # Convert execution time to hours
        longest_gpu_executions['Execution Time (hours)'] = longest_gpu_executions['Execution Duration (seconds)'] / 3600

        # Clean up GPU names and create capability ranking
        longest_gpu_executions['GPU Type'] = longest_gpu_executions['GPU Device Name'].str.replace('NVIDIA ', '').str.replace('Tesla ', '')

        # Define GPU capability ranking (higher number = more capable)
        gpu_capability_map = {
            'H200': 4,           # Latest/most capable
            'A100-SXM4-80GB': 3,  # Same compute capability as 40GB
            'A100-SXM4-40GB': 3,  # Same compute capability as 80GB
            'A40': 2,
            'RTX A5000': 1,
            'V100-SXM2-32GB': 1  # Older architecture
        }

        # Add capability score
        longest_gpu_executions['GPU Capability'] = longest_gpu_executions['GPU Type'].map(gpu_capability_map)

        # Filter out unknown GPU types
        longest_gpu_executions = longest_gpu_executions[longest_gpu_executions['GPU Capability'].notna()]

        # Group by capability and number of GPUs, calculate average time
        heatmap_data = longest_gpu_executions.groupby(['GPU Capability', 'Number of GPUs'])['Execution Time (hours)'].agg(['mean', 'count']).reset_index()

        # Filter out cells with very few data points (less than 2 jobs)
        heatmap_data = heatmap_data[heatmap_data['count'] >= 2]

        if heatmap_data.empty:
            print("Insufficient data for heatmap (need at least 2 jobs per cell)")
            return

        # Pivot for heatmap
        pivot_data = heatmap_data.pivot(index='GPU Capability', columns='Number of GPUs', values='mean')

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Create heatmap
        sns.heatmap(pivot_data,
                   annot=True,
                   fmt='.1f',
                   cmap='viridis_r',  # Reverse viridis so darker = longer time
                   cbar_kws={'label': 'Average Execution Time (hours)'},
                   ax=ax)

        # Customize the plot
        ax.set_title('Average Epoch Time by GPU Capability and GPU Count', fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Number of GPUs', fontsize=14, fontweight='bold')
        ax.set_ylabel('CUDA Compute Capability', fontsize=14, fontweight='bold')

        # Create capability labels with CUDA compute capabilities
        capability_labels = {
            1: 'Compute 7.0-7.5\n(V100-SXM2-32GB,\nRTX A5000)',
            2: 'Compute 8.6\n(A40)',
            3: 'Compute 8.0\n(A100-SXM4-40GB,\nA100-SXM4-80GB)',
            4: 'Compute 9.0\n(H200)'
        }

        # Update y-axis labels with capability descriptions
        y_ticks = ax.get_yticks()
        y_labels = []
        for tick in y_ticks:
            if tick >= 0 and tick < len(pivot_data.index):
                capability_level = pivot_data.index[int(tick)]
                label = capability_labels.get(capability_level, f'Level {capability_level}')
                y_labels.append(label)
            else:
                y_labels.append('')
        ax.set_yticklabels(y_labels, rotation=0)

        plt.tight_layout()
        self.save_figure(fig, 'gpu_capability_heatmap')

        # Print summary
        print("\nGPU Capability Heatmap Summary:")
        print("Data points per cell:")
        count_pivot = heatmap_data.pivot(index='GPU Capability', columns='Number of GPUs', values='count')
        for capability in sorted(count_pivot.index):
            cap_label = capability_labels.get(capability, f'Level {capability}')
            for gpu_count in sorted(count_pivot.columns):
                if pd.notna(count_pivot.loc[capability, gpu_count]):
                    jobs = int(count_pivot.loc[capability, gpu_count])
                    avg_time = pivot_data.loc[capability, gpu_count]
                    print(f"  {cap_label}, {gpu_count} GPUs: {jobs} jobs, {avg_time:.1f}h avg")

    def plot_epochs_completed_over_time(self):
        """Generate cumulative epochs completed over time visualization."""
        # Filter for successfully completed jobs with end times and epoch information
        completed_data = self.df[
            (self.df['Final Status'].isin(['completed', 'checkpointed'])) &
            (self.df['End Time'].notna()) &
            (self.df['Epoch'].notna()) &
            (self.df['Epochs Completed'].notna()) &
            (self.df['Epochs Completed'] > 0)
        ].copy()

        if completed_data.empty:
            print("No completed epoch data found for time series analysis")
            return

        # Convert End Time to datetime and extract date
        completed_data['End Date'] = completed_data['End Time'].dt.date

        # For each job completion, count the epochs completed on that day
        daily_epochs = completed_data.groupby(['End Date', 'Targeted Resource'])['Epochs Completed'].sum().reset_index()

        # Get date range
        min_date = daily_epochs['End Date'].min()
        max_date = daily_epochs['End Date'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create complete date series for each resource
        resources = daily_epochs['Targeted Resource'].unique()

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Get consistent colors for resources
        colors = self.get_resource_colors(resources)

        cumulative_data = {}
        total_daily_epochs = pd.Series(0, index=date_range)

        for i, resource in enumerate(resources):
            resource_data = daily_epochs[daily_epochs['Targeted Resource'] == resource]

            # Create a complete date series with zeros for missing dates
            resource_series = pd.Series(0, index=date_range)

            # Fill in actual epoch counts
            for _, row in resource_data.iterrows():
                resource_series[pd.Timestamp(row['End Date'])] = row['Epochs Completed']

            # Add to total daily epochs
            total_daily_epochs += resource_series

            # Calculate cumulative sum
            cumulative_epochs = resource_series.cumsum()
            cumulative_data[resource] = cumulative_epochs

            # Plot the line
            formatted_resource = self.format_resource_name(resource)
            ax.plot(cumulative_epochs.index, cumulative_epochs.values,
                   label=formatted_resource, color=colors[i], linewidth=2, marker='o', markersize=3, alpha=0.7)

        # Calculate and plot total cumulative epochs
        total_cumulative = total_daily_epochs.cumsum()
        ax.plot(total_cumulative.index, total_cumulative.values,
               label='Total (All Resources)', color='black', linewidth=3.5, marker='s', markersize=5)

        # Customize the plot
        ax.set_title('Cumulative Epochs Completed Over Time by Resource', fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Epochs Completed', fontsize=14, fontweight='bold')

        # Format x-axis
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add legend
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                          frameon=True, fancybox=True, shadow=True, fontsize=11)
        legend.get_title().set_fontweight('bold')

        # Format dates on x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(date_range)//10)))

        # Enhance plot borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('darkgray')

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend

        self.save_figure(fig, 'epochs_completed_over_time')

        # Print summary statistics
        print("\nEpochs Completed Over Time Summary:")
        total_epochs_by_resource = {}
        for resource in resources:
            if resource in cumulative_data:
                total_epochs = cumulative_data[resource].iloc[-1]
                total_epochs_by_resource[resource] = total_epochs
                formatted_resource = self.format_resource_name(resource)
                print(f"{formatted_resource}: {int(total_epochs)} total epochs completed")

        total_all_epochs = int(total_cumulative.iloc[-1])
        print(f"Total across all resources: {total_all_epochs} epochs completed")

        # Print date range
        print(f"Time period: {min_date} to {max_date} ({(max_date - min_date).days + 1} days)")

    def plot_epochs_completed_over_time_september(self):
        """Generate cumulative epochs completed over time visualization for September only."""
        # Filter for successfully completed jobs with end times and epoch information
        completed_data = self.df[
            (self.df['Final Status'].isin(['completed', 'checkpointed'])) &
            (self.df['End Time'].notna()) &
            (self.df['Epoch'].notna()) &
            (self.df['Epochs Completed'].notna()) &
            (self.df['Epochs Completed'] > 0)
        ].copy()

        if completed_data.empty:
            print("No completed epoch data found for time series analysis")
            return

        # Convert End Time to datetime and extract date
        completed_data['End Date'] = completed_data['End Time'].dt.date

        # Filter for September only (any year, but typically 2024)
        completed_data = completed_data[
            (completed_data['End Time'].dt.month == 9)
        ].copy()

        if completed_data.empty:
            print("No completed epoch data found for September")
            return

        # For each job completion, count the epochs completed on that day
        daily_epochs = completed_data.groupby(['End Date', 'Targeted Resource'])['Epochs Completed'].sum().reset_index()

        # Get date range for September
        min_date = daily_epochs['End Date'].min()
        max_date = daily_epochs['End Date'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create complete date series for each resource
        resources = daily_epochs['Targeted Resource'].unique()

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Get consistent colors for resources
        colors = self.get_resource_colors(resources)

        cumulative_data = {}
        total_daily_epochs = pd.Series(0, index=date_range)

        for i, resource in enumerate(resources):
            resource_data = daily_epochs[daily_epochs['Targeted Resource'] == resource]

            # Create a complete date series with zeros for missing dates
            resource_series = pd.Series(0, index=date_range)

            # Fill in actual epoch counts
            for _, row in resource_data.iterrows():
                resource_series[pd.Timestamp(row['End Date'])] = row['Epochs Completed']

            # Add to total daily epochs
            total_daily_epochs += resource_series

            # Calculate cumulative sum
            cumulative_epochs = resource_series.cumsum()
            cumulative_data[resource] = cumulative_epochs

            # Plot the line
            formatted_resource = self.format_resource_name(resource)
            ax.plot(cumulative_epochs.index, cumulative_epochs.values,
                   label=formatted_resource, color=colors[i], linewidth=2, marker='o', markersize=3, alpha=0.7)

        # Calculate and plot total cumulative epochs
        total_cumulative = total_daily_epochs.cumsum()
        ax.plot(total_cumulative.index, total_cumulative.values,
               label='Total (All Resources)', color='black', linewidth=3.5, marker='s', markersize=5)

        # Customize the plot
        ax.set_title('Cumulative Epochs Completed Over Time by Resource (September)', fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Epochs Completed', fontsize=14, fontweight='bold')

        # Format x-axis
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add legend
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                          frameon=True, fancybox=True, shadow=True, fontsize=11)
        legend.get_title().set_fontweight('bold')

        # Format dates on x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(date_range)//10)))

        # Enhance plot borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('darkgray')

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend

        self.save_figure(fig, 'epochs_completed_over_time_september')

        # Print summary statistics
        print("\nEpochs Completed Over Time Summary (September):")
        total_epochs_by_resource = {}
        for resource in resources:
            if resource in cumulative_data:
                total_epochs = cumulative_data[resource].iloc[-1]
                total_epochs_by_resource[resource] = total_epochs
                formatted_resource = self.format_resource_name(resource)
                print(f"{formatted_resource}: {int(total_epochs)} total epochs completed")

        total_all_epochs = int(total_cumulative.iloc[-1])
        print(f"Total across all resources: {total_all_epochs} epochs completed")

        # Print date range
        print(f"Time period: {min_date} to {max_date} ({(max_date - min_date).days + 1} days)")

    def plot_epochs_trained_per_day(self):
        """Generate bar plot showing the number of epochs trained per day."""
        # Filter for successfully completed jobs with end times and epoch information
        completed_data = self.df[
            (self.df['Final Status'].isin(['completed', 'checkpointed'])) &
            (self.df['End Time'].notna()) &
            (self.df['Epoch'].notna()) &
            (self.df['Epochs Completed'].notna()) &
            (self.df['Epochs Completed'] > 0)
        ].copy()

        if completed_data.empty:
            print("No completed epoch data found for epochs per day analysis")
            return

        # Convert End Time to datetime and extract date
        completed_data['End Date'] = completed_data['End Time'].dt.date

        # For each job completion, count the epochs completed on that day
        daily_epochs = completed_data.groupby(['End Date', 'Targeted Resource'])['Epochs Completed'].sum().reset_index()

        # Get date range
        min_date = daily_epochs['End Date'].min()
        max_date = daily_epochs['End Date'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create complete date series for each resource
        resources = daily_epochs['Targeted Resource'].unique()

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Get consistent colors for resources
        colors = self.get_resource_colors(resources)

        # Prepare data for stacked bar chart
        resource_data = {}
        for resource in resources:
            resource_subset = daily_epochs[daily_epochs['Targeted Resource'] == resource]
            # Create a complete date series with zeros for missing dates
            resource_series = pd.Series(0, index=date_range)
            # Fill in actual epoch counts
            for _, row in resource_subset.iterrows():
                resource_series[pd.Timestamp(row['End Date'])] = row['Epochs Completed']
            resource_data[resource] = resource_series

        # Create stacked bar chart
        bottom = pd.Series(0, index=date_range)
        for i, resource in enumerate(resources):
            formatted_resource = self.format_resource_name(resource)
            ax.bar(resource_data[resource].index, resource_data[resource].values,
                   bottom=bottom, label=formatted_resource, color=colors[i], alpha=0.8)
            bottom += resource_data[resource]

        # Customize the plot
        ax.set_title('Epochs Trained Per Day by Resource', fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Epochs Trained', fontsize=14, fontweight='bold')

        # Format x-axis
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)

        # Add legend
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                          frameon=True, fancybox=True, shadow=True, fontsize=11)
        legend.get_title().set_fontweight('bold')

        # Format dates on x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(date_range)//10)))

        # Enhance plot borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('darkgray')

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend

        self.save_figure(fig, 'epochs_trained_per_day')

        # Print summary statistics
        print("\nEpochs Trained Per Day Summary:")
        total_epochs_by_day = bottom
        days_with_epochs = (total_epochs_by_day > 0).sum()
        total_epochs = int(total_epochs_by_day.sum())
        avg_epochs_per_active_day = total_epochs / days_with_epochs if days_with_epochs > 0 else 0
        max_epochs_day = total_epochs_by_day.max()
        max_epochs_date = total_epochs_by_day[total_epochs_by_day == max_epochs_day].index[0].date()

        print(f"Total epochs trained: {total_epochs}")
        print(f"Days with epoch completions: {days_with_epochs} out of {len(date_range)} days")
        print(f"Average epochs per active day: {avg_epochs_per_active_day:.1f}")
        print(f"Peak day: {max_epochs_date} with {int(max_epochs_day)} epochs")

        # Print breakdown by resource
        print("\nEpochs per day by resource:")
        for resource in resources:
            total_resource_epochs = resource_data[resource].sum()
            formatted_resource = self.format_resource_name(resource)
            print(f"{formatted_resource}: {int(total_resource_epochs)} total epochs")

    def plot_gpu_hours_over_time(self):
        """Generate cumulative GPU hours utilized over time visualization."""
        # Filter for GPU jobs with execution time data
        gpu_data = self.df[
            (self.df['Number of GPUs'] > 0) &
            (self.df['End Time'].notna()) &
            (self.df['Execution Duration (seconds)'].notna()) &
            (self.df['Execution Duration (seconds)'] > 0) &
            (self.df['Final Status'].isin(['completed', 'checkpointed', 'held', 'evicted']))  # Include all jobs that used GPU time
        ].copy()

        if gpu_data.empty:
            print("No GPU usage data found for time series analysis")
            return

        # Calculate GPU hours for each job (execution time * number of GPUs)
        gpu_data['GPU Hours'] = (gpu_data['Execution Duration (seconds)'] / 3600) * gpu_data['Number of GPUs']

        # Convert End Time to datetime and extract date
        gpu_data['End Date'] = gpu_data['End Time'].dt.date

        # For each job completion, sum the GPU hours used on that day
        daily_gpu_hours = gpu_data.groupby(['End Date', 'Targeted Resource'])['GPU Hours'].sum().reset_index()

        # Get date range
        min_date = daily_gpu_hours['End Date'].min()
        max_date = daily_gpu_hours['End Date'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create complete date series for each resource
        resources = daily_gpu_hours['Targeted Resource'].unique()

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Get consistent colors for resources
        colors = self.get_resource_colors(resources)

        cumulative_data = {}
        total_daily_gpu_hours = pd.Series(0, index=date_range)

        for i, resource in enumerate(resources):
            resource_data = daily_gpu_hours[daily_gpu_hours['Targeted Resource'] == resource]

            # Create a complete date series with zeros for missing dates
            resource_series = pd.Series(0, index=date_range)

            # Fill in actual GPU hours
            for _, row in resource_data.iterrows():
                resource_series[pd.Timestamp(row['End Date'])] = row['GPU Hours']

            # Add to total daily GPU hours
            total_daily_gpu_hours += resource_series

            # Calculate cumulative sum
            cumulative_gpu_hours = resource_series.cumsum()
            cumulative_data[resource] = cumulative_gpu_hours

            # Plot the line
            formatted_resource = self.format_resource_name(resource)
            ax.plot(cumulative_gpu_hours.index, cumulative_gpu_hours.values,
                   label=formatted_resource, color=colors[i], linewidth=2, marker='o', markersize=3, alpha=0.7)

        # Calculate and plot total cumulative GPU hours
        total_cumulative = total_daily_gpu_hours.cumsum()
        ax.plot(total_cumulative.index, total_cumulative.values,
               label='Total (All Resources)', color='black', linewidth=3.5, marker='s', markersize=5)

        # Customize the plot
        ax.set_title('Cumulative GPU Hours Utilized Over Time by Resource', fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative GPU Hours', fontsize=14, fontweight='bold')

        # Format x-axis
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add legend inside the plot area
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
        legend.get_title().set_fontweight('bold')

        # Format dates on x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(date_range)//10)))

        # Enhance plot borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('darkgray')

        plt.tight_layout()

        self.save_figure(fig, 'gpu_hours_over_time')

        # Print summary statistics
        print("\nGPU Hours Over Time Summary:")
        total_gpu_hours_by_resource = {}
        for resource in resources:
            if resource in cumulative_data:
                total_hours = cumulative_data[resource].iloc[-1]
                total_gpu_hours_by_resource[resource] = total_hours
                formatted_resource = self.format_resource_name(resource)
                print(f"{formatted_resource}: {total_hours:.1f} total GPU hours utilized")

        total_all_gpu_hours = total_cumulative.iloc[-1]
        print(f"Total across all resources: {total_all_gpu_hours:.1f} GPU hours utilized")

        # Print date range and efficiency metrics
        print(f"Time period: {min_date} to {max_date} ({(max_date - min_date).days + 1} days)")
        avg_daily_gpu_hours = total_all_gpu_hours / ((max_date - min_date).days + 1)
        print(f"Average daily GPU hours: {avg_daily_gpu_hours:.1f} GPU hours/day")

    def plot_gpu_hours_over_time_september(self):
        """Generate cumulative GPU hours utilized over time visualization for September only."""
        # Filter for GPU jobs with execution time data
        gpu_data = self.df[
            (self.df['Number of GPUs'] > 0) &
            (self.df['End Time'].notna()) &
            (self.df['Execution Duration (seconds)'].notna()) &
            (self.df['Execution Duration (seconds)'] > 0) &
            (self.df['Final Status'].isin(['completed', 'checkpointed', 'held', 'evicted']))  # Include all jobs that used GPU time
        ].copy()

        if gpu_data.empty:
            print("No GPU usage data found for time series analysis")
            return

        # Calculate GPU hours for each job (execution time * number of GPUs)
        gpu_data['GPU Hours'] = (gpu_data['Execution Duration (seconds)'] / 3600) * gpu_data['Number of GPUs']

        # Convert End Time to datetime and extract date
        gpu_data['End Date'] = gpu_data['End Time'].dt.date

        # Filter for September only (any year, but typically 2024)
        gpu_data = gpu_data[
            (gpu_data['End Time'].dt.month == 9)
        ].copy()

        if gpu_data.empty:
            print("No GPU usage data found for September")
            return

        # For each job completion, sum the GPU hours used on that day
        daily_gpu_hours = gpu_data.groupby(['End Date', 'Targeted Resource'])['GPU Hours'].sum().reset_index()

        # Get date range
        min_date = daily_gpu_hours['End Date'].min()
        max_date = daily_gpu_hours['End Date'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create complete date series for each resource
        resources = daily_gpu_hours['Targeted Resource'].unique()

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Get consistent colors for resources
        colors = self.get_resource_colors(resources)

        cumulative_data = {}
        total_daily_gpu_hours = pd.Series(0, index=date_range)

        for i, resource in enumerate(resources):
            resource_data = daily_gpu_hours[daily_gpu_hours['Targeted Resource'] == resource]

            # Create a complete date series with zeros for missing dates
            resource_series = pd.Series(0, index=date_range)

            # Fill in actual GPU hours
            for _, row in resource_data.iterrows():
                resource_series[pd.Timestamp(row['End Date'])] = row['GPU Hours']

            # Add to total daily GPU hours
            total_daily_gpu_hours += resource_series

            # Calculate cumulative sum
            cumulative_gpu_hours = resource_series.cumsum()
            cumulative_data[resource] = cumulative_gpu_hours

            # Plot the line
            formatted_resource = self.format_resource_name(resource)
            ax.plot(cumulative_gpu_hours.index, cumulative_gpu_hours.values,
                   label=formatted_resource, color=colors[i], linewidth=2, marker='o', markersize=3, alpha=0.7)

        # Calculate and plot total cumulative GPU hours
        total_cumulative = total_daily_gpu_hours.cumsum()
        ax.plot(total_cumulative.index, total_cumulative.values,
               label='Total (All Resources)', color='black', linewidth=3.5, marker='s', markersize=5)

        # Customize the plot
        ax.set_title('Cumulative GPU Hours Utilized Over Time by Resource (September)', fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative GPU Hours', fontsize=14, fontweight='bold')

        # Format x-axis
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add legend inside the plot area
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
        legend.get_title().set_fontweight('bold')

        # Format dates on x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(date_range)//10)))

        # Enhance plot borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('darkgray')

        plt.tight_layout()
        self.save_figure(fig, 'gpu_hours_over_time_september')

        # Print summary statistics
        print("\nGPU Hours Over Time Summary (September):")
        total_gpu_hours_by_resource = {}
        for resource in resources:
            if resource in cumulative_data:
                total_hours = cumulative_data[resource].iloc[-1]
                total_gpu_hours_by_resource[resource] = total_hours
                formatted_resource = self.format_resource_name(resource)
                print(f"{formatted_resource}: {total_hours:.1f} total GPU hours utilized")

        total_all_gpu_hours = total_cumulative.iloc[-1]
        print(f"Total across all resources: {total_all_gpu_hours:.1f} GPU hours utilized")

        # Print date range and efficiency metrics
        print(f"Time period: {min_date} to {max_date} ({(max_date - min_date).days + 1} days)")
        avg_daily_gpu_hours = total_all_gpu_hours / ((max_date - min_date).days + 1)
        print(f"Average daily GPU hours: {avg_daily_gpu_hours:.1f} GPU hours/day")

    def plot_transfer_input_timing(self):
        """Generate input file transfer timing analysis plots."""
        transfer_timing = self.analyze_transfer_input_timing()

        if not transfer_timing['has_data']:
            print("No transfer input timing data found for analysis")
            return

        timing_by_resource = transfer_timing['by_resource']
        raw_data = transfer_timing['raw_data']

        # Plot 1: Box plot of transfer times by resource
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Get consistent colors for resources
        resources = raw_data['Targeted Resource'].unique()
        colors = self.get_resource_colors(resources)

        # Box plot
        formatted_resources = [self.format_resource_name(r) for r in resources]
        box_data = [raw_data[raw_data['Targeted Resource'] == r]['Transfer Input Duration (minutes)'].values
                    for r in resources]

        bp = ax1.boxplot(box_data, labels=formatted_resources, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_title('Input File Transfer Duration by Resource', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Resource', fontsize=12)
        ax1.set_ylabel('Transfer Duration (minutes)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Bar plot of median transfer times
        median_times = timing_by_resource['Median (min)']
        bars = ax2.bar(range(len(median_times)), median_times.values, color=colors, alpha=0.7)
        ax2.set_title('Median Input File Transfer Duration', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Resource', fontsize=12)
        ax2.set_ylabel('Median Duration (minutes)', fontsize=12)
        ax2.set_xticks(range(len(median_times)))
        ax2.set_xticklabels([self.format_resource_name(r) for r in median_times.index], rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        self.save_figure(fig, 'transfer_input_timing_by_resource')

        # Print summary statistics
        print("\nInput File Transfer Timing Summary:")
        print(f"Total jobs with transfer timing: {transfer_timing['summary']['total_jobs']}")
        print(f"Mean transfer duration: {transfer_timing['summary']['mean_duration_min']:.1f} minutes")
        print(f"Median transfer duration: {transfer_timing['summary']['median_duration_min']:.1f} minutes")
        print(f"Total transfer time: {transfer_timing['summary']['total_transfer_time_hours']:.1f} hours")
        print("\nBy Resource:")
        print(timing_by_resource.to_string())

    def plot_transfer_input_timing_over_time(self):
        """Generate scatter plot of input file transfer duration over time by resource."""
        transfer_timing = self.analyze_transfer_input_timing()

        if not transfer_timing['has_data']:
            print("No transfer input timing data found for time series analysis")
            return

        raw_data = transfer_timing['raw_data'].copy()

        # Use End Time if available, otherwise use Start Time
        if 'End Time' in raw_data.columns and raw_data['End Time'].notna().any():
            raw_data['Event Time'] = raw_data['End Time']
        elif 'Start Time' in raw_data.columns and raw_data['Start Time'].notna().any():
            raw_data['Event Time'] = raw_data['Start Time']
        else:
            print("No time data available for transfer timing over time plot")
            return

        # Filter out rows with no time
        raw_data = raw_data[raw_data['Event Time'].notna()].copy()

        if raw_data.empty:
            print("No transfer timing data with valid times")
            return

        # For OSPool, use GLIDEIN Resource Name; for others, use Targeted Resource
        def get_site_label(row):
            if row['Targeted Resource'].lower() == 'ospool':
                glidein = row.get('GLIDEIN Resource Name', 'Unknown')
                if pd.isna(glidein) or glidein == '':
                    return 'Unknown'
                return glidein
            else:
                return row['Targeted Resource']

        def get_site_group(row):
            if row['Targeted Resource'].lower() == 'ospool':
                return 'OSPool'
            else:
                return row['Targeted Resource']

        raw_data['Site'] = raw_data.apply(get_site_label, axis=1)
        raw_data['Group'] = raw_data.apply(get_site_group, axis=1)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # Group sites by resource group and sort
        grouped_sites = {}
        for group in sorted(raw_data['Group'].unique()):
            group_sites = sorted(raw_data[raw_data['Group'] == group]['Site'].unique())
            grouped_sites[group] = group_sites

        # Flatten to get ordered list of sites
        sites = []
        for group in sorted(grouped_sites.keys()):
            sites.extend(grouped_sites[group])

        # Use highly saturated, vibrant colors
        n_sites = len(sites)

        # Define a custom palette of highly saturated, distinct colors
        vibrant_colors = [
            '#FF0000',  # Bright Red
            '#00FF00',  # Bright Lime
            '#0000FF',  # Bright Blue
            '#FFFF00',  # Bright Yellow
            '#00FFFF',  # Bright Cyan
            '#FF00FF',  # Bright Magenta
            '#FF8000',  # Bright Orange
            '#8000FF',  # Bright Purple
            '#00FF80',  # Bright Spring Green
            '#FF0099',  # Bright Deep Pink
            '#99FF00',  # Bright Yellow-Green
            '#0099FF',  # Bright Sky Blue
            '#FF6600',  # Bright Red-Orange
            '#6600FF',  # Bright Violet
            '#00FF66',  # Bright Sea Green
            '#CC0000',  # Dark Red
            '#FF9900',  # Bright Tangerine
            '#9900FF',  # Bright Electric Purple
            '#00FFCC',  # Bright Turquoise
            '#FFCC00',  # Bright Gold
        ]

        # If we have more sites than predefined colors, generate additional ones
        if n_sites <= len(vibrant_colors):
            colors = vibrant_colors[:n_sites]
        else:
            # Use tab20 for many sites
            colors = sns.color_palette("tab20", n_sites)

        # Create scatter plot for each site with grouped labels
        for i, site in enumerate(sites):
            site_data = raw_data[raw_data['Site'] == site]
            group = raw_data[raw_data['Site'] == site]['Group'].iloc[0]

            # Create label with group prefix for OSPool sites
            if group == 'OSPool':
                label = f"OSPool: {site}"
            else:
                label = site

            ax.scatter(site_data['Event Time'],
                      site_data['Transfer Input Duration (minutes)'],
                      label=label,
                      color=colors[i],
                      alpha=0.85,
                      s=80,
                      edgecolors='black',
                      linewidth=0.5)

        # Customize the plot
        ax.set_title('Input File Transfer Duration Over Time by Site', fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Transfer Duration (minutes)', fontsize=14, fontweight='bold')

        # Format x-axis
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add legend with multiple columns if many sites
        if n_sites > 10:
            ncol = 2
        else:
            ncol = 1
        legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                          fontsize=9, ncol=ncol, markerscale=1.2)
        legend.get_title().set_fontweight('bold')

        # Format dates on x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Determine appropriate date interval based on data range
        date_range = (raw_data['Event Time'].max() - raw_data['Event Time'].min()).days
        if date_range > 30:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range//10)))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range//5)))

        # Enhance plot borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('darkgray')

        plt.tight_layout()

        self.save_figure(fig, 'transfer_input_timing_over_time')

        # Print summary statistics
        print("\nInput Transfer Duration Over Time Summary:")
        for site in sites:
            site_data = raw_data[raw_data['Site'] == site]
            print(f"{site}: {len(site_data)} transfers, "
                  f"median: {site_data['Transfer Input Duration (minutes)'].median():.1f} min, "
                  f"mean: {site_data['Transfer Input Duration (minutes)'].mean():.1f} min")

    def plot_data_transfer_analysis(self):
        """Generate data transfer analysis plots."""
        transfer_analysis = self.analyze_data_transfer()

        if transfer_analysis['by_resource'].empty:
            print("No transfer data found for analysis")
            return

        # Plot 1: Data transfer by resource
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        transfer_by_resource = transfer_analysis['by_resource']

        # Total data transfer by resource
        ax1.bar(transfer_by_resource.index, transfer_by_resource['Total Transfer (GB)'],
                color=sns.color_palette("Set2", len(transfer_by_resource)))
        ax1.set_title('Total Data Transfer by Resource')
        ax1.set_xlabel('Resource')
        ax1.set_ylabel('Total Transfer (GB)')
        ax1.tick_params(axis='x', rotation=45)

        # Sent vs Received breakdown
        x = np.arange(len(transfer_by_resource.index))
        width = 0.35

        ax2.bar(x - width/2, transfer_by_resource['Data Sent (GB)'], width, label='Sent', alpha=0.8)
        ax2.bar(x + width/2, transfer_by_resource['Data Received (GB)'], width, label='Received', alpha=0.8)
        ax2.set_title('Data Sent vs Received by Resource')
        ax2.set_xlabel('Resource')
        ax2.set_ylabel('Data Transfer (GB)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(transfer_by_resource.index, rotation=45)
        ax2.legend()

        plt.tight_layout()
        self.save_figure(fig, 'data_transfer_by_resource')

        # Transfer efficiency plot removed per user request

    def generate_summary_report(self):
        """Generate a text summary report with key statistics."""
        epoch_stats = self.extract_epoch_stats()
        gpu_analysis = self.analyze_gpu_utilization()
        transfer_analysis = self.analyze_data_transfer()

        report_lines = [
            "EXPERIMENT ANALYSIS SUMMARY REPORT",
            "=" * 50,
            "",
            f"Analysis Period: {self.df['Submit Time'].min().strftime('%Y-%m-%d')} to {self.df['Submit Time'].max().strftime('%Y-%m-%d')}",
            f"Total Job Attempts: {len(self.df):,}",
            f"Unique Jobs: {self.df['Job Name'].nunique():,}",
            "",
            "RESOURCE UTILIZATION:",
            "-" * 20,
        ]

        for _, row in epoch_stats.iterrows():
            report_lines.extend([
                f"{row['Resource'].upper()}:",
                f"  • Successful Jobs: {row['Successful Jobs']:,} ({row['Success Rate']:.1%})",
                f"  • Computation Time: {row['Total Time (hours)']:,.1f} hours",
                f"  • Time Efficiency: {row['Time Efficiency']:.1%}",
                ""
            ])

        # Overall statistics
        total_jobs = epoch_stats['Total Jobs'].sum()
        total_successful = epoch_stats['Successful Jobs'].sum()
        total_time = epoch_stats['Total Time (hours)'].sum()
        total_successful_time = epoch_stats['Successful Time (hours)'].sum()

        report_lines.extend([
            "OVERALL METRICS:",
            "-" * 15,
            f"Total Success Rate: {total_successful/total_jobs:.1%}",
            f"Total Computation Time: {total_time:,.1f} hours",
            f"Overall Time Efficiency: {total_successful_time/total_time:.1%}",
            "",
        ])

        # GPU analysis
        if not gpu_analysis['efficiency'].empty:
            report_lines.extend([
                "GPU UTILIZATION:",
                "-" * 16,
                f"Total GPU Hours: {gpu_analysis['total_gpu_hours']:,.1f}",
                f"GPU Types Used: {len(gpu_analysis['efficiency'])}",
                ""
            ])

            for gpu_type, row in gpu_analysis['efficiency'].iterrows():
                report_lines.append(f"{gpu_type}: {row['Total Jobs']} jobs, {row['Success Rate']:.1%} success rate")

            report_lines.append("")

        # Data transfer analysis
        if not transfer_analysis['by_resource'].empty:
            report_lines.extend([
                "DATA TRANSFER:",
                "-" * 13,
                f"Total Data Transferred: {transfer_analysis['total_transfer_gb']:,.1f} GB",
                ""
            ])

            for resource, row in transfer_analysis['by_resource'].iterrows():
                report_lines.append(f"{resource}: {row['Total Transfer (GB)']:,.1f} GB ({row['Job Count']} jobs)")

        # Input file transfer timing analysis
        transfer_timing = self.analyze_transfer_input_timing()
        if transfer_timing['has_data']:
            report_lines.extend([
                "",
                "INPUT FILE TRANSFER TIMING:",
                "-" * 27,
                f"Jobs with Transfer Timing: {transfer_timing['summary']['total_jobs']:,}",
                f"Mean Transfer Duration: {transfer_timing['summary']['mean_duration_min']:.1f} minutes",
                f"Median Transfer Duration: {transfer_timing['summary']['median_duration_min']:.1f} minutes",
                f"Total Transfer Time: {transfer_timing['summary']['total_transfer_time_hours']:.1f} hours",
                ""
            ])

            for resource, row in transfer_timing['by_resource'].iterrows():
                report_lines.append(
                    f"{resource}: median {row['Median (min)']:.1f} min, "
                    f"mean {row['Mean (min)']:.1f} min ({int(row['Job Count'])} jobs)"
                )

        # Save report
        report_text = "\n".join(report_lines)
        report_file = self.output_dir / "experiment_summary_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)

        print(f"Summary report saved: {report_file}")
        print("\n" + report_text)

    def generate_all_reports(self):
        """Generate all analysis reports and visualizations."""
        print("\nGenerating experiment analysis reports...")

        try:
            print("\n1. Generating resource breakdown plots...")
            self.plot_resource_breakdown()

            print("\n2. Generating GPU analysis plots...")
            self.plot_gpu_analysis()

            print("\n3. Generating GPU runtime by resource analysis...")
            self.plot_gpu_runtime_by_resource()

            print("\n4. Generating GPU capability heatmap...")
            self.plot_gpu_capability_heatmap()

            print("\n5. Generating epochs completed over time plot...")
            self.plot_epochs_completed_over_time()

            print("\n5b. Generating epochs completed over time plot (September only)...")
            self.plot_epochs_completed_over_time_september()

            print("\n5c. Generating epochs trained per day plot...")
            self.plot_epochs_trained_per_day()

            print("\n6. Generating GPU hours over time plot...")
            self.plot_gpu_hours_over_time()

            print("\n6b. Generating GPU hours over time plot (September only)...")
            self.plot_gpu_hours_over_time_september()

            print("\n7. Generating data transfer analysis plots...")
            self.plot_data_transfer_analysis()

            print("\n8. Generating input file transfer timing analysis...")
            self.plot_transfer_input_timing()

            print("\n8b. Generating input file transfer timing over time plot...")
            self.plot_transfer_input_timing_over_time()

            print("\n9. Generating summary report...")
            self.generate_summary_report()

            print(f"\n✅ All reports generated successfully in {self.output_dir}/")

        except Exception as e:
            print(f"❌ Error during report generation: {e}")
            raise


def main():
    """Main entry point for the experiment report generator."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive experiment analysis reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reports with default input file
  python experiment_report.py

  # Specify custom input file
  python experiment_report.py my_experiment_data.csv

  # Specify custom output directory
  python experiment_report.py --output-dir /path/to/reports/
        """
    )

    parser.add_argument("input_file", nargs="?", default="job_summary.csv",
                       help="Input CSV file path (default: job_summary.csv)")
    parser.add_argument("--output-dir", default="reports",
                       help="Output directory for reports (default: reports)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    try:
        # Create analyzer and generate reports
        analyzer = ExperimentAnalyzer(args.input_file, args.output_dir)
        analyzer.generate_all_reports()

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Make sure {args.input_file} exists and contains valid experiment data.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
