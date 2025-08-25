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
        print(f"- Resources: {sorted(self.df['Targeted Resource'].unique())}")
        print(f"- Job statuses: {sorted(self.df['Final Status'].unique())}")
        print(f"- Epochs found: {self.df['Epoch'].min()}-{self.df['Epoch'].max()}")
    
    def extract_epoch_stats(self) -> pd.DataFrame:
        """Extract epoch completion statistics by resource, using longest execution per job."""
        # Filter for jobs with epoch information and successful status
        epoch_data = self.df[
            (self.df['Epoch'].notna()) & 
            (self.df['Is Successful']) &
            (self.df['Execution Duration (seconds)'] > 0)
        ].copy()
        
        # For each unique job (DAG + Job Name), select the attempt with longest execution time
        # This represents where the real computational work was done
        longest_executions = epoch_data.loc[
            epoch_data.groupby(['DAG Source', 'Job Name'])['Execution Duration (seconds)'].idxmax()
        ].copy()
        
        print(f"Selected {len(longest_executions)} longest executions from {len(epoch_data)} successful attempts")
        
        # Calculate summary statistics
        summary_stats = []
        for resource in longest_executions['Targeted Resource'].unique():
            resource_data = longest_executions[longest_executions['Targeted Resource'] == resource]
            
            total_jobs = len(resource_data)
            completed_epochs = len(resource_data[resource_data['Final Status'] == 'completed'])
            checkpointed_epochs = len(resource_data[resource_data['Final Status'] == 'checkpointed'])
            successful_epochs = total_jobs  # All are successful by definition
            
            total_time = resource_data['Execution Duration (seconds)'].sum()
            
            # Calculate all-time efficiency (successful time vs total wall-clock time including failed attempts)
            all_resource_data = self.df[self.df['Targeted Resource'] == resource]
            total_wall_time = all_resource_data['Total Duration (seconds)'].sum()
            
            summary_stats.append({
                'Resource': resource,
                'Total Jobs': total_jobs,
                'Completed Epochs': completed_epochs,
                'Checkpointed Epochs': checkpointed_epochs,
                'Successful Epochs': successful_epochs,
                'Success Rate': successful_epochs / total_jobs if total_jobs > 0 else 0,  # Always 1.0 by definition
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
    
    def plot_resource_breakdown(self):
        """Generate resource utilization breakdown plots."""
        epoch_stats = self.extract_epoch_stats()
        
        # Plot 1: Epoch completion by resource
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Successful epochs by resource
        ax.bar(epoch_stats['Resource'], epoch_stats['Successful Epochs'], color=sns.color_palette("husl", len(epoch_stats)))
        ax.set_title('Successful Epochs by Resource')
        ax.set_xlabel('Resource')
        ax.set_ylabel('Number of Successful Epochs')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.save_figure(fig, 'resource_epoch_breakdown')
        
        # Plot 2: Computation time and epochs breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total computation time by resource (pie chart)
        ax1.pie(epoch_stats['Total Time (hours)'], labels=epoch_stats['Resource'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Total Computation Time by Resource')
        
        # Successful epochs by resource (pie chart)
        ax2.pie(epoch_stats['Successful Epochs'], labels=epoch_stats['Resource'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Successful Epochs by Resource')
        
        plt.tight_layout()
        self.save_figure(fig, 'resource_time_breakdown')
    
    def plot_gpu_analysis(self):
        """Generate GPU utilization analysis plots."""
        gpu_analysis = self.analyze_gpu_utilization()
        
        if gpu_analysis['efficiency'].empty:
            print("No GPU data found for analysis")
            return
        
        # Plot 1: GPU type distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        gpu_counts = gpu_analysis['efficiency']['Total Jobs']
        ax.pie(gpu_counts.values, labels=gpu_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('GPU Type Distribution (by Job Count)')
        
        plt.tight_layout()
        self.save_figure(fig, 'gpu_distribution_analysis')
        
        # Plot 2: GPU efficiency metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average job time by GPU type
        gpu_avg_time = gpu_analysis['efficiency']['Avg Time'] / 3600  # Convert to hours
        bars = ax1.bar(range(len(gpu_avg_time)), gpu_avg_time.values, color=sns.color_palette("viridis", len(gpu_avg_time)))
        ax1.set_title('Average Job Duration by GPU Type')
        ax1.set_xlabel('GPU Type')
        ax1.set_ylabel('Average Duration (hours)')
        ax1.set_xticks(range(len(gpu_avg_time)))
        ax1.set_xticklabels([name.replace('NVIDIA ', '').replace('Tesla ', '') for name in gpu_avg_time.index], rotation=45, ha='right')
        
        # GPU memory distribution
        gpu_memory = gpu_analysis['efficiency']['Memory MB'] / 1024  # Convert to GB
        bars = ax2.bar(range(len(gpu_memory)), gpu_memory.values, color=sns.color_palette("plasma", len(gpu_memory)))
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
                f"  • Successful Epochs: {row['Successful Epochs']:,} ({row['Success Rate']:.1%})",
                f"  • Computation Time: {row['Total Time (hours)']:,.1f} hours",
                f"  • Time Efficiency: {row['Time Efficiency']:.1%}",
                ""
            ])
        
        # Overall statistics
        total_jobs = epoch_stats['Total Jobs'].sum()
        total_successful = epoch_stats['Successful Epochs'].sum()
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
            
            print("\n4. Generating data transfer analysis plots...")
            self.plot_data_transfer_analysis()
            
            print("\n5. Generating summary report...")
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