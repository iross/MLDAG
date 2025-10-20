#!/usr/bin/env python3
"""
Standalone GPU Runtime Box Plot Generator

Creates a box plot showing execution times for each GPU device name grouped by targeted resource.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def create_gpu_runtime_boxplot(csv_file="job_summary.csv", output_file="gpu_runtime_boxplot.png"):
    """Create box plot of GPU execution times by device and resource."""
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Filter for GPU jobs with execution time data
    gpu_runtime_data = df[
        (df['Number of GPUs'] > 0) & 
        (df['GPU Device Name'].notna()) & 
        (df['GPU Device Name'] != '') &
        (df['Execution Duration (seconds)'].notna()) & 
        (df['Execution Duration (seconds)'] > 0)
    ].copy()
    
    if gpu_runtime_data.empty:
        print("No GPU runtime data found for analysis")
        return
    
    print(f"Found {len(gpu_runtime_data)} GPU jobs for analysis")
    
    # Convert execution time to hours for better readability
    gpu_runtime_data['Execution Time (hours)'] = gpu_runtime_data['Execution Duration (seconds)'] / 3600
    
    # Clean up GPU names for better display
    gpu_runtime_data['GPU Type'] = gpu_runtime_data['GPU Device Name'].str.replace('NVIDIA ', '').str.replace('Tesla ', '')
    
    # Set up plotting style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (16, 10),
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100
    })
    
    # Create figure with improved size
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # Get unique GPU types and sort them for consistent ordering
    gpu_types = sorted(gpu_runtime_data['GPU Type'].unique())
    
    # Create box plot with more spacing and better separation
    sns.boxplot(data=gpu_runtime_data, 
               x='GPU Type', 
               y='Execution Time (hours)', 
               hue='Targeted Resource',
               ax=ax,
               width=0.6)  # Make boxes narrower for better separation
    
    # Customize the plot with enhanced styling
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
        print("Using log scale due to wide range of execution times")
    
    # Add a subtle border around the plot area
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('darkgray')
    
    # Adjust layout to accommodate labels and legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend
    
    # Save the plot
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Box plot saved: {output_file}")
    
    # Print summary statistics
    print("\nGPU Runtime Summary by Device and Resource:")
    summary_stats = gpu_runtime_data.groupby(['GPU Type', 'Targeted Resource'])['Execution Time (hours)'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    print(f"{'GPU Type':<20} {'Resource':<10} {'Jobs':<5} {'Median':<8} {'Mean':<12} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 85)
    
    for (gpu_type, resource), stats in summary_stats.iterrows():
        if stats['count'] > 0:
            print(f"{gpu_type:<20} {resource:<10} {int(stats['count']):<5} {stats['median']:>6.1f}h {stats['mean']:>6.1f}±{stats['std']:<5.1f} {stats['min']:>6.1f}h {stats['max']:>6.1f}h")

def main():
    parser = argparse.ArgumentParser(description="Generate GPU runtime box plot by device and resource")
    parser.add_argument("input_file", nargs="?", default="job_summary.csv", 
                       help="Input CSV file (default: job_summary.csv)")
    parser.add_argument("-o", "--output", default="gpu_runtime_boxplot.png",
                       help="Output PNG file (default: gpu_runtime_boxplot.png)")
    
    args = parser.parse_args()
    
    try:
        create_gpu_runtime_boxplot(args.input_file, args.output)
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()