#!/usr/bin/env python3
"""
Modified Histogram Analysis for DVRK Data Collection
Based on the transcript requirements for simplified histogram visualization

This script creates modified histograms and comparison tables for:
1. Strict Match (Online) vs Interpolation (Offline) recorders
2. Simplified categories: Overall + specific categories only
3. Comparison table with mean(std) format
4. Additional attributes like frequency, post-processing requirements, etc.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure matplotlib for better font rendering
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 28
plt.rcParams['axes.unicode_minus'] = False

class ModifiedHistogramAnalyzer:
    """Modified analyzer for creating simplified histograms and comparison tables."""
    
    def __init__(self, strict_match_dir: str = "strict_match/output", 
                 interpolation_dir: str = "interpolation/output"):
        self.strict_match_dir = Path(strict_match_dir)
        self.interpolation_dir = Path(interpolation_dir)
        self.output_dir = Path("modified_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.strict_match_data = []
        self.interpolation_data = []
        self.comparison_stats = {}
        
    def load_data(self) -> None:
        """Load data from both strict_match and interpolation analyses."""
        print("Loading data from both analysis types...")
        
        # Load strict match data
        self._load_strict_match_data()
        
        # Load interpolation data  
        self._load_interpolation_data()
        
        print(f"Loaded {len(self.strict_match_data)} strict match data points")
        print(f"Loaded {len(self.interpolation_data)} interpolation data points")
    
    def _load_strict_match_data(self) -> None:
        """Load strict match data from CSV files."""
        # Only load overall dataset to match original analysis
        overall_csv = self.strict_match_dir / "overall" / "detailed_offset_data.csv"
        
        if overall_csv.exists():
            try:
                df = pd.read_csv(overall_csv)
                if 'offset_ms' in df.columns:
                    # Convert offset_ms to delay_ms for consistency
                    df['delay_ms'] = df['offset_ms']
                    df['abs_delay_ms'] = df['delay_ms'].abs()
                    df['recorder_type'] = 'strict_match'
                    self.strict_match_data.append(df)
                    print(f"  Loaded strict match data from: {overall_csv}")
            except Exception as e:
                print(f"Error loading {overall_csv}: {e}")
        else:
            print(f"  Warning: Overall strict match data not found at {overall_csv}")
    
    def _load_interpolation_data(self) -> None:
        """Load interpolation data from CSV files using all available data."""
        # Only load overall dataset to match original analysis
        overall_csv = self.interpolation_dir / "overall" / "detailed_delay_data.csv"
        
        if overall_csv.exists():
            try:
                df = pd.read_csv(overall_csv)
                if 'delay_ms' in df.columns:
                    # Use all data without sampling
                    df['abs_delay_ms'] = df['delay_ms'].abs()
                    df['recorder_type'] = 'interpolation'
                    self.interpolation_data.append(df)
                    print(f"  Loaded interpolation data from: {overall_csv} ({len(df)} points)")
            except Exception as e:
                print(f"Error loading {overall_csv}: {e}")
        else:
            print(f"  Warning: Overall interpolation data not found at {overall_csv}")
    
    
    def create_modified_histograms(self) -> None:
        """Create modified histograms with simplified categories."""
        print("Creating modified histograms...")
        
        if not self.strict_match_data and not self.interpolation_data:
            print("No data loaded.")
            return
        
        # Combine all data
        all_data = []
        if self.strict_match_data:
            all_data.extend(self.strict_match_data)
        if self.interpolation_data:
            all_data.extend(self.interpolation_data)
        
        if not all_data:
            print("No data to plot.")
            return
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create subplot layout: 2 rows, 1 column
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: Strict Match (Online) - Overall only
        self._plot_simplified_histogram(combined_df, 'strict_match', ax1, "Online-Matching Recorder")
        
        # Plot 2: Interpolation (Offline) - Overall only  
        self._plot_simplified_histogram(combined_df, 'interpolation', ax2, "Offline-Matching Recorder")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'modified_histograms.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'modified_histograms.pdf', bbox_inches='tight')
        plt.close()
        
        print("Modified histograms saved.")
    
    def _plot_simplified_histogram(self, df: pd.DataFrame, recorder_type: str, ax, title: str) -> None:
        """Plot simplified histogram for a specific recorder type."""
        # Filter data for this recorder type
        data = df[df['recorder_type'] == recorder_type]['delay_ms']
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f'No data available for {recorder_type}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=28)
            ax.set_title(title, fontsize=34, fontweight='bold')
            return
        
        # Define bins - symmetric around 0
        max_abs_delay = data.abs().max()
        bins = np.linspace(-max_abs_delay, max_abs_delay, 40)
        
        # Plot histogram
        ax.hist(data, bins=bins, alpha=0.7, color='skyblue', 
               edgecolor='black', linewidth=0.5)
        

        
        # Customize plot
        ax.set_xlabel('Timestamp Delay (ms)', fontsize=32, fontweight='normal')
        ax.set_ylabel('Frequency', fontsize=32, fontweight='normal')
        ax.set_title(title, fontsize=42, fontweight='bold', pad=20)
        ax.legend(fontsize=30)
        
        # Make tick labels larger and more frequent
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.locator_params(axis='x', nbins=10)  # Increase x-axis tick density
        ax.locator_params(axis='y', nbins=8)   # Increase y-axis tick density
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box (median removed as requested)
        stats_text = f'Count: {len(data):,}\n'
        stats_text += f'Mean: {data.abs().mean():.2f} ms\n'
        stats_text += f'Std: {data.abs().std():.2f} ms\n'
        stats_text += f'Range: [{data.min():.2f}, {data.max():.2f}] ms'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=24)
    
    def create_comparison_table(self) -> None:
        """Create comparison table between the two recorder types."""
        print("Creating comparison table...")
        
        if not self.strict_match_data and not self.interpolation_data:
            print("No data for comparison table.")
            return
        
        # Prepare data for comparison
        comparison_data = []
        
        # Strict Match (Online) data
        if self.strict_match_data:
            strict_df = pd.concat(self.strict_match_data, ignore_index=True)
            strict_stats = self._calculate_recorder_stats(strict_df, 'Online-Matching')
            comparison_data.append(strict_stats)
        
        # Interpolation (Offline) data
        if self.interpolation_data:
            interp_df = pd.concat(self.interpolation_data, ignore_index=True)
            interp_stats = self._calculate_recorder_stats(interp_df, 'Offline-Matching')
            comparison_data.append(interp_stats)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save as CSV with error handling
        try:
            comparison_df.to_csv(self.output_dir / 'recorder_comparison_table.csv', index=False)
        except PermissionError:
            print("Warning: Could not write CSV file (file may be open in Excel). Trying alternative filename...")
            comparison_df.to_csv(self.output_dir / 'recorder_comparison_table_backup.csv', index=False)
        
        # Save as LaTeX table with error handling
        try:
            latex_table = comparison_df.to_latex(index=False, escape=False)
            with open(self.output_dir / 'recorder_comparison_table.tex', 'w') as f:
                f.write(latex_table)
        except PermissionError:
            print("Warning: Could not write LaTeX file. Skipping...")
        
        print("Comparison table saved.")
        print("\nComparison Table:")
        print(comparison_df.to_string(index=False))
    
    def _calculate_recorder_stats(self, df: pd.DataFrame, recorder_name: str) -> Dict:
        """Calculate statistics for a specific recorder type."""
        delay_data = df['delay_ms']
        abs_delay_data = df['abs_delay_ms']
        
        # Determine recorder type based on data characteristics
        if 'online' in recorder_name.lower() or 'strict_match' in recorder_name.lower():
            recorder_type = 'Online-Matching'
            frequency = '2-3 Hz'
            post_processing = 'No'
            ready_to_use = 'Yes'
            data_size = 'Larger (uncompressed images)'
        else:
            recorder_type = 'Offline-Matching'
            frequency = '10-15 Hz'
            post_processing = 'Yes (interpolation required)'
            ready_to_use = 'No'
            data_size = 'Smaller (compressed data)'
        
        return {
            'Recorder Type': recorder_type,
            'Count': len(df),
            'Mean (ms)': f"{abs_delay_data.mean():.2f}",
            'Std (ms)': f"{abs_delay_data.std():.2f}",
            'Mean±Std (ms)': f"{abs_delay_data.mean():.2f}±{abs_delay_data.std():.2f}",
            'Median (ms)': f"{delay_data.median():.2f}",
            '95th Percentile (ms)': f"{delay_data.quantile(0.95):.2f}",
            'Frequency': frequency,
            'Post Collection Processing': post_processing,
            'Ready to Use': ready_to_use,
            'Data Size': data_size,
            'Min Delay (ms)': f"{delay_data.min():.2f}",
            'Max Delay (ms)': f"{delay_data.max():.2f}"
        }
    
    def create_detailed_comparison_plots(self) -> None:
        """Create detailed comparison plots between the two methods."""
        print("Creating detailed comparison plots...")
        
        if not self.strict_match_data and not self.interpolation_data:
            print("No data for detailed comparison.")
            return
        
        # Combine all data
        all_data = []
        if self.strict_match_data:
            all_data.extend(self.strict_match_data)
        if self.interpolation_data:
            all_data.extend(self.interpolation_data)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Side-by-side histograms
        self._plot_side_by_side_histograms(combined_df, axes[0, 0])
        
        # Plot 2: Box plot comparison
        self._plot_box_comparison(combined_df, axes[0, 1])
        
        # Plot 3: Cumulative distribution
        self._plot_cumulative_distribution(combined_df, axes[1, 0])
        
        # Plot 4: Statistical comparison
        self._plot_statistical_comparison(combined_df, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'detailed_comparison_plots.pdf', bbox_inches='tight')
        plt.close()
        
        print("Detailed comparison plots saved.")
    
    def _plot_side_by_side_histograms(self, df: pd.DataFrame, ax) -> None:
        """Plot side-by-side histograms for both recorder types."""
        strict_data = df[df['recorder_type'] == 'strict_match']['delay_ms']
        interp_data = df[df['recorder_type'] == 'interpolation']['delay_ms']
        
        # Define bins
        all_data = df['delay_ms']
        max_abs_delay = all_data.abs().max()
        bins = np.linspace(-max_abs_delay, max_abs_delay, 40)
        
        # Plot histograms
        if len(strict_data) > 0:
            ax.hist(strict_data, bins=bins, alpha=0.6, label='Online-Matching', 
                   color='skyblue', edgecolor='black', linewidth=0.5)
        
        if len(interp_data) > 0:
            ax.hist(interp_data, bins=bins, alpha=0.6, label='Offline-Matching', 
                   color='lightcoral', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Timestamp Delay (ms)', fontsize=30, fontweight='normal')
        ax.set_ylabel('Frequency', fontsize=30, fontweight='normal')
        ax.set_title('Delay Distribution Comparison', fontsize=38, fontweight='bold')
        ax.legend(fontsize=28)
        
        # Make tick labels larger and more frequent
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.locator_params(axis='x', nbins=10)  # Increase x-axis tick density
        ax.locator_params(axis='y', nbins=8)   # Increase y-axis tick density
        ax.grid(True, alpha=0.3)
    
    def _plot_box_comparison(self, df: pd.DataFrame, ax) -> None:
        """Plot box plot comparison."""
        sns.boxplot(data=df, x='recorder_type', y='abs_delay_ms', ax=ax)
        ax.set_xlabel('Recorder Type', fontsize=30, fontweight='normal')
        ax.set_ylabel('Absolute Delay (ms)', fontsize=30, fontweight='normal')
        ax.set_title('Delay Distribution by Recorder Type', fontsize=38, fontweight='bold')
        
        # Make tick labels larger and more frequent
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.locator_params(axis='y', nbins=8)   # Increase y-axis tick density
        ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_distribution(self, df: pd.DataFrame, ax) -> None:
        """Plot cumulative distribution comparison."""
        strict_data = df[df['recorder_type'] == 'strict_match']['abs_delay_ms']
        interp_data = df[df['recorder_type'] == 'interpolation']['abs_delay_ms']
        
        if len(strict_data) > 0:
            ax.hist(strict_data, bins=50, alpha=0.6, cumulative=True, density=True, 
                   label='Online-Matching', color='skyblue')
        
        if len(interp_data) > 0:
            ax.hist(interp_data, bins=50, alpha=0.6, cumulative=True, density=True, 
                   label='Offline-Matching', color='lightcoral')
        
        ax.set_xlabel('Absolute Delay (ms)', fontsize=30, fontweight='normal')
        ax.set_ylabel('Cumulative Probability', fontsize=30, fontweight='normal')
        ax.set_title('Cumulative Distribution Comparison', fontsize=38, fontweight='bold')
        ax.legend(fontsize=28)
        
        # Make tick labels larger and more frequent
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.locator_params(axis='x', nbins=10)  # Increase x-axis tick density
        ax.locator_params(axis='y', nbins=8)   # Increase y-axis tick density
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_comparison(self, df: pd.DataFrame, ax) -> None:
        """Plot statistical comparison bar chart."""
        stats_data = []
        
        for recorder_type in df['recorder_type'].unique():
            data = df[df['recorder_type'] == recorder_type]['abs_delay_ms']
            stats_data.append({
                'Recorder Type': recorder_type,
                'Mean': data.mean(),
                'Std': data.std(),
                'Median': data.median()
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        x = np.arange(len(stats_df))
        width = 0.25
        
        ax.bar(x - width, stats_df['Mean'], width, label='Mean', alpha=0.8, color='skyblue')
        ax.bar(x, stats_df['Median'], width, label='Median', alpha=0.8, color='lightcoral')
        ax.bar(x + width, stats_df['Std'], width, label='Std Dev', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Recorder Type', fontsize=30, fontweight='normal')
        ax.set_ylabel('Delay (ms)', fontsize=30, fontweight='normal')
        ax.set_title('Statistical Comparison', fontsize=38, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['Recorder Type'])
        ax.legend(fontsize=28)
        
        # Make tick labels larger and more frequent
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.locator_params(axis='y', nbins=8)   # Increase y-axis tick density
        ax.grid(True, alpha=0.3)
    
    def run_analysis(self) -> None:
        """Run the complete modified analysis."""
        print("Starting Modified Histogram Analysis...")
        print("=" * 50)
        
        try:
            self.load_data()
            self.create_modified_histograms()
            self.create_comparison_table()
            self.create_detailed_comparison_plots()
            
            print("=" * 50)
            print("Modified analysis completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Main function to run the modified analysis."""
    analyzer = ModifiedHistogramAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
