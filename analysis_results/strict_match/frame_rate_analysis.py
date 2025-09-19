#!/usr/bin/env python3
"""
Strict Match Frame Rate Analysis for DVRK Multimodal Data Collection

This script calculates real-time frame rate and overall frame rate for strict match data
using left image timestamps as the baseline reference.

Date: 2025
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

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StrictMatchFrameRateAnalyzer:
    """
    Analyzer for calculating frame rates in strict match data using left image timestamps as baseline.
    """
    
    def __init__(self, data_root: str = "../../data/data_new"):
        self.data_root = Path(data_root)
        self.output_dir = Path("output_frame_rate")
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset processing rules - only process strict_match data
        self.dataset_rules = {
            'data_20250909': ['strict_match/1', 'strict_match/2', 'strict_match/3', 'strict_match/4']
        }
        
        # Data containers
        self.frame_data = []
        self.frame_rate_stats = {}
        
        # Robot arms to process - only process one arm to get correct frame count
        self.robot_arms = ['PSM1']  # Only process PSM1 to get the actual frame count
        
    def load_data(self) -> None:
        """Load all JSON files and extract timestamp information."""
        print("Loading strict match data...")
        
        for dataset_folder in self.data_root.iterdir():
            if not dataset_folder.is_dir():
                continue
                
            dataset_name = dataset_folder.name
            if dataset_name not in self.dataset_rules:
                continue
                
            print(f"Processing dataset: {dataset_name}")
            
            for subfolder in self.dataset_rules[dataset_name]:
                subfolder_path = dataset_folder / subfolder
                if not subfolder_path.exists():
                    continue
                    
                self._process_dataset_subfolder(dataset_name, subfolder, subfolder_path)
        
        print(f"Loaded {len(self.frame_data)} frame data points")
    
    def _process_dataset_subfolder(self, dataset_name: str, subfolder: str, subfolder_path: Path) -> None:
        """Process a specific subfolder within a dataset."""
        regular_path = subfolder_path / "regular"
        if not regular_path.exists():
            return
            
        for arm in self.robot_arms:
            arm_path = regular_path / "kinematic" / arm
            
            if not arm_path.exists() or not any(arm_path.glob("*.json")):
                continue
                
            json_files = sorted(glob.glob(str(arm_path / "*.json")))
            
            for json_file in json_files:
                try:
                    self._process_json_file(json_file, arm, dataset_name, subfolder)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
    
    def _process_json_file(self, json_file: str, arm: str, dataset_name: str, subfolder: str) -> None:
        """Process a single JSON file and extract left image timestamp."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        header = data.get('header', {})
        left_img_data = header.get('header_img_left', {})
        
        if 'sec' not in left_img_data or 'nsec' not in left_img_data:
            return
            
        # Convert to timestamp in seconds
        left_img_timestamp = left_img_data['sec'] + left_img_data['nsec'] * 1e-9
        frame_num = int(Path(json_file).stem)
        
        # Store frame data
        self.frame_data.append({
            'frame': frame_num,
            'dataset': dataset_name,
            'subfolder': subfolder,
            'arm': arm,
            'left_img_timestamp': left_img_timestamp,
            'sec': left_img_data['sec'],
            'nsec': left_img_data['nsec']
        })
    
    def calculate_frame_rates(self) -> None:
        """Calculate per-second and overall frame rates using actual frame intervals."""
        print("Calculating frame rates...")
        
        if not self.frame_data:
            print("No data loaded.")
            return
            
        df = pd.DataFrame(self.frame_data)
        
        # Sort by timestamp to ensure proper order
        df = df.sort_values('left_img_timestamp')
        
        # Calculate frame rates for each dataset/subfolder combination
        for (dataset, subfolder), group in df.groupby(['dataset', 'subfolder']):
            group = group.sort_values('left_img_timestamp')
            
            if len(group) < 2:
                continue
                
            # Calculate per-second frame rates using actual frame intervals
            per_second_fps = self._calculate_frame_interval_fps(group)
            
            # Calculate overall frame rate
            total_duration = group['left_img_timestamp'].max() - group['left_img_timestamp'].min()
            overall_fps = (len(group) - 1) / total_duration if total_duration > 0 else 0
            
            # Store statistics
            key = f"{dataset}_{subfolder}"
            self.frame_rate_stats[key] = {
                'dataset': dataset,
                'subfolder': subfolder,
                'total_frames': len(group),
                'total_duration_seconds': total_duration,
                'overall_fps': overall_fps,
                'per_second_fps_mean': per_second_fps['fps'].mean() if len(per_second_fps) > 0 else 0,
                'per_second_fps_std': per_second_fps['fps'].std() if len(per_second_fps) > 0 else 0,
                'per_second_fps_min': per_second_fps['fps'].min() if len(per_second_fps) > 0 else 0,
                'per_second_fps_max': per_second_fps['fps'].max() if len(per_second_fps) > 0 else 0,
                'per_second_fps_median': per_second_fps['fps'].median() if len(per_second_fps) > 0 else 0,
                'intervals_count': len(per_second_fps),
                'per_second_fps_data': per_second_fps['fps'].tolist()
            }
            
            # Add per-second frame rate data for visualization
            if len(per_second_fps) > 0:
                for _, row in per_second_fps.iterrows():
                    self.frame_data.append({
                        'start_time': row['start_time'],
                        'end_time': row['end_time'],
                        'interval_duration': row['interval_duration'],
                        'dataset': dataset,
                        'subfolder': subfolder,
                        'per_second_fps': row['fps'],
                        'overall_fps': overall_fps
                    })
        
        print("Frame rate calculations completed.")
    
    def _calculate_frame_interval_fps(self, group):
        """
        Calculate frame rate by accumulating frames until 1 second is reached.
        Simple for loop: accumulate frames, when 1 second is reached, calculate FPS and continue.
        """
        if len(group) < 2:
            return pd.DataFrame()
        
        timestamps = group['left_img_timestamp'].values
        frames = group['frame'].values
        
        intervals = []
        i = 0
        
        while i < len(timestamps) - 1:
            # Start from current frame
            start_frame = frames[i]
            start_time = timestamps[i]
            frame_count = 1  # Include the starting frame
            
            # Accumulate frames until we reach 1 second
            j = i + 1
            while j < len(timestamps):
                current_time = timestamps[j]
                time_elapsed = current_time - start_time
                
                # If we've reached or exceeded 1 second, calculate FPS
                if time_elapsed >= 1.0:
                    end_frame = frames[j]
                    end_time = current_time
                    interval_duration = end_time - start_time
                    frame_count = j - i + 1  # Total frames in this interval
                    fps = frame_count / interval_duration
                    
                    intervals.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'interval_duration': interval_duration,
                        'fps': fps,
                        'frame_count': frame_count,
                        'start_frame': start_frame,
                        'end_frame': end_frame
                    })
                    
                    # Move to the next frame after the current one
                    i = j + 1
                    break
                else:
                    # Continue accumulating frames
                    j += 1
            else:
                # If we reached the end without hitting 1 second, break
                break
        
        return pd.DataFrame(intervals)
    
    def create_visualizations(self) -> None:
        """Create simple line plot of frame rate over time."""
        print("Creating frame rate over time plot...")
        
        if not self.frame_data:
            print("No frame data to visualize.")
            return
            
        df = pd.DataFrame(self.frame_data)
        
        # Filter data that has per_second_fps
        fps_data = df.dropna(subset=['per_second_fps'])
        
        if fps_data.empty:
            print("No per-second FPS data to visualize.")
            return
        
        # Create simple line plot
        self._plot_simple_fps_over_time(fps_data)
        
        print("Visualization created successfully.")
    
    def _plot_simple_fps_over_time(self, fps_data):
        """Create simple line plot of FPS over time."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot each dataset as a separate line
        datasets = fps_data['subfolder'].unique()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, subfolder in enumerate(datasets):
            subfolder_data = fps_data[fps_data['subfolder'] == subfolder]
            
            # Convert start_time to relative time (seconds from start)
            start_time = subfolder_data['start_time'].min()
            relative_time = subfolder_data['start_time'] - start_time
            
            ax.plot(relative_time, subfolder_data['per_second_fps'], 
                   label=f'Dataset {subfolder.replace("strict_match/", "")}', 
                   color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frame Rate (FPS)', fontsize=14, fontweight='bold')
        ax.set_title('Frame Rate Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fps_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_frame_rate_comparison(self) -> None:
        """Plot comparison between overall and per-second frame rates."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        datasets = list(self.frame_rate_stats.keys())
        overall_fps = [stats['overall_fps'] for stats in self.frame_rate_stats.values()]
        per_second_fps_mean = [stats['per_second_fps_mean'] for stats in self.frame_rate_stats.values()]
        per_second_fps_std = [stats['per_second_fps_std'] for stats in self.frame_rate_stats.values()]
        
        # Bar plot comparison
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, overall_fps, width, label='Overall FPS', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, per_second_fps_mean, width, label='Per-second FPS (Mean)', 
                       alpha=0.8, color='lightcoral', yerr=per_second_fps_std, capsize=5)
        
        ax1.set_xlabel('Dataset', fontweight='bold')
        ax1.set_ylabel('Frame Rate (FPS)', fontweight='bold')
        ax1.set_title('Overall vs Per-second Frame Rate Comparison', fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.replace('data_20250909_strict_match/', '') for d in datasets], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar1, bar2, overall, per_second) in enumerate(zip(bars1, bars2, overall_fps, per_second_fps_mean)):
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1,
                    f'{overall:.2f}', ha='center', va='bottom', fontsize=10)
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + per_second_fps_std[i] + 0.1,
                    f'{per_second:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Scatter plot: Overall vs Per-second FPS
        ax2.scatter(overall_fps, per_second_fps_mean, s=100, alpha=0.7, color='darkblue')
        
        # Add diagonal line for reference (perfect correlation)
        min_val = min(min(overall_fps), min(per_second_fps_mean))
        max_val = max(max(overall_fps), max(per_second_fps_mean))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Correlation')
        
        ax2.set_xlabel('Overall FPS', fontweight='bold')
        ax2.set_ylabel('Per-second FPS (Mean)', fontweight='bold')
        ax2.set_title('Overall vs Per-second FPS Correlation', fontweight='bold', pad=20)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add dataset labels to scatter points
        for i, dataset in enumerate(datasets):
            ax2.annotate(dataset.replace('data_20250909_strict_match/', ''), 
                        (overall_fps[i], per_second_fps_mean[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_second_fps_over_time(self) -> None:
        """Plot per-second frame rate over time for each dataset."""
        if not self.frame_data:
            return
            
        df = pd.DataFrame(self.frame_data)
        
        # Filter data that has per_second_fps
        fps_data = df.dropna(subset=['per_second_fps'])
        
        if fps_data.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        datasets = fps_data['subfolder'].unique()
        
        for i, subfolder in enumerate(datasets):
            if i >= 4:  # Limit to 4 subplots
                break
                
            ax = axes[i]
            subfolder_data = fps_data[fps_data['subfolder'] == subfolder]
            
            # Plot per-second FPS over time
            ax.plot(subfolder_data['start_time'], subfolder_data['per_second_fps'], 
                   alpha=0.7, linewidth=1, color='blue', marker='o', markersize=3)
            
            # Add mean line
            mean_fps = subfolder_data['per_second_fps'].mean()
            ax.axhline(y=mean_fps, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_fps:.2f} FPS')
            
            ax.set_xlabel('Time (seconds)', fontweight='bold')
            ax.set_ylabel('Per-second FPS', fontweight='bold')
            ax.set_title(f'Per-second Frame Rate - {subfolder}', fontweight='bold', pad=15)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(datasets), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_second_fps_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_second_fps_distribution(self) -> None:
        """Plot distribution of per-second frame rates."""
        if not self.frame_data:
            return
            
        df = pd.DataFrame(self.frame_data)
        fps_data = df.dropna(subset=['per_second_fps'])
        
        if fps_data.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of per-second FPS
        ax1.hist(fps_data['per_second_fps'], bins=30, alpha=0.7, color='lightgreen', 
                edgecolor='black', linewidth=0.5)
        ax1.axvline(fps_data['per_second_fps'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {fps_data["per_second_fps"].mean():.2f} FPS')
        ax1.set_xlabel('Per-second FPS', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Distribution of Per-second Frame Rates', fontweight='bold', pad=20)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by dataset
        fps_data['subfolder_short'] = fps_data['subfolder'].str.replace('strict_match/', '')
        sns.boxplot(data=fps_data, x='subfolder_short', y='per_second_fps', ax=ax2)
        ax2.set_xlabel('Dataset', fontweight='bold')
        ax2.set_ylabel('Per-second FPS', fontweight='bold')
        ax2.set_title('Per-second FPS Distribution by Dataset', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_second_fps_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_frame_rate_by_dataset(self) -> None:
        """Plot frame rate statistics by dataset."""
        if not self.frame_rate_stats:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        datasets = list(self.frame_rate_stats.keys())
        overall_fps = [stats['overall_fps'] for stats in self.frame_rate_stats.values()]
        per_second_fps_mean = [stats['per_second_fps_mean'] for stats in self.frame_rate_stats.values()]
        per_second_fps_std = [stats['per_second_fps_std'] for stats in self.frame_rate_stats.values()]
        total_frames = [stats['total_frames'] for stats in self.frame_rate_stats.values()]
        total_duration = [stats['total_duration_seconds'] for stats in self.frame_rate_stats.values()]
        
        # Create grouped bar chart
        x = np.arange(len(datasets))
        width = 0.2
        
        bars1 = ax.bar(x - 2*width, overall_fps, width, label='Overall FPS', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x - width, per_second_fps_mean, width, label='Per-second FPS (Mean)', 
                      alpha=0.8, color='lightcoral', yerr=per_second_fps_std, capsize=5)
        bars3 = ax.bar(x, total_frames, width, label='Total Frames', alpha=0.8, color='lightgreen')
        bars4 = ax.bar(x + width, total_duration, width, label='Duration (s)', alpha=0.8, color='gold')
        
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('Frame Rate Statistics by Dataset', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace('data_20250909_strict_match/', '') for d in datasets], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9, rotation=90)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_rate_by_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self) -> None:
        """Save analysis results."""
        print("Saving results...")
        
        if not self.frame_rate_stats:
            print("No frame rate data to save.")
            return
        
        # Create simple summary table
        self._create_simple_summary_table()
        
        # Save FPS over time data
        if self.frame_data:
            df = pd.DataFrame(self.frame_data)
            fps_data = df.dropna(subset=['per_second_fps'])
            if not fps_data.empty:
                # Save only the essential columns for time series
                time_series_data = fps_data[['subfolder', 'start_time', 'end_time', 'per_second_fps']].copy()
                time_series_data.to_csv(self.output_dir / 'fps_over_time_data.csv', index=False)
        
        print(f"Results saved to: {self.output_dir}")
    
    def _create_simple_summary_table(self) -> None:
        """Create a simple summary table."""
        if not self.frame_rate_stats:
            return
        
        # Create simple summary table
        summary_data = []
        
        for key, stats in self.frame_rate_stats.items():
            summary_data.append({
                'Dataset': key.replace('data_20250909_strict_match/', ''),
                'Total Frames': stats['total_frames'],
                'Duration (s)': f"{stats['total_duration_seconds']:.2f}",
                'Overall FPS': f"{stats['overall_fps']:.2f}",
                'Per-second FPS (Mean)': f"{stats['per_second_fps_mean']:.2f}",
                'Per-second FPS (Std)': f"{stats['per_second_fps_std']:.2f}",
                'Per-second FPS (Min)': f"{stats['per_second_fps_min']:.2f}",
                'Per-second FPS (Max)': f"{stats['per_second_fps_max']:.2f}",
                'Intervals Count': stats['intervals_count']
            })
        
        # Add overall summary row
        overall_stats = self._calculate_overall_stats()
        summary_data.append(overall_stats)
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'fps_summary_with_overall.csv', index=False)
    
    def _calculate_overall_stats(self) -> None:
        """Calculate overall statistics across all datasets."""
        if not self.frame_rate_stats:
            return {}
        
        # Collect all data
        total_frames = 0
        total_duration = 0
        all_per_second_fps = []
        total_intervals = 0
        
        for stats in self.frame_rate_stats.values():
            total_frames += stats['total_frames']
            total_duration += stats['total_duration_seconds']
            total_intervals += stats['intervals_count']
            
            # Collect per-second FPS data
            if 'per_second_fps_data' in stats and stats['per_second_fps_data']:
                all_per_second_fps.extend(stats['per_second_fps_data'])
        
        # Calculate overall FPS
        overall_fps = (total_frames - 1) / total_duration if total_duration > 0 else 0
        
        # Calculate per-second FPS statistics
        if all_per_second_fps:
            per_second_fps_mean = np.mean(all_per_second_fps)
            per_second_fps_std = np.std(all_per_second_fps)
            per_second_fps_min = np.min(all_per_second_fps)
            per_second_fps_max = np.max(all_per_second_fps)
        else:
            per_second_fps_mean = per_second_fps_std = per_second_fps_min = per_second_fps_max = 0
        
        return {
            'Dataset': 'Overall',
            'Total Frames': total_frames,
            'Duration (s)': f"{total_duration:.2f}",
            'Overall FPS': f"{overall_fps:.2f}",
            'Per-second FPS (Mean)': f"{per_second_fps_mean:.2f}",
            'Per-second FPS (Std)': f"{per_second_fps_std:.2f}",
            'Per-second FPS (Min)': f"{per_second_fps_min:.2f}",
            'Per-second FPS (Max)': f"{per_second_fps_max:.2f}",
            'Intervals Count': total_intervals
        }
    
    def _create_summary_table(self) -> None:
        """Create a summary table of frame rate statistics."""
        if not self.frame_rate_stats:
            return
        
        # Create summary table
        summary_data = []
        
        for key, stats in self.frame_rate_stats.items():
            summary_data.append({
                'Dataset': key.replace('data_20250909_strict_match/', ''),
                'Total Frames': stats['total_frames'],
                'Duration (s)': f"{stats['total_duration_seconds']:.2f}",
                'Overall FPS': f"{stats['overall_fps']:.2f}",
                'Per-second FPS (Mean)': f"{stats['per_second_fps_mean']:.2f}",
                'Per-second FPS (Std)': f"{stats['per_second_fps_std']:.2f}",
                'Per-second FPS (Min)': f"{stats['per_second_fps_min']:.2f}",
                'Per-second FPS (Max)': f"{stats['per_second_fps_max']:.2f}",
                'Per-second FPS (Median)': f"{stats['per_second_fps_median']:.2f}",
                'Intervals Count': stats['intervals_count']
            })
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'frame_rate_summary.csv', index=False)
        
        # Also save as LaTeX table
        latex_table = summary_df.to_latex(index=False, escape=False)
        with open(self.output_dir / 'frame_rate_summary.tex', 'w') as f:
            f.write(latex_table)
    
    def run_analysis(self) -> None:
        """Run the complete frame rate analysis."""
        print("Starting Strict Match Frame Rate Analysis...")
        print("=" * 50)
        
        try:
            self.load_data()
            self.calculate_frame_rates()
            self.create_visualizations()
            self.save_results()
            
            print("=" * 50)
            print("Analysis completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            
            # Print summary
            if self.frame_rate_stats:
                print(f"\nQuick Summary:")
                for key, stats in self.frame_rate_stats.items():
                    print(f"  {key}:")
                    print(f"    Total Frames: {stats['total_frames']}")
                    print(f"    Duration: {stats['total_duration_seconds']:.2f}s")
                    print(f"    Overall FPS: {stats['overall_fps']:.2f}")
                    print(f"    Per-second FPS (Mean): {stats['per_second_fps_mean']:.2f} ± {stats['per_second_fps_std']:.2f}")
                    print(f"    Per-second FPS (Min): {stats['per_second_fps_min']:.2f}")
                    print(f"    Per-second FPS (Max): {stats['per_second_fps_max']:.2f}")
                    print(f"    Intervals Count: {stats['intervals_count']}")
                    print()
                
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Main function to run the frame rate analysis."""
    analyzer = StrictMatchFrameRateAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
