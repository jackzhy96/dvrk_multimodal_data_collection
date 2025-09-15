#!/usr/bin/env python3
"""
Simplified DVRK Timestamp Analysis for PhD Thesis
Focus on essential metrics and clear visualizations
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure matplotlib for better font rendering
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

class SimplifiedTimestampAnalyzer:
    """Simplified analyzer focused on key metrics for PhD thesis."""
    
    def __init__(self, data_root: str = "../../data/data_new"):
        self.data_root = Path(data_root)
        self.output_dir = Path("output_simplified")
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset processing rules
        self.dataset_rules = {
            'data_20250908': ['2'],
            'data_20250909': ['strict_match/1', 'strict_match/2', 'strict_match/3', 'strict_match/4'],
            'data_20250911': ['suturing/strict_match/1', 'dissection/1']
        }
        
        # Data containers
        self.offset_data = []
        self.summary_stats = {}
        
        # Modality categorization for PhD thesis
        self.sensor_categories = {
            'image': ['header_img_left', 'header_img_right', 'header_img_side'],
            'joint_states': ['header_js_set', 'header_js_meas'],
            'cartesian_states': ['header_cp_set', 'header_cv', 'header_lcp', 'header_measure_cp']
        }
        
        # Setpoint/Measure categorization for three-group analysis
        self.setpoint_measure_categories = {
            'setpoint': ['header_cp_set', 'header_js_set'],
            'measured': ['header_measure_cp', 'header_cv', 'header_lcp'],
            'measured+img': ['header_measure_cp', 'header_cv', 'header_lcp', 
                            'header_img_left', 'header_img_right', 'header_img_side']
        }
        
        self.robot_arms = ['ECM', 'PSM1', 'PSM2', 'PSM3']
        
    def load_data(self) -> None:
        """Load all JSON files and extract timestamp information."""
        print("Loading data...")
        
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
        
        print(f"Loaded {len(self.offset_data)} data points")
    
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
        """Process a single JSON file and extract timestamp offsets."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        header = data.get('header', {})
        baseline_data = header.get('header_js_meas', {})
        
        if 'sec' not in baseline_data or 'nsec' not in baseline_data:
            return
            
        baseline_timestamp = baseline_data['sec'] + baseline_data['nsec'] * 1e-9
        frame_num = int(Path(json_file).stem)
        
        # Calculate offsets for each modality timestamp
        for sensor_key, sensor_data in header.items():
            if sensor_key == 'header_js_meas':
                continue
            
            # Skip jaw modalities completely (temporarily disabled)
            if sensor_key in ['header_jaw_meas', 'header_jaw_set']:
                continue
                
            if isinstance(sensor_data, dict) and 'sec' in sensor_data and 'nsec' in sensor_data:
                sensor_timestamp = sensor_data['sec'] + sensor_data['nsec'] * 1e-9
                offset_ms = (baseline_timestamp - sensor_timestamp) * 1000
                
                category = self._get_sensor_category(sensor_key)
                setpoint_measure_category = self._get_setpoint_measure_category(sensor_key)
                
                self.offset_data.append({
                    'frame': frame_num,
                    'dataset': dataset_name,
                    'subfolder': subfolder,
                    'arm': arm,
                    'sensor': sensor_key,
                    'category': category,
                    'setpoint_measure_category': setpoint_measure_category,
                    'offset_ms': offset_ms,
                    'abs_offset_ms': abs(offset_ms)
                })
    
    def _get_sensor_category(self, sensor_key: str) -> str:
        """Determine the category of a modality based on its name."""
        for category, sensors in self.sensor_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def _get_setpoint_measure_category(self, sensor_key: str) -> str:
        """Determine the setpoint/measure category of a modality based on its name."""
        for category, sensors in self.setpoint_measure_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def calculate_statistics(self) -> None:
        """Calculate key statistics for thesis."""
        print("Calculating statistics...")
        
        if not self.offset_data:
            print("No data loaded.")
            return
            
        df = pd.DataFrame(self.offset_data)
        
        # Overall statistics
        self.summary_stats['overall'] = {
            'count': len(df),
            'mean_offset_ms': df['abs_offset_ms'].mean(),
            'std_offset_ms': df['abs_offset_ms'].std(),
            'median_offset_ms': df['abs_offset_ms'].median(),
            'q95_offset_ms': df['abs_offset_ms'].quantile(0.95),
            'max_offset_ms': df['abs_offset_ms'].max()
        }
        
        # Statistics by modality category
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            self.summary_stats[f'category_{category}'] = {
                'count': len(category_data),
                'mean_offset_ms': category_data['abs_offset_ms'].mean(),
                'std_offset_ms': category_data['abs_offset_ms'].std(),
                'median_offset_ms': category_data['abs_offset_ms'].median(),
                'q95_offset_ms': category_data['abs_offset_ms'].quantile(0.95)
            }
        
        # Statistics by setpoint/measure category (img, measured, setpoint)
        for setpoint_measure_category in df['setpoint_measure_category'].unique():
            setpoint_measure_data = df[df['setpoint_measure_category'] == setpoint_measure_category]
            self.summary_stats[f'setpoint_measure_{setpoint_measure_category}'] = {
                'count': len(setpoint_measure_data),
                'mean_offset_ms': setpoint_measure_data['abs_offset_ms'].mean(),
                'std_offset_ms': setpoint_measure_data['abs_offset_ms'].std(),
                'median_offset_ms': setpoint_measure_data['abs_offset_ms'].median(),
                'q95_offset_ms': setpoint_measure_data['abs_offset_ms'].quantile(0.95)
            }
        
        # Statistics by robot arm
        for arm in df['arm'].unique():
            arm_data = df[df['arm'] == arm]
            self.summary_stats[f'arm_{arm}'] = {
                'count': len(arm_data),
                'mean_offset_ms': arm_data['abs_offset_ms'].mean(),
                'std_offset_ms': arm_data['abs_offset_ms'].std(),
                'median_offset_ms': arm_data['abs_offset_ms'].median(),
                'q95_offset_ms': arm_data['abs_offset_ms'].quantile(0.95)
            }
        
        print("Statistics calculated successfully.")
    
    def create_visualizations(self) -> None:
        """Create essential visualizations for thesis."""
        print("Creating visualizations...")
        
        if not self.offset_data:
            print("No data loaded.")
            return
            
        df = pd.DataFrame(self.offset_data)
        
        # Set up plotting style for publication
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        # 1. Overall offset distribution
        self._plot_overall_distribution(df)
        
        # 2. Three-group comparison (img, measured, setpoint)
        self._plot_three_group_comparison(df)
        
        # 3. Robot arm comparison
        self._plot_robot_arm_comparison(df)
        
        # 4. Combined comparison
        self._plot_combined_comparison(df)
        
        # 5. Four-group stacked histogram
        self._plot_four_group_histogram(df)
        
        # 6. Four-group box plot (backup)
        self._plot_four_group_boxplot(df)
        
        # 7. Four-group histogram with overall mean
        self._plot_four_group_histogram_overall_mean(df)
        
        print("Visualizations created successfully.")
    
    def _plot_overall_distribution(self, df: pd.DataFrame) -> None:
        """Plot overall offset distribution with stacked histograms."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Stacked histogram for different groups - use raw offset (with sign)
        all_data = df['offset_ms']
        image_data = df[df['category'] == 'image']['offset_ms']
        measured_data = df[df['setpoint_measure_category'] == 'measured']['offset_ms']
        setpoint_data = df[df['setpoint_measure_category'] == 'setpoint']['offset_ms']
        
        # Define bins - symmetric around 0
        max_abs_offset = df['offset_ms'].abs().max()
        bins = np.linspace(-max_abs_offset, max_abs_offset, 50)
        
        # Plot stacked histogram
        ax1.hist([all_data, image_data, measured_data, setpoint_data], 
                bins=bins, alpha=0.7, stacked=True, 
                label=['All', 'Image', 'Measured', 'Setpoint'],
                color=['skyblue', 'lightcoral', 'lightgreen', 'gold'],
                edgecolor='black', linewidth=0.5)
        
        # Removed 0ms vertical line as requested
        
        ax1.set_xlabel('Timestamp Offset (ms)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=20, fontweight='bold')
        ax1.set_title('Distribution of Timestamp Offsets by Group', fontsize=22, fontweight='bold', pad=20)
        ax1.legend(fontsize=22, prop={'family': 'DejaVu Sans', 'size': 22})
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot by category - still use absolute values for box plot
        sns.boxplot(data=df, x='category', y='abs_offset_ms', ax=ax2)
        ax2.set_xlabel('Modality Category', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Absolute Timestamp Offset (ms)', fontsize=20, fontweight='bold')
        ax2.set_title('Offset Distribution by Modality Category', fontsize=22, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_three_group_comparison(self, df: pd.DataFrame) -> None:
        """Plot three-group comparison (img, measured, setpoint)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot with error bars
        group_stats = df.groupby('setpoint_measure_category')['abs_offset_ms'].agg(['mean', 'std', 'count']).reset_index()
        
        # Filter to only show measured, setpoint, measured+img
        group_stats = group_stats[group_stats['setpoint_measure_category'].isin(['measured', 'setpoint', 'measured+img'])]
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        bars = ax1.bar(group_stats['setpoint_measure_category'], group_stats['mean'], 
                      yerr=group_stats['std'], capsize=5, alpha=0.8, color=colors)
        
        ax1.set_xlabel('Modality Group', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Mean Absolute Offset (ms)', fontsize=16, fontweight='bold')
        ax1.set_title('Timestamp Offset by Modality Group', fontsize=18, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, group_stats['mean'], group_stats['std'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.1,
                    f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom')
        
        # Box plot for detailed distribution
        sns.boxplot(data=df[df['setpoint_measure_category'].isin(['measured', 'setpoint', 'measured+img'])], 
                   x='setpoint_measure_category', y='abs_offset_ms', ax=ax2)
        ax2.set_xlabel('Modality Group', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Absolute Timestamp Offset (ms)', fontsize=16, fontweight='bold')
        ax2.set_title('Offset Distribution by Modality Group', fontsize=18, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'three_group_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensor_category_comparison(self, df: pd.DataFrame) -> None:
        """Plot modality category comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot with error bars
        category_stats = df.groupby('category')['abs_offset_ms'].agg(['mean', 'std', 'count']).reset_index()
        
        bars = ax.bar(category_stats['category'], category_stats['mean'], 
                     yerr=category_stats['std'], capsize=5, alpha=0.7, 
                     color=['skyblue', 'lightcoral', 'lightgreen'])
        
        ax.set_xlabel('Modality Category', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Absolute Offset (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Timestamp Offset by Modality Category', fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, category_stats['mean'], category_stats['std'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.1,
                   f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensor_category_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robot_arm_comparison(self, df: pd.DataFrame) -> None:
        """Plot robot arm comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot with error bars
        arm_stats = df.groupby('arm')['abs_offset_ms'].agg(['mean', 'std', 'count']).reset_index()
        
        bars = ax.bar(arm_stats['arm'], arm_stats['mean'], 
                     yerr=arm_stats['std'], capsize=5, alpha=0.7,
                     color=['gold', 'lightblue', 'lightgreen', 'lightpink'])
        
        ax.set_xlabel('Robot Arm', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Absolute Offset (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Timestamp Offset by Robot Arm', fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, arm_stats['mean'], arm_stats['std'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.1,
                   f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robot_arm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_comparison(self, df: pd.DataFrame) -> None:
        """Plot combined comparison (category x arm)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar plot
        pivot_data = df.groupby(['category', 'arm'])['abs_offset_ms'].mean().unstack()
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
        
        ax.set_xlabel('Modality Category', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Absolute Offset (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Timestamp Offset: Modality Category vs Robot Arm', fontsize=18, fontweight='bold', pad=20)
        ax.legend(title='Robot Arm', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, title_fontsize=20,
                 prop={'family': 'DejaVu Sans', 'size': 18})
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_four_group_histogram(self, df: pd.DataFrame) -> None:
        """Create side-by-side histograms on the same plot for four modality groups."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data - use raw offset (not absolute) for visualization
        all_data = df['offset_ms']
        image_data = df[df['category'] == 'image']['offset_ms']
        joint_states_data = df[df['category'] == 'joint_states']['offset_ms']
        cartesian_states_data = df[df['category'] == 'cartesian_states']['offset_ms']
        
        # Define bins - symmetric around 0 to show both positive and negative offsets
        max_abs_offset = df['offset_ms'].abs().max()
        bins = np.linspace(-max_abs_offset, max_abs_offset, 40)
        
        # Data and labels for each histogram
        datasets = [
            (all_data, 'Overall', '#3B4992'),
            (image_data, 'Image', '#A20056'),
            (joint_states_data, 'Joint States', '#008280'),
            (cartesian_states_data, 'Cartesian States', '#631879')
        ]
        
        # Mean line colors
        mean_colors = {
            'Overall': '#4DBBD5',
            'Image': '#E64B35',
            'Joint States': '#00A087',
            'Cartesian States': '#3C5488'
        }
        
        # Plot each histogram on the same axes
        for data, label, color in datasets:
            if len(data) > 0:
                ax.hist(data, bins=bins, alpha=0.6, 
                       color=color, edgecolor='black', linewidth=0.5)
                
                # Add vertical line for mean value of each group
                mean_val = data.mean()
                mean_color = mean_colors[label]
                ax.axvline(mean_val, color=mean_color, linestyle='--', linewidth=2, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Timestamp Offset (ms)', fontsize=22, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=22, fontweight='bold')
        ax.set_title('Distribution of Timestamp Offsets by Modality Group', 
                    fontsize=24, fontweight='bold', pad=20)
        
        # Add custom legend with smaller font for histogram labels
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # Create custom legend elements
        legend_elements = []
        for data, label, color in datasets:
            if len(data) > 0:
                mean_val = data.mean()
                mean_color = mean_colors[label]
                # Histogram bar
                legend_elements.append(Patch(facecolor=color, alpha=0.6, 
                                           label=f'{label} (n={len(data):,})'))
                # Mean line
                legend_elements.append(Line2D([0], [0], color=mean_color, linestyle='--', linewidth=2,
                                            label=f'{label} Mean ({mean_val:.2f} ms)'))
        
        # Create legend with smaller font for histogram labels
        legend = ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
                          prop={'family': 'DejaVu Sans', 'size': 18})
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        stats_text = f'Total Data Points: {len(df):,}\n'
        stats_text += f'Mean Absolute Offset: {df["abs_offset_ms"].mean():.2f} ms\n'
        stats_text += f'Std Absolute Offset: {df["abs_offset_ms"].std():.2f} ms\n'
        stats_text += f'95th Percentile Absolute Offset: {df["abs_offset_ms"].quantile(0.95):.2f} ms\n'
        stats_text += f'Range: [{df["offset_ms"].min():.2f}, {df["offset_ms"].max():.2f}] ms'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=18, 
                fontfamily='DejaVu Sans')
        
        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        # Set x-axis limits to actual data range
        ax.set_xlim(df['offset_ms'].min(), df['offset_ms'].max())
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'four_group_histogram.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_four_group_boxplot(self, df: pd.DataFrame) -> None:
        """Create box plot for four modality groups as backup visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for box plot
        plot_data = []
        labels = []
        colors = []
        
        # Add each group
        groups = [
            (df[df['category'] == 'image'], 'Image', '#A20056'),
            (df[df['category'] == 'joint_states'], 'Joint States', '#008280'),
            (df[df['category'] == 'cartesian_states'], 'Cartesian States', '#631879'),
            (df, 'Overall', '#3B4992')
        ]
        
        for group_df, label, color in groups:
            if len(group_df) > 0:
                plot_data.append(group_df['offset_ms'])
                labels.append(f'{label}\n(n={len(group_df):,})')
                colors.append(color)
        
        # Create box plot
        box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                             showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.6))
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Removed 0ms horizontal line as requested
        
        # Customize plot
        ax.set_ylabel('Timestamp Offset (ms)', fontsize=22, fontweight='bold')
        ax.set_xlabel('Modality Group', fontsize=22, fontweight='bold')
        ax.set_title('Distribution of Timestamp Offsets by Modality Group (Box Plot)', 
                    fontsize=24, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=24, framealpha=0.9, 
                 prop={'family': 'DejaVu Sans', 'size': 24})
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        stats_text = f'Total Data Points: {len(df):,}\n'
        stats_text += f'Mean Absolute Offset: {df["abs_offset_ms"].mean():.2f} ms\n'
        stats_text += f'Std Absolute Offset: {df["abs_offset_ms"].std():.2f} ms\n'
        stats_text += f'95th Percentile Absolute Offset: {df["abs_offset_ms"].quantile(0.95):.2f} ms\n'
        stats_text += f'Range: [{df["offset_ms"].min():.2f}, {df["offset_ms"].max():.2f}] ms'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=18, 
                fontfamily='DejaVu Sans')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'four_group_boxplot.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_four_group_histogram_overall_mean(self, df: pd.DataFrame) -> None:
        """Create histogram with overall mean line for four modality groups comparison."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data - use raw offset (not absolute) for visualization
        all_data = df['offset_ms']
        image_data = df[df['category'] == 'image']['offset_ms']
        joint_states_data = df[df['category'] == 'joint_states']['offset_ms']
        cartesian_states_data = df[df['category'] == 'cartesian_states']['offset_ms']
        
        # Define bins - symmetric around 0 to show both positive and negative offsets
        max_abs_offset = df['offset_ms'].abs().max()
        bins = np.linspace(-max_abs_offset, max_abs_offset, 40)
        
        # Data and labels for each histogram
        datasets = [
            (all_data, 'Overall', '#3B4992'),
            (image_data, 'Image', '#A20056'),
            (joint_states_data, 'Joint States', '#008280'),
            (cartesian_states_data, 'Cartesian States', '#631879')
        ]
        
        # Plot each histogram on the same axes
        for data, label, color in datasets:
            if len(data) > 0:
                ax.hist(data, bins=bins, alpha=0.6, label=f'{label} (n={len(data):,})', 
                       color=color, edgecolor='black', linewidth=0.5)
        
        # Add vertical line for overall mean value
        overall_mean = df['offset_ms'].mean()
        ax.axvline(overall_mean, color='red', linestyle='--', linewidth=3, alpha=0.9, 
                  label=f'Overall Mean ({overall_mean:.2f} ms)')
        
        # Customize plot
        ax.set_xlabel('Timestamp Offset (ms)', fontsize=22, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=22, fontweight='bold')
        ax.set_title('Distribution of Timestamp Offsets by Modality Group (Overall Mean)', 
                    fontsize=24, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=24, framealpha=0.9, 
                 prop={'family': 'DejaVu Sans', 'size': 24})
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        stats_text = f'Total Data Points: {len(df):,}\n'
        stats_text += f'Overall Mean: {overall_mean:.2f} ms\n'
        stats_text += f'Mean Absolute Offset: {df["abs_offset_ms"].mean():.2f} ms\n'
        stats_text += f'Std Absolute Offset: {df["abs_offset_ms"].std():.2f} ms\n'
        stats_text += f'95th Percentile Absolute Offset: {df["abs_offset_ms"].quantile(0.95):.2f} ms\n'
        stats_text += f'Range: [{df["offset_ms"].min():.2f}, {df["offset_ms"].max():.2f}] ms'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=18, 
                fontfamily='DejaVu Sans')
        
        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        # Set x-axis limits to actual data range
        ax.set_xlim(df['offset_ms'].min(), df['offset_ms'].max())
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'four_group_histogram_overall_mean.pdf', bbox_inches='tight')
        plt.close()
    
    def save_results(self) -> None:
        """Save analysis results."""
        print("Saving results...")
        
        if not self.offset_data:
            print("No data to save.")
            return
            
        df = pd.DataFrame(self.offset_data)
        
        # Save detailed data
        df.to_csv(self.output_dir / 'detailed_data.csv', index=False)
        
        # Create summary table
        self._create_summary_table()
        
        # Save statistics
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(self.summary_stats, f, indent=2)
        
        print(f"Results saved to: {self.output_dir}")
    
    def _create_summary_table(self) -> None:
        """Create a summary table for thesis."""
        if not self.offset_data:
            return
            
        df = pd.DataFrame(self.offset_data)
        
        # Create summary table
        summary_data = []
        
        # Overall statistics
        overall = self.summary_stats['overall']
        summary_data.append({
            'Category': 'Overall',
            'Count': overall['count'],
            'Mean (ms)': f"{overall['mean_offset_ms']:.2f}",
            'Std (ms)': f"{overall['std_offset_ms']:.2f}",
            'Median (ms)': f"{overall['median_offset_ms']:.2f}",
            '95th Percentile (ms)': f"{overall['q95_offset_ms']:.2f}"
        })
        
        # By sensor category
        for category in df['category'].unique():
            if f'category_{category}' in self.summary_stats:
                cat_stats = self.summary_stats[f'category_{category}']
                summary_data.append({
                    'Category': f'Modality: {category}',
                    'Count': cat_stats['count'],
                    'Mean (ms)': f"{cat_stats['mean_offset_ms']:.2f}",
                    'Std (ms)': f"{cat_stats['std_offset_ms']:.2f}",
                    'Median (ms)': f"{cat_stats['median_offset_ms']:.2f}",
                    '95th Percentile (ms)': f"{cat_stats['q95_offset_ms']:.2f}"
                })
        
        # By setpoint/measure category (measured, setpoint, measured+img)
        for setpoint_measure_category in ['measured', 'setpoint', 'measured+img']:
            if f'setpoint_measure_{setpoint_measure_category}' in self.summary_stats:
                sm_stats = self.summary_stats[f'setpoint_measure_{setpoint_measure_category}']
                summary_data.append({
                    'Category': f'Group: {setpoint_measure_category}',
                    'Count': sm_stats['count'],
                    'Mean (ms)': f"{sm_stats['mean_offset_ms']:.2f}",
                    'Std (ms)': f"{sm_stats['std_offset_ms']:.2f}",
                    'Median (ms)': f"{sm_stats['median_offset_ms']:.2f}",
                    '95th Percentile (ms)': f"{sm_stats['q95_offset_ms']:.2f}"
                })
        
        # By robot arm
        for arm in df['arm'].unique():
            if f'arm_{arm}' in self.summary_stats:
                arm_stats = self.summary_stats[f'arm_{arm}']
                summary_data.append({
                    'Category': f'Arm: {arm}',
                    'Count': arm_stats['count'],
                    'Mean (ms)': f"{arm_stats['mean_offset_ms']:.2f}",
                    'Std (ms)': f"{arm_stats['std_offset_ms']:.2f}",
                    'Median (ms)': f"{arm_stats['median_offset_ms']:.2f}",
                    '95th Percentile (ms)': f"{arm_stats['q95_offset_ms']:.2f}"
                })
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        # Also save as LaTeX table
        latex_table = summary_df.to_latex(index=False, escape=False)
        with open(self.output_dir / 'summary_table.tex', 'w') as f:
            f.write(latex_table)
    
    def run_analysis(self) -> None:
        """Run the complete simplified analysis."""
        print("Starting Simplified DVRK Timestamp Analysis...")
        print("=" * 50)
        
        try:
            self.load_data()
            self.calculate_statistics()
            self.create_visualizations()
            self.save_results()
            
            print("=" * 50)
            print("Analysis completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            
            # Print summary
            if self.offset_data:
                df = pd.DataFrame(self.offset_data)
                print(f"\nQuick Summary:")
                print(f"  Total data points: {len(df)}")
                print(f"  Mean offset: {df['abs_offset_ms'].mean():.2f} ms")
                print(f"  Std deviation: {df['abs_offset_ms'].std():.2f} ms")
                print(f"  95th percentile: {df['abs_offset_ms'].quantile(0.95):.2f} ms")
                
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Main function to run the simplified analysis."""
    analyzer = SimplifiedTimestampAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
