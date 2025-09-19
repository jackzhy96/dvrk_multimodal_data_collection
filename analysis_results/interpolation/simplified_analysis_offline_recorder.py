#!/usr/bin/env python3
"""
Simplified DVRK Timestamp Delay Analysis for Interpolation Data
Focus on essential metrics and clear visualizations for PhD thesis
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

class SimplifiedInterpolationAnalyzer:
    """Simplified analyzer for interpolation data focused on key metrics for PhD thesis."""
    
    def __init__(self, data_root: str = "../../data/data_new"):
        self.data_root = Path(data_root)
        self.output_dir = Path("output_simplified")
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset processing rules for interpolation data - only process data_20250909
        self.dataset_rules = {
            'data_20250909': ['interpolation/1', 'interpolation/2', 'interpolation/3', 'interpolation/4']
        }
        
        # Data containers
        self.delay_data = []
        self.summary_stats = {}
        
        # Sensor categorization for interpolation data
        self.sensor_categories = {
            'image': [],
            'kinematics': [
                'measured_cp_stamp', 'measured_cv_stamp', 'measured_js_stamp',
                'setpoint_cp_stamp', 'setpoint_js_stamp'
            ],
            'jaw': [
                'jaw_measured_js_stamp', 'jaw_setpoint_js_stamp'
            ]
        }
        
        # Setpoint/Measure categorization for three-group analysis
        self.setpoint_measure_categories = {
            'setpoint': ['setpoint_cp_stamp', 'setpoint_js_stamp', 'jaw_setpoint_js_stamp'],
            'measured': ['measured_cp_stamp', 'measured_cv_stamp', 'measured_js_stamp', 'jaw_measured_js_stamp'],
            'measured+img': ['measured_cp_stamp', 'measured_cv_stamp', 'measured_js_stamp', 'jaw_measured_js_stamp']
        }
        
        # Data type categorization
        self.data_type_categories = {
            'image': [],
            'joint_states': ['measured_js_stamp', 'setpoint_js_stamp', 'jaw_measured_js_stamp', 'jaw_setpoint_js_stamp'],
            'cartesian_states': ['measured_cp_stamp', 'measured_cv_stamp', 'setpoint_cp_stamp']
        }
        
        self.robot_arms = ['ECM', 'PSM1', 'PSM2']
        
    def load_data(self) -> None:
        """Load all JSON files and extract timestamp delay information."""
        print("Loading interpolation data...")
        
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
        
        print(f"Loaded {len(self.delay_data)} delay measurements")
    
    def _process_dataset_subfolder(self, dataset_name: str, subfolder: str, subfolder_path: Path) -> None:
        """Process a specific subfolder within a dataset."""
        print(f"  Processing subfolder: {subfolder}")
        regular_path = subfolder_path / "regular"
        if not regular_path.exists():
            print(f"    Warning: No regular folder found in {subfolder_path}")
            return
            
        # Load baseline timestamps from time_syn folder
        time_syn_path = regular_path / "time_syn"
        if not time_syn_path.exists():
            print(f"    Warning: No time_syn folder found in {subfolder_path}")
            return
            
        print(f"    Loading baseline timestamps from {time_syn_path}")
        baseline_timestamps = self._load_baseline_timestamps(time_syn_path)
        if not baseline_timestamps:
            print(f"    Warning: No baseline timestamps found in {time_syn_path}")
            return
            
        print(f"    Found {len(baseline_timestamps)} baseline timestamps")
            
        # Process each robot arm
        for arm in self.robot_arms:
            arm_path = regular_path / "kinematic" / arm
            
            if not arm_path.exists() or not any(arm_path.glob("*.json")):
                print(f"    Skipping {arm} - no data found")
                continue
                
            json_files = sorted(glob.glob(str(arm_path / "*.json")))
            print(f"    Processing {arm} data ({len(json_files)} files)...")
            
            processed_count = 0
            for json_file in json_files:
                try:
                    self._process_json_file(json_file, arm, dataset_name, subfolder, baseline_timestamps)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
            
            print(f"    Processed {processed_count} files for {arm}")
    
    def _load_baseline_timestamps(self, time_syn_path: Path) -> Dict[int, float]:
        """Load baseline timestamps from time_syn folder."""
        baseline_timestamps = {}
        
        for json_file in sorted(glob.glob(str(time_syn_path / "*.json"))):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                frame_num = int(Path(json_file).stem)
                
                # Extract image_stamp_left as baseline
                if 'image_stamp_left' in data and isinstance(data['image_stamp_left'], dict):
                    if 'sec' in data['image_stamp_left'] and 'nsec' in data['image_stamp_left']:
                        timestamp = data['image_stamp_left']['sec'] + data['image_stamp_left']['nsec'] * 1e-9
                        baseline_timestamps[frame_num] = timestamp
                        
            except Exception as e:
                print(f"Error loading baseline timestamp from {json_file}: {e}")
                continue
                
        return baseline_timestamps
    
    def _process_json_file(self, json_file: str, arm: str, dataset_name: str, subfolder: str, baseline_timestamps: Dict[int, float]) -> None:
        """Process a single JSON file and extract timestamp delays."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        frame_num = int(Path(json_file).stem)
        
        # Get baseline timestamp for this frame
        if frame_num not in baseline_timestamps:
            return
            
        baseline_timestamp = baseline_timestamps[frame_num]
        
        # Process each candidate data point (interpolation data is a direct array)
        if not isinstance(data, list):
            if len(self.delay_data) < 5:  # Only print for first few files
                print(f"      Data is not a list in {json_file}")
            return
            
        delays_added = 0
        for candidate_idx, candidate_data in enumerate(data):
            if not isinstance(candidate_data, dict):
                continue
                
            # Extract all timestamps from this candidate
            timestamps = self._extract_all_timestamps(candidate_data, arm)
            
            # Debug: print first few files
            if len(self.delay_data) < 5:
                print(f"      Frame {frame_num}, Candidate {candidate_idx}: Found {len([t for t in timestamps.values() if t is not None])} valid timestamps")
            
            # Calculate delays for each sensor
            for sensor_key, sensor_timestamp in timestamps.items():
                if sensor_timestamp is None:
                    continue
                    
                # Calculate delay: baseline_timestamp - sensor_timestamp (consistent with strict_match)
                delay_ms = (baseline_timestamp - sensor_timestamp) * 1000
                
                # Determine categories
                category = self._get_sensor_category(sensor_key)
                setpoint_measure_category = self._get_setpoint_measure_category(sensor_key)
                data_type_category = self._get_data_type_category(sensor_key)
                
                self.delay_data.append({
                    'frame': frame_num,
                    'candidate': candidate_idx,
                    'dataset': dataset_name,
                    'subfolder': subfolder,
                    'arm': arm,
                    'sensor': sensor_key,
                    'category': category,
                    'setpoint_measure_category': setpoint_measure_category,
                    'data_type_category': data_type_category,
                    'delay_ms': delay_ms,
                    'abs_delay_ms': abs(delay_ms)
                })
                delays_added += 1
        
        
        # Debug: print first few files
        if delays_added > 0 and len(self.delay_data) <= 10:
            print(f"      Added {delays_added} delays from {json_file}")
    
    def _extract_all_timestamps(self, candidate_data: Dict, arm: str) -> Dict[str, float]:
        """Extract all timestamps from a candidate data point."""
        timestamps = {}

        # Extract from arm data
        arm_data = candidate_data.get('arm', {})

        # Measured data timestamps
        measured_data = arm_data.get('measured_data', {})
        for key in ['measured_cp_stamp', 'measured_cv_stamp', 'measured_js_stamp']:
            if key in measured_data and isinstance(measured_data[key], dict):
                if 'sec' in measured_data[key] and 'nsec' in measured_data[key]:
                    timestamps[key] = measured_data[key]['sec'] + measured_data[key]['nsec'] * 1e-9
                else:
                    timestamps[key] = None
            else:
                timestamps[key] = None

        # Setpoint data timestamps
        setpoint_data = arm_data.get('setpoint_data', {})
        for key in ['setpoint_cp_stamp', 'setpoint_js_stamp']:
            if key in setpoint_data and isinstance(setpoint_data[key], dict):
                if 'sec' in setpoint_data[key] and 'nsec' in setpoint_data[key]:
                    timestamps[key] = setpoint_data[key]['sec'] + setpoint_data[key]['nsec'] * 1e-9
                else:
                    timestamps[key] = None
            else:
                timestamps[key] = None

        # Jaw data timestamps (only for PSM1 and PSM2)
        # Note: jaw data is at the top level, not inside arm_data
        if arm in ['PSM1', 'PSM2']:
            jaw_data = candidate_data.get('jaw', {})
            
            # Process jaw measured data
            jaw_measured = jaw_data.get('measured_data', {})
            if 'stamp' in jaw_measured and isinstance(jaw_measured['stamp'], dict):
                if 'sec' in jaw_measured['stamp'] and 'nsec' in jaw_measured['stamp']:
                    timestamps['jaw_measured_js_stamp'] = jaw_measured['stamp']['sec'] + jaw_measured['stamp']['nsec'] * 1e-9
                else:
                    timestamps['jaw_measured_js_stamp'] = None
            else:
                timestamps['jaw_measured_js_stamp'] = None
            
            # Process jaw setpoint data
            jaw_setpoint = jaw_data.get('setpoint_data', {})
            if 'stamp' in jaw_setpoint and isinstance(jaw_setpoint['stamp'], dict):
                if 'sec' in jaw_setpoint['stamp'] and 'nsec' in jaw_setpoint['stamp']:
                    timestamps['jaw_setpoint_js_stamp'] = jaw_setpoint['stamp']['sec'] + jaw_setpoint['stamp']['nsec'] * 1e-9
                else:
                    timestamps['jaw_setpoint_js_stamp'] = None
            else:
                timestamps['jaw_setpoint_js_stamp'] = None

        # Header timestamps - header_cv removed from processing
        # header = candidate_data.get('header', {})
        # for key in ['header_cv']:
        #     if key in header and isinstance(header[key], dict):
        #         if 'sec' in header[key] and 'nsec' in header[key]:
        #             timestamps[key] = header[key]['sec'] + header[key]['nsec'] * 1e-9
        #         else:
        #             timestamps[key] = None
        #     else:
        #         timestamps[key] = None

        return timestamps
    
    def _get_sensor_category(self, sensor_key: str) -> str:
        """Determine the category of a sensor based on its name."""
        for category, sensors in self.sensor_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def _get_setpoint_measure_category(self, sensor_key: str) -> str:
        """Determine the setpoint/measure category of a sensor based on its name."""
        for category, sensors in self.setpoint_measure_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def _get_data_type_category(self, sensor_key: str) -> str:
        """Determine the data type category of a sensor based on its name."""
        for category, sensors in self.data_type_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def calculate_statistics(self) -> None:
        """Calculate key statistics for thesis."""
        print("Calculating statistics...")
        
        if not self.delay_data:
            print("No data loaded.")
            return
            
        df = pd.DataFrame(self.delay_data)
        
        # Overall statistics
        self.summary_stats['overall'] = {
            'count': len(df),
            'mean_delay_ms': df['abs_delay_ms'].mean(),
            'std_delay_ms': df['abs_delay_ms'].std(),
            'min_delay_ms': df['delay_ms'].min(),
            'max_delay_ms': df['delay_ms'].max(),
            'median_delay_ms': df['delay_ms'].median(),
            'abs_median_delay_ms': df['abs_delay_ms'].median(),
            'q95_delay_ms': df['delay_ms'].quantile(0.95)
        }
        
        # Statistics by sensor category
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            self.summary_stats[f'category_{category}'] = {
                'count': len(category_data),
                'mean_delay_ms': category_data['abs_delay_ms'].mean(),
                'std_delay_ms': category_data['abs_delay_ms'].std(),
                'min_delay_ms': category_data['delay_ms'].min(),
                'max_delay_ms': category_data['delay_ms'].max(),
                'median_delay_ms': category_data['delay_ms'].median(),
                'abs_median_delay_ms': category_data['abs_delay_ms'].median(),
                'q95_delay_ms': category_data['delay_ms'].quantile(0.95)
            }
        
        # Statistics by setpoint/measure category
        for setpoint_measure_category in df['setpoint_measure_category'].unique():
            setpoint_measure_data = df[df['setpoint_measure_category'] == setpoint_measure_category]
            self.summary_stats[f'setpoint_measure_{setpoint_measure_category}'] = {
                'count': len(setpoint_measure_data),
                'mean_delay_ms': setpoint_measure_data['abs_delay_ms'].mean(),
                'std_delay_ms': setpoint_measure_data['abs_delay_ms'].std(),
                'min_delay_ms': setpoint_measure_data['delay_ms'].min(),
                'max_delay_ms': setpoint_measure_data['delay_ms'].max(),
                'median_delay_ms': setpoint_measure_data['delay_ms'].median(),
                'abs_median_delay_ms': setpoint_measure_data['abs_delay_ms'].median(),
                'q95_delay_ms': setpoint_measure_data['delay_ms'].quantile(0.95)
            }
        
        # Statistics by data type category
        for data_type_category in df['data_type_category'].unique():
            data_type_data = df[df['data_type_category'] == data_type_category]
            self.summary_stats[f'data_type_{data_type_category}'] = {
                'count': len(data_type_data),
                'mean_delay_ms': data_type_data['abs_delay_ms'].mean(),
                'std_delay_ms': data_type_data['abs_delay_ms'].std(),
                'min_delay_ms': data_type_data['delay_ms'].min(),
                'max_delay_ms': data_type_data['delay_ms'].max(),
                'median_delay_ms': data_type_data['delay_ms'].median(),
                'abs_median_delay_ms': data_type_data['abs_delay_ms'].median(),
                'q95_delay_ms': data_type_data['delay_ms'].quantile(0.95)
            }
        
        # Statistics by robot arm
        for arm in df['arm'].unique():
            arm_data = df[df['arm'] == arm]
            self.summary_stats[f'arm_{arm}'] = {
                'count': len(arm_data),
                'mean_delay_ms': arm_data['abs_delay_ms'].mean(),
                'std_delay_ms': arm_data['abs_delay_ms'].std(),
                'min_delay_ms': arm_data['delay_ms'].min(),
                'max_delay_ms': arm_data['delay_ms'].max(),
                'median_delay_ms': arm_data['delay_ms'].median(),
                'abs_median_delay_ms': arm_data['abs_delay_ms'].median(),
                'q95_delay_ms': arm_data['delay_ms'].quantile(0.95)
            }
        
        print("Statistics calculated successfully.")
    
    def create_visualizations(self) -> None:
        """Create essential visualizations for thesis."""
        print("Creating visualizations...")
        
        if not self.delay_data:
            print("No data loaded.")
            return
            
        df = pd.DataFrame(self.delay_data)
        
        # Set up plotting style for publication
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        # 1. Overall delay distribution
        self._plot_overall_distribution(df)
        
        # 2. Three-group comparison (measured, setpoint, header)
        self._plot_three_group_comparison(df)
        
        # 3. Data type comparison (joint_states, cartesian_states)
        self._plot_data_type_comparison(df)
        
        # 4. Robot arm comparison
        self._plot_robot_arm_comparison(df)
        
        # 5. Combined comparison
        self._plot_combined_comparison(df)
        
        # 6. Four-group histogram (overall, measured, setpoint, header)
        self._plot_four_group_histogram(df)
        
        # 7. Four-group histogram (overall, cartesian_states, joint_states, header)
        self._plot_four_group_histogram_data_types(df)
        
        print("Visualizations created successfully.")
    
    def _plot_overall_distribution(self, df: pd.DataFrame) -> None:
        """Plot overall delay distribution with stacked histograms."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Stacked histogram for different groups - use raw delay (with sign)
        all_data = df['delay_ms']
        measured_data = df[df['setpoint_measure_category'] == 'measured']['delay_ms']
        setpoint_data = df[df['setpoint_measure_category'] == 'setpoint']['delay_ms']
        header_data = df[df['setpoint_measure_category'] == 'header']['delay_ms']
        
        # Define bins - symmetric around 0
        max_abs_delay = df['delay_ms'].abs().max()
        bins = np.linspace(-max_abs_delay, max_abs_delay, 50)
        
        # Plot stacked histogram
        ax1.hist([all_data, measured_data, setpoint_data, header_data], 
                bins=bins, alpha=0.7, stacked=True, 
                label=['All', 'Measured', 'Setpoint', 'Header'],
                color=['skyblue', 'lightcoral', 'lightgreen', 'gold'],
                edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Timestamp Delay (ms)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=20, fontweight='bold')
        ax1.set_title('Distribution of Timestamp Delays by Group', fontsize=22, fontweight='bold', pad=20)
        ax1.legend(fontsize=22, prop={'family': 'DejaVu Sans', 'size': 22})
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot by data type category
        sns.boxplot(data=df, x='data_type_category', y='abs_delay_ms', ax=ax2)
        ax2.set_xlabel('Data Type Category', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Absolute Timestamp Delay (ms)', fontsize=20, fontweight='bold')
        ax2.set_title('Delay Distribution by Data Type Category', fontsize=22, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_three_group_comparison(self, df: pd.DataFrame) -> None:
        """Plot three-group comparison (measured, setpoint, header)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot with error bars
        group_stats = df.groupby('setpoint_measure_category')['abs_delay_ms'].agg(['mean', 'std', 'count']).reset_index()
        
        colors = ['lightcoral', 'lightgreen', 'gold']
        bars = ax1.bar(group_stats['setpoint_measure_category'], group_stats['mean'], 
                      yerr=group_stats['std'], capsize=5, alpha=0.8, color=colors)
        
        ax1.set_xlabel('Data Group', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Mean Absolute Delay (ms)', fontsize=16, fontweight='bold')
        ax1.set_title('Timestamp Delay by Data Group', fontsize=18, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, group_stats['mean'], group_stats['std'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.1,
                    f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom')
        
        # Box plot for detailed distribution
        sns.boxplot(data=df, x='setpoint_measure_category', y='abs_delay_ms', ax=ax2)
        ax2.set_xlabel('Data Group', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Absolute Timestamp Delay (ms)', fontsize=16, fontweight='bold')
        ax2.set_title('Delay Distribution by Data Group', fontsize=18, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'three_group_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_data_type_comparison(self, df: pd.DataFrame) -> None:
        """Plot data type category comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot with error bars
        category_stats = df.groupby('data_type_category')['abs_delay_ms'].agg(['mean', 'std', 'count']).reset_index()
        
        bars = ax.bar(category_stats['data_type_category'], category_stats['mean'], 
                     yerr=category_stats['std'], capsize=5, alpha=0.7, 
                     color=['skyblue', 'lightcoral'])
        
        ax.set_xlabel('Data Type Category', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Absolute Delay (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Timestamp Delay by Data Type Category', fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, category_stats['mean'], category_stats['std'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.1,
                   f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robot_arm_comparison(self, df: pd.DataFrame) -> None:
        """Plot robot arm comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot with error bars
        arm_stats = df.groupby('arm')['abs_delay_ms'].agg(['mean', 'std', 'count']).reset_index()
        
        bars = ax.bar(arm_stats['arm'], arm_stats['mean'], 
                     yerr=arm_stats['std'], capsize=5, alpha=0.7,
                     color=['gold', 'lightblue', 'lightgreen', 'lightpink'])
        
        ax.set_xlabel('Robot Arm', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Absolute Delay (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Timestamp Delay by Robot Arm', fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, arm_stats['mean'], arm_stats['std'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.1,
                   f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robot_arm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_comparison(self, df: pd.DataFrame) -> None:
        """Plot combined comparison (data_type x arm)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar plot
        pivot_data = df.groupby(['data_type_category', 'arm'])['abs_delay_ms'].mean().unstack()
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
        
        ax.set_xlabel('Data Type Category', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Absolute Delay (ms)', fontsize=16, fontweight='bold')
        ax.set_title('Timestamp Delay: Data Type vs Robot Arm', fontsize=18, fontweight='bold', pad=20)
        ax.legend(title='Robot Arm', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, title_fontsize=20,
                 prop={'family': 'DejaVu Sans', 'size': 18})
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_four_group_histogram(self, df: pd.DataFrame) -> None:
        """Create side-by-side histograms for four data groups."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data - use raw delay (not absolute) for visualization
        all_data = df['delay_ms']
        measured_data = df[df['setpoint_measure_category'] == 'measured']['delay_ms']
        setpoint_data = df[df['setpoint_measure_category'] == 'setpoint']['delay_ms']
        header_data = df[df['setpoint_measure_category'] == 'header']['delay_ms']
        
        # Define bins - symmetric around 0 to show both positive and negative delays
        max_abs_delay = df['delay_ms'].abs().max()
        bins = np.linspace(-max_abs_delay, max_abs_delay, 40)
        
        # Data and labels for each histogram
        datasets = [
            (all_data, 'Overall', '#3B4992'),
            (measured_data, 'Measured', '#A20056'),
            (setpoint_data, 'Setpoint', '#008280'),
            (header_data, 'Header', '#631879')
        ]
        
        # Mean line colors
        mean_colors = {
            'Overall': '#4DBBD5',
            'Measured': '#E64B35',
            'Setpoint': '#00A087',
            'Header': '#3C5488'
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
        ax.set_xlabel('Timestamp Delay (ms)', fontsize=22, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=22, fontweight='bold')
        ax.set_title('Distribution of Timestamp Delays by Data Group', 
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
        stats_text += f'Mean Absolute Delay: {df["abs_delay_ms"].mean():.2f} ms\n'
        stats_text += f'Std Absolute Delay: {df["abs_delay_ms"].std():.2f} ms\n'
        stats_text += f'95th Percentile Delay: {df["delay_ms"].quantile(0.95):.2f} ms\n'
        stats_text += f'Range: [{df["delay_ms"].min():.2f}, {df["delay_ms"].max():.2f}] ms'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=18, 
                fontfamily='DejaVu Sans')
        
        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        # Set x-axis limits to actual data range
        ax.set_xlim(df['delay_ms'].min(), df['delay_ms'].max())
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'four_group_histogram.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_four_group_histogram_data_types(self, df: pd.DataFrame) -> None:
        """Create side-by-side histograms for four data groups (Overall, Cartesian States, Joint States, Header)."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data - use raw delay (not absolute) for visualization
        all_data = df['delay_ms']
        cartesian_states_data = df[df['data_type_category'] == 'cartesian_states']['delay_ms']
        joint_states_data = df[df['data_type_category'] == 'joint_states']['delay_ms']
        header_data = df[df['setpoint_measure_category'] == 'header']['delay_ms']
        
        # Define bins - symmetric around 0 to show both positive and negative delays
        max_abs_delay = df['delay_ms'].abs().max()
        bins = np.linspace(-max_abs_delay, max_abs_delay, 40)
        
        # Data and labels for each histogram
        datasets = [
            (all_data, 'Overall', '#3B4992'),
            (cartesian_states_data, 'Cartesian States', '#A20056'),
            (joint_states_data, 'Joint States', '#008280'),
            (header_data, 'Header', '#631879')
        ]
        
        # Mean line colors
        mean_colors = {
            'Overall': '#4DBBD5',
            'Cartesian States': '#E64B35',
            'Joint States': '#00A087',
            'Header': '#3C5488'
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
        ax.set_xlabel('Timestamp Delay (ms)', fontsize=22, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=22, fontweight='bold')
        ax.set_title('Distribution of Timestamp Delays by Data Type Group', 
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
        stats_text += f'Mean Absolute Delay: {df["abs_delay_ms"].mean():.2f} ms\n'
        stats_text += f'Std Absolute Delay: {df["abs_delay_ms"].std():.2f} ms\n'
        stats_text += f'95th Percentile Delay: {df["delay_ms"].quantile(0.95):.2f} ms\n'
        stats_text += f'Range: [{df["delay_ms"].min():.2f}, {df["delay_ms"].max():.2f}] ms'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=18, 
                fontfamily='DejaVu Sans')
        
        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        # Set x-axis limits to actual data range
        ax.set_xlim(df['delay_ms'].min(), df['delay_ms'].max())
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'four_group_histogram_data_types.pdf', bbox_inches='tight')
        plt.close()
    
    def save_results(self) -> None:
        """Save analysis results."""
        print("Saving results...")
        
        if not self.delay_data:
            print("No data to save.")
            return
            
        df = pd.DataFrame(self.delay_data)
        
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
        if not self.delay_data:
            return
            
        df = pd.DataFrame(self.delay_data)
        
        # Create summary table
        summary_data = []
        
        # Overall statistics
        overall = self.summary_stats['overall']
        summary_data.append({
            'Category': 'Overall',
            'Count': overall['count'],
            'Mean (ms)': f"{overall['mean_delay_ms']:.2f}",
            'Std (ms)': f"{overall['std_delay_ms']:.2f}",
            'Median (ms)': f"{overall['median_delay_ms']:.2f}",
            'Abs Median (ms)': f"{overall['abs_median_delay_ms']:.2f}",
            '95th Percentile (ms)': f"{overall['q95_delay_ms']:.2f}"
        })
        
        # By data type category
        for data_type_category in df['data_type_category'].unique():
            if f'data_type_{data_type_category}' in self.summary_stats:
                cat_stats = self.summary_stats[f'data_type_{data_type_category}']
                summary_data.append({
                    'Category': f'Data Type: {data_type_category}',
                    'Count': cat_stats['count'],
                    'Mean (ms)': f"{cat_stats['mean_delay_ms']:.2f}",
                    'Std (ms)': f"{cat_stats['std_delay_ms']:.2f}",
                    'Median (ms)': f"{cat_stats['median_delay_ms']:.2f}",
                    'Abs Median (ms)': f"{cat_stats['abs_median_delay_ms']:.2f}",
                    '95th Percentile (ms)': f"{cat_stats['q95_delay_ms']:.2f}"
                })
        
        # By setpoint/measure category
        for setpoint_measure_category in ['measured', 'setpoint', 'header']:
            if f'setpoint_measure_{setpoint_measure_category}' in self.summary_stats:
                sm_stats = self.summary_stats[f'setpoint_measure_{setpoint_measure_category}']
                summary_data.append({
                    'Category': f'Group: {setpoint_measure_category}',
                    'Count': sm_stats['count'],
                    'Mean (ms)': f"{sm_stats['mean_delay_ms']:.2f}",
                    'Std (ms)': f"{sm_stats['std_delay_ms']:.2f}",
                    'Median (ms)': f"{sm_stats['median_delay_ms']:.2f}",
                    'Abs Median (ms)': f"{sm_stats['abs_median_delay_ms']:.2f}",
                    '95th Percentile (ms)': f"{sm_stats['q95_delay_ms']:.2f}"
                })
        
        # By robot arm
        for arm in df['arm'].unique():
            if f'arm_{arm}' in self.summary_stats:
                arm_stats = self.summary_stats[f'arm_{arm}']
                summary_data.append({
                    'Category': f'Arm: {arm}',
                    'Count': arm_stats['count'],
                    'Mean (ms)': f"{arm_stats['mean_delay_ms']:.2f}",
                    'Std (ms)': f"{arm_stats['std_delay_ms']:.2f}",
                    'Median (ms)': f"{arm_stats['median_delay_ms']:.2f}",
                    'Abs Median (ms)': f"{arm_stats['abs_median_delay_ms']:.2f}",
                    '95th Percentile (ms)': f"{arm_stats['q95_delay_ms']:.2f}"
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
        print("Starting Simplified DVRK Interpolation Timestamp Delay Analysis...")
        print("=" * 60)
        
        try:
            self.load_data()
            self.calculate_statistics()
            self.create_visualizations()
            self.save_results()
            
            print("=" * 60)
            print("Analysis completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            
            # Print summary
            if self.delay_data:
                df = pd.DataFrame(self.delay_data)
                print(f"\nQuick Summary:")
                print(f"  Total data points: {len(df)}")
                print(f"  Mean delay: {df['abs_delay_ms'].mean():.2f} ms")
                print(f"  Std deviation: {df['abs_delay_ms'].std():.2f} ms")
                print(f"  95th percentile: {df['delay_ms'].quantile(0.95):.2f} ms")
                
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Main function to run the simplified analysis."""
    analyzer = SimplifiedInterpolationAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
