#!/usr/bin/env python3
"""
Timestamp Offset Analysis for DVRK Multimodal Data Collection

This script analyzes timestamp offsets in synchronized multimodal data.
Each JSON file represents a frame with timestamps from different sensors.
The main timestamp serves as the reference, and we calculate offsets
for each sensor timestamp relative to this reference.

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

class TimestampAnalyzer:
    """Main class for analyzing timestamp offsets in DVRK data."""
    
    def __init__(self, data_root: str = "../../data/data_new"):
        """
        Initialize the analyzer.
        
        Args:
            data_root: Root directory containing the data_new folder
        """
        self.data_root = Path(data_root)
        self.base_output_dir = Path("output")
        self.plots_dir = self.base_output_dir / "plots"
        
        # Dataset processing rules
        self.dataset_rules = {
            'data_20250908': ['2'],  # Only process subfolder 2
            'data_20250909': ['strict_match/1', 'strict_match/2', 'strict_match/3', 'strict_match/4'],  # Process strict_match subfolders
            'data_20250911': ['suturing/strict_match/1', 'dissection/1']  # Process new data folders
        }
        
        # Data containers
        self.offset_data = []
        self.summary_stats = {}
        self.processed_datasets = []
        self.temporal_data = []  # For temporal analysis
        
        # Sensor categorization
        self.sensor_categories = {
            'image': ['header_img_left', 'header_img_right', 'header_img_side'],
            # 'jaw': ['header_jaw_meas', 'header_jaw_set'],  # 暂时注释掉jaw传感器
            'kinematics': [
                'header_cp_set', 'header_cv', 'header_js_set', 
                'header_lcp', 'header_measure_cp'
            ]
        }
        
        # Setpoint/Measure categorization
        self.setpoint_measure_categories = {
            'setpoint': ['header_cp_set', 'header_js_set'],
            'measured': ['header_measure_cp', 'header_cv', 'header_lcp'],
            'measured+img': ['header_measure_cp', 'header_cv', 'header_lcp', 
                            'header_img_left', 'header_img_right', 'header_img_side']
        }
        
        # All data is from CSR Camera system
        
        # Robot arms
        self.robot_arms = ['ECM', 'PSM1', 'PSM2', 'PSM3']
        
    def load_data(self) -> None:
        """Load all JSON files and extract timestamp information."""
        print("Loading data...")
        
        # Find all dataset folders in data_new
        if not self.data_root.exists():
            print(f"Error: Data root directory {self.data_root} not found!")
            return
            
        # Look for dataset folders that match our rules
        for dataset_folder in self.data_root.iterdir():
            if not dataset_folder.is_dir():
                continue
                
            dataset_name = dataset_folder.name
            print(f"\nFound dataset: {dataset_name}")
            
            # Check if this dataset has processing rules
            if dataset_name not in self.dataset_rules:
                print(f"  Skipping {dataset_name} - no processing rules defined")
                continue
                
            # Process each subfolder according to rules
            for subfolder in self.dataset_rules[dataset_name]:
                subfolder_path = dataset_folder / subfolder
                if not subfolder_path.exists():
                    print(f"  Warning: Subfolder {subfolder} not found in {dataset_name}")
                    continue
                    
                print(f"  Processing subfolder: {subfolder}")
                self._process_dataset_subfolder(dataset_name, subfolder, subfolder_path)
        
        print(f"\nLoaded {len(self.offset_data)} data points from {len(self.processed_datasets)} datasets")
    
    def _process_dataset_subfolder(self, dataset_name: str, subfolder: str, subfolder_path: Path) -> None:
        """Process a specific subfolder within a dataset."""
        # All data is from CSR Camera system
        camera_system = 'CSR Camera'
            
        # Look for regular folder
        regular_path = subfolder_path / "regular"
        if not regular_path.exists():
            print(f"    Warning: 'regular' folder not found in {subfolder_path}")
            return
            
        # Process kinematic data for each arm
        for arm in self.robot_arms:
            arm_path = regular_path / "kinematic" / arm
            
            if not arm_path.exists() or not any(arm_path.glob("*.json")):
                print(f"    Skipping {arm} - no data found")
                continue
                
            print(f"    Processing {arm} data...")
            
            # Get all JSON files for this arm
            json_files = sorted(glob.glob(str(arm_path / "*.json")))
            
            for json_file in json_files:
                try:
                    self._process_json_file(json_file, camera_system, arm, dataset_name, subfolder)
                except Exception as e:
                    print(f"      Error processing {json_file}: {e}")
                    continue
        
        # Process time_syn data for temporal analysis
        self._process_time_syn_data(dataset_name, subfolder, subfolder_path)
        
        # Record processed dataset
        self.processed_datasets.append({
            'dataset': dataset_name,
            'subfolder': subfolder,
            'camera_system': camera_system
        })
        
    def _process_json_file(self, json_file: str, camera_system: str, arm: str, dataset_name: str, subfolder: str) -> None:
        """Process a single JSON file and extract timestamp offsets."""
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract baseline timestamp (header_js_meas)
        header = data.get('header', {})
        baseline_data = header.get('header_js_meas', {})
        
        if 'sec' not in baseline_data or 'nsec' not in baseline_data:
            return
            
        baseline_timestamp = baseline_data['sec'] + baseline_data['nsec'] * 1e-9
        
        # Extract frame number from filename
        frame_num = int(Path(json_file).stem)
        
        # Calculate offsets for each sensor timestamp
        for sensor_key, sensor_data in header.items():
            # Skip the baseline timestamp (header_js_meas)
            if sensor_key == 'header_js_meas':
                continue
            
            # Skip jaw sensors completely (temporarily disabled)
            if sensor_key in ['header_jaw_meas', 'header_jaw_set']:
                continue
                
            if isinstance(sensor_data, dict) and 'sec' in sensor_data and 'nsec' in sensor_data:
                sensor_timestamp = sensor_data['sec'] + sensor_data['nsec'] * 1e-9
                offset_ms = (baseline_timestamp - sensor_timestamp) * 1000  # Convert to milliseconds
                
                # Determine sensor category
                category = self._get_sensor_category(sensor_key)
                setpoint_measure_category = self._get_setpoint_measure_category(sensor_key)
                
                self.offset_data.append({
                    'frame': frame_num,
                    'dataset': dataset_name,
                    'subfolder': subfolder,
                    'camera_system': camera_system,
                    'arm': arm,
                    'sensor': sensor_key,
                    'category': category,
                    'setpoint_measure_category': setpoint_measure_category,
                    'offset_ms': offset_ms,
                    'baseline_timestamp': baseline_timestamp,
                    'sensor_timestamp': sensor_timestamp
                })
            elif sensor_key in ['sec', 'nsec']:
                # Handle the unnamed timestamp (measure_cp)
                if sensor_key == 'sec':
                    sensor_timestamp = sensor_data + header.get('nsec', 0) * 1e-9
                    offset_ms = (baseline_timestamp - sensor_timestamp) * 1000
                    
                    self.offset_data.append({
                        'frame': frame_num,
                        'dataset': dataset_name,
                        'subfolder': subfolder,
                        'camera_system': camera_system,
                        'arm': arm,
                        'sensor': 'header_measure_cp',
                        'category': 'kinematics',
                        'setpoint_measure_category': self._get_setpoint_measure_category('header_measure_cp'),
                        'offset_ms': offset_ms,
                        'baseline_timestamp': baseline_timestamp,
                        'sensor_timestamp': sensor_timestamp
                    })
    
    def _process_time_syn_data(self, dataset_name: str, subfolder: str, subfolder_path: Path) -> None:
        """Process time_syn data to analyze actual sampling rate."""
        time_syn_path = subfolder_path / "regular" / "time_syn"
        
        if not time_syn_path.exists():
            print(f"    Warning: time_syn folder not found in {subfolder_path}")
            return
            
        # Get all time_syn JSON files
        time_syn_files = sorted(glob.glob(str(time_syn_path / "*.json")))
        
        if not time_syn_files:
            print(f"    Warning: No time_syn files found in {time_syn_path}")
            return
            
        print(f"    Processing time_syn data ({len(time_syn_files)} files)...")
        
        timestamps = []
        for json_file in time_syn_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Parse timestamp from format "sec_nsec"
                timestamp_str = data.get('timestamp', '')
                if '_' in timestamp_str:
                    sec_str, nsec_str = timestamp_str.split('_')
                    timestamp = float(sec_str) + float(nsec_str) * 1e-9
                    timestamps.append(timestamp)
                    
            except Exception as e:
                print(f"      Error processing time_syn file {json_file}: {e}")
                continue
        
        if len(timestamps) < 2:
            print(f"    Warning: Insufficient time_syn data for {dataset_name}/{subfolder}")
            return
            
        # Calculate sampling rate and duration
        timestamps = sorted(timestamps)
        duration = timestamps[-1] - timestamps[0]
        sampling_rate = (len(timestamps) - 1) / duration if duration > 0 else 0
        
        # Calculate frame intervals
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # Store temporal data
        self.temporal_data.append({
            'dataset': dataset_name,
            'subfolder': subfolder,
            'frame_count': len(timestamps),
            'duration_seconds': duration,
            'sampling_rate_hz': sampling_rate,
            'mean_interval_seconds': np.mean(intervals) if intervals else 0,
            'std_interval_seconds': np.std(intervals) if intervals else 0,
            'min_interval_seconds': np.min(intervals) if intervals else 0,
            'max_interval_seconds': np.max(intervals) if intervals else 0,
            'start_time': timestamps[0],
            'end_time': timestamps[-1]
        })
        
        print(f"      Duration: {duration:.2f}s, Sampling rate: {sampling_rate:.2f} Hz")
    
    def _get_real_timestamps(self, dataset_name: str, subfolder: str) -> List[float]:
        """Get real timestamps for each frame in a dataset."""
        # Find the dataset path
        dataset_path = self.data_root / dataset_name / subfolder
        time_syn_path = dataset_path / "regular" / "time_syn"
        
        if not time_syn_path.exists():
            return []
            
        # Get all time_syn JSON files
        time_syn_files = sorted(glob.glob(str(time_syn_path / "*.json")))
        
        timestamps = []
        for json_file in time_syn_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Parse timestamp from format "sec_nsec"
                timestamp_str = data.get('timestamp', '')
                if '_' in timestamp_str:
                    sec_str, nsec_str = timestamp_str.split('_')
                    timestamp = float(sec_str) + float(nsec_str) * 1e-9
                    timestamps.append(timestamp)
                    
            except Exception as e:
                print(f"      Error processing time_syn file {json_file}: {e}")
                continue
        
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
    
    def calculate_statistics(self) -> None:
        """Calculate comprehensive statistics for timestamp offsets."""
        print("Calculating statistics...")
        
        if not self.offset_data:
            print("No data loaded. Please run load_data() first.")
            return
            
        df = pd.DataFrame(self.offset_data)
        
        # Overall statistics
        self.summary_stats['overall'] = {
            'count': len(df),
            'mean_offset_ms': df['offset_ms'].abs().mean(),
            'std_offset_ms': df['offset_ms'].abs().std(),
            'min_offset_ms': df['offset_ms'].min(),
            'max_offset_ms': df['offset_ms'].max(),
            'median_offset_ms': df['offset_ms'].median(),
            'q25_offset_ms': df['offset_ms'].quantile(0.25),
            'q75_offset_ms': df['offset_ms'].quantile(0.75),
            'q95_offset_ms': df['offset_ms'].quantile(0.95)
        }
        
        # Statistics by camera system (removed - only one dataset)
        
        # Statistics by arm
        for arm in df['arm'].unique():
            arm_data = df[df['arm'] == arm]
            self.summary_stats[f'arm_{arm}'] = {
                'count': len(arm_data),
                'mean_offset_ms': arm_data['offset_ms'].abs().mean(),
                'std_offset_ms': arm_data['offset_ms'].abs().std(),
                'min_offset_ms': arm_data['offset_ms'].min(),
                'max_offset_ms': arm_data['offset_ms'].max(),
                'median_offset_ms': arm_data['offset_ms'].median()
            }
        
        # Statistics by sensor category
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            self.summary_stats[f'category_{category}'] = {
                'count': len(category_data),
                'mean_offset_ms': category_data['offset_ms'].abs().mean(),
                'std_offset_ms': category_data['offset_ms'].abs().std(),
                'min_offset_ms': category_data['offset_ms'].min(),
                'max_offset_ms': category_data['offset_ms'].max(),
                'median_offset_ms': category_data['offset_ms'].median()
            }
        
        # Statistics by setpoint/measure category
        for setpoint_measure_category in df['setpoint_measure_category'].unique():
            setpoint_measure_data = df[df['setpoint_measure_category'] == setpoint_measure_category]
            self.summary_stats[f'{setpoint_measure_category}'] = {
                'count': len(setpoint_measure_data),
                'mean_offset_ms': setpoint_measure_data['offset_ms'].abs().mean(),
                'std_offset_ms': setpoint_measure_data['offset_ms'].abs().std(),
                'min_offset_ms': setpoint_measure_data['offset_ms'].min(),
                'max_offset_ms': setpoint_measure_data['offset_ms'].max(),
                'median_offset_ms': setpoint_measure_data['offset_ms'].median()
            }
        
        # Combined img + measured statistics
        img_measured_data = df[df['setpoint_measure_category'].isin(['img', 'measured'])]
        if len(img_measured_data) > 0:
            self.summary_stats['img_measured_combined'] = {
                'count': len(img_measured_data),
                'mean_offset_ms': img_measured_data['offset_ms'].abs().mean(),
                'std_offset_ms': img_measured_data['offset_ms'].abs().std(),
                'min_offset_ms': img_measured_data['offset_ms'].min(),
                'max_offset_ms': img_measured_data['offset_ms'].max(),
                'median_offset_ms': img_measured_data['offset_ms'].median()
            }
        
        # Statistics by individual sensor
        for sensor in df['sensor'].unique():
            sensor_data = df[df['sensor'] == sensor]
            self.summary_stats[f'sensor_{sensor}'] = {
                'count': len(sensor_data),
                'mean_offset_ms': sensor_data['offset_ms'].abs().mean(),
                'std_offset_ms': sensor_data['offset_ms'].abs().std(),
                'min_offset_ms': sensor_data['offset_ms'].min(),
                'max_offset_ms': sensor_data['offset_ms'].max(),
                'median_offset_ms': sensor_data['offset_ms'].median()
            }
        
        # Add temporal statistics (per second)
        self._calculate_temporal_statistics(df)
        
        print("Statistics calculated successfully.")
    
    def _calculate_temporal_statistics(self, df: pd.DataFrame) -> None:
        """Calculate statistics per second based on temporal data."""
        if not self.temporal_data:
            print("No temporal data available for per-second statistics")
            return
            
        # Create temporal statistics
        temporal_df = pd.DataFrame(self.temporal_data)
        
        # Overall temporal statistics
        self.summary_stats['temporal_overall'] = {
            'total_datasets': len(temporal_df),
            'total_frames': temporal_df['frame_count'].sum(),
            'total_duration_seconds': temporal_df['duration_seconds'].sum(),
            'overall_sampling_rate_hz': temporal_df['sampling_rate_hz'].mean(),
            'mean_duration_per_dataset_seconds': temporal_df['duration_seconds'].mean(),
            'std_duration_per_dataset_seconds': temporal_df['duration_seconds'].std(),
            'mean_sampling_rate_hz': temporal_df['sampling_rate_hz'].mean(),
            'std_sampling_rate_hz': temporal_df['sampling_rate_hz'].std(),
            'min_sampling_rate_hz': temporal_df['sampling_rate_hz'].min(),
            'max_sampling_rate_hz': temporal_df['sampling_rate_hz'].max()
        }
        
        # Per-second offset statistics (using real timestamps)
        if not df.empty:
            # Calculate offsets per second for each dataset
            per_second_stats = {}
            
            for _, temporal_info in temporal_df.iterrows():
                dataset_name = temporal_info['dataset']
                subfolder = temporal_info['subfolder']
                sampling_rate = temporal_info['sampling_rate_hz']
                
                # Filter data for this dataset/subfolder
                dataset_data = df[(df['dataset'] == dataset_name) & (df['subfolder'] == subfolder)]
                
                if len(dataset_data) > 0:
                    # Get real timestamps for this dataset
                    real_timestamps = self._get_real_timestamps(dataset_name, subfolder)
                    
                    if real_timestamps:
                        # Create a mapping from frame number to timestamp
                        frame_to_timestamp = {}
                        for i, timestamp in enumerate(real_timestamps):
                            frame_to_timestamp[i] = timestamp
                        
                        # Add real timestamps to dataset_data
                        dataset_data = dataset_data.sort_values('frame')
                        dataset_data['real_timestamp'] = dataset_data['frame'].map(frame_to_timestamp)
                        
                        # Remove rows where timestamp is NaN (frames without time_syn data)
                        dataset_data = dataset_data.dropna(subset=['real_timestamp'])
                        
                        if len(dataset_data) > 0:
                            # Group by real time (second level)
                            dataset_data['second_group'] = dataset_data['real_timestamp'].astype(int)
                        
                        per_second_data = dataset_data.groupby('second_group')['offset_ms'].agg([
                            'count', 'mean', 'std', 'min', 'max', 'median'
                        ]).reset_index()
                        
                        per_second_data.columns = ['second', 'count', 'mean_offset_ms', 'std_offset_ms', 
                                                'min_offset_ms', 'max_offset_ms', 'median_offset_ms']
                        # Calculate absolute mean for per-second analysis
                        per_second_data['mean_offset_ms'] = dataset_data.groupby('second_group')['offset_ms'].apply(lambda x: x.abs().mean()).values
                        
                        per_second_stats[f'{dataset_name}_{subfolder}'] = {
                            'sampling_rate_hz': sampling_rate,
                            'total_seconds': len(per_second_data),
                            'per_second_stats': per_second_data.to_dict('records')
                        }
                    else:
                        # Fallback to frame-based grouping if no real timestamps
                        frames_per_second = int(sampling_rate) if sampling_rate > 0 else 1
                        dataset_data = dataset_data.sort_values('frame')
                        dataset_data['second_group'] = (dataset_data['frame'] // frames_per_second)
                        
                        per_second_data = dataset_data.groupby('second_group')['offset_ms'].agg([
                            'count', 'mean', 'std', 'min', 'max', 'median'
                        ]).reset_index()
                        
                        per_second_data.columns = ['second', 'count', 'mean_offset_ms', 'std_offset_ms', 
                                                'min_offset_ms', 'max_offset_ms', 'median_offset_ms']
                        # Calculate absolute mean for per-second analysis
                        per_second_data['mean_offset_ms'] = dataset_data.groupby('second_group')['offset_ms'].apply(lambda x: x.abs().mean()).values
                        
                        per_second_stats[f'{dataset_name}_{subfolder}'] = {
                            'sampling_rate_hz': sampling_rate,
                            'frames_per_second': frames_per_second,
                            'total_seconds': len(per_second_data),
                            'per_second_stats': per_second_data.to_dict('records')
                        }
            
            self.summary_stats['per_second_analysis'] = per_second_stats
            
            # Overall per-second statistics across all datasets
            all_per_second_offsets = []
            for dataset_name, subfolder in zip(temporal_df['dataset'], temporal_df['subfolder']):
                dataset_data = df[(df['dataset'] == dataset_name) & (df['subfolder'] == subfolder)]
                if len(dataset_data) > 0:
                    # Get real timestamps for this dataset
                    real_timestamps = self._get_real_timestamps(dataset_name, subfolder)
                    
                    if real_timestamps:
                        # Create a mapping from frame number to timestamp
                        frame_to_timestamp = {}
                        for i, timestamp in enumerate(real_timestamps):
                            frame_to_timestamp[i] = timestamp
                        
                        # Use real timestamps for grouping
                        dataset_data = dataset_data.sort_values('frame')
                        dataset_data['real_timestamp'] = dataset_data['frame'].map(frame_to_timestamp)
                        dataset_data = dataset_data.dropna(subset=['real_timestamp'])
                        
                        if len(dataset_data) > 0:
                            dataset_data['second_group'] = dataset_data['real_timestamp'].astype(int)
                        
                        per_second_means = dataset_data.groupby('second_group')['offset_ms'].apply(lambda x: x.abs().mean())
                        all_per_second_offsets.extend(per_second_means.tolist())
                    else:
                        # Fallback to frame-based grouping
                        sampling_rate = temporal_df[(temporal_df['dataset'] == dataset_name) & 
                                                  (temporal_df['subfolder'] == subfolder)]['sampling_rate_hz'].iloc[0]
                        frames_per_second = int(sampling_rate) if sampling_rate > 0 else 1
                        
                        dataset_data = dataset_data.sort_values('frame')
                        dataset_data['second_group'] = (dataset_data['frame'] // frames_per_second)
                        
                        per_second_means = dataset_data.groupby('second_group')['offset_ms'].apply(lambda x: x.abs().mean())
                        all_per_second_offsets.extend(per_second_means.tolist())
            
            if all_per_second_offsets:
                self.summary_stats['per_second_overall'] = {
                    'mean_offset_per_second_ms': np.mean(all_per_second_offsets),
                    'std_offset_per_second_ms': np.std(all_per_second_offsets),
                    'min_offset_per_second_ms': np.min(all_per_second_offsets),
                    'max_offset_per_second_ms': np.max(all_per_second_offsets),
                    'median_offset_per_second_ms': np.median(all_per_second_offsets),
                    'total_seconds_analyzed': len(all_per_second_offsets)
                }

    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations of timestamp offsets."""
        print("Creating visualizations...")
        
        if not self.offset_data:
            print("No data loaded. Please run load_data() first.")
            return
            
        df = pd.DataFrame(self.offset_data)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Overall offset distribution
        self._plot_overall_distribution(df)
        
        # 2. Camera system comparison (removed - only one dataset)
        # self._plot_camera_comparison(df)
        
        # 2. Arm comparison
        self._plot_arm_comparison(df)
        
        # 3. Sensor category analysis
        self._plot_sensor_category_analysis(df)
        
        # 4. Setpoint/Measure analysis
        self._plot_setpoint_measure_analysis(df)
        
        # 5. Individual sensor analysis
        self._plot_individual_sensors(df)
        
        # 6. Temporal stability analysis
        self._plot_temporal_stability(df)
        
        # 7. Sampling rate analysis
        self._plot_sampling_rate_analysis()
        
        print("Visualizations created successfully.")
    
    def _plot_overall_distribution(self, df: pd.DataFrame, plots_dir: Path = None) -> None:
        """Plot overall offset distribution."""
        if plots_dir is None:
            plots_dir = self.plots_dir
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(df['offset_ms'], bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Timestamp Offset (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Timestamp Offsets')
        ax1.axvline(df['offset_ms'].abs().mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["offset_ms"].abs().mean():.2f} ms')
        ax1.legend()
        
        # Box plot by category
        sns.boxplot(data=df, x='category', y='offset_ms', ax=ax2)
        ax2.set_xlabel('Sensor Category')
        ax2.set_ylabel('Timestamp Offset (ms)')
        ax2.set_title('Offset Distribution by Sensor Category')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'overall_offset_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_camera_comparison(self, df: pd.DataFrame) -> None:
        """Plot camera system comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot comparison
        sns.boxplot(data=df, x='camera_system', y='offset_ms', ax=ax1)
        ax1.set_title('Timestamp Offset by Camera System')
        ax1.set_ylabel('Offset (ms)')
        
        # Violin plot for detailed distribution
        sns.violinplot(data=df, x='camera_system', y='offset_ms', ax=ax2)
        ax2.set_title('Offset Distribution by Camera System (Detailed)')
        ax2.set_ylabel('Offset (ms)')
        
        # Category comparison within each camera
        sns.boxplot(data=df, x='category', y='offset_ms', hue='camera_system', ax=ax3)
        ax3.set_title('Sensor Category Offsets by Camera System')
        ax3.set_ylabel('Offset (ms)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Camera System')
        
        # Arm comparison within each camera
        sns.boxplot(data=df, x='arm', y='offset_ms', hue='camera_system', ax=ax4)
        ax4.set_title('Robot Arm Offsets by Camera System')
        ax4.set_ylabel('Offset (ms)')
        ax4.legend(title='Camera System')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'camera_system_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_arm_comparison(self, df: pd.DataFrame, plots_dir: Path = None) -> None:
        """Plot robot arm comparison."""
        if plots_dir is None:
            plots_dir = self.plots_dir
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall arm comparison
        sns.boxplot(data=df, x='arm', y='offset_ms', ax=ax1)
        ax1.set_title('Timestamp Offset by Robot Arm')
        ax1.set_ylabel('Offset (ms)')
        
        # Violin plot for detailed distribution
        sns.violinplot(data=df, x='arm', y='offset_ms', ax=ax2)
        ax2.set_title('Offset Distribution by Robot Arm (Detailed)')
        ax2.set_ylabel('Offset (ms)')
        
        # Category comparison within each arm
        sns.boxplot(data=df, x='category', y='offset_ms', hue='arm', ax=ax3)
        ax3.set_title('Sensor Category Offsets by Robot Arm')
        ax3.set_ylabel('Offset (ms)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Robot Arm')
        
        # Individual sensor comparison
        sensor_counts = df['sensor'].value_counts()
        top_sensors = sensor_counts.head(8).index
        df_top_sensors = df[df['sensor'].isin(top_sensors)]
        
        sns.boxplot(data=df_top_sensors, x='sensor', y='offset_ms', hue='arm', ax=ax4)
        ax4.set_title('Top Sensors Offsets by Robot Arm')
        ax4.set_ylabel('Offset (ms)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Robot Arm')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'robot_arm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensor_category_analysis(self, df: pd.DataFrame, plots_dir: Path = None) -> None:
        """Plot sensor category analysis."""
        if plots_dir is None:
            plots_dir = self.plots_dir
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Category comparison
        sns.boxplot(data=df, x='category', y='offset_ms', ax=ax1)
        ax1.set_title('Timestamp Offset by Sensor Category')
        ax1.set_ylabel('Offset (ms)')
        
        # Violin plot
        sns.violinplot(data=df, x='category', y='offset_ms', ax=ax2)
        ax2.set_title('Offset Distribution by Sensor Category (Detailed)')
        ax2.set_ylabel('Offset (ms)')
        
        # Category distribution by arm (replacing camera comparison)
        sns.boxplot(data=df, x='category', y='offset_ms', hue='arm', ax=ax3)
        ax3.set_title('Sensor Category Offsets by Robot Arm')
        ax3.set_ylabel('Offset (ms)')
        ax3.legend(title='Robot Arm')
        
        # Individual sensor distribution
        sensor_counts = df['sensor'].value_counts()
        top_sensors = sensor_counts.head(6).index
        df_top_sensors = df[df['sensor'].isin(top_sensors)]
        
        sns.boxplot(data=df_top_sensors, x='sensor', y='offset_ms', ax=ax4)
        ax4.set_title('Top Sensors Offset Distribution')
        ax4.set_ylabel('Offset (ms)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'sensor_category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_setpoint_measure_analysis(self, df: pd.DataFrame, plots_dir: Path = None) -> None:
        """Plot setpoint/measure analysis."""
        if plots_dir is None:
            plots_dir = self.plots_dir
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Setpoint/Measure comparison
        sns.boxplot(data=df, x='setpoint_measure_category', y='offset_ms', ax=ax1)
        ax1.set_title('Timestamp Offset by Setpoint/Measure Category')
        ax1.set_ylabel('Offset (ms)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(data=df, x='setpoint_measure_category', y='offset_ms', ax=ax2)
        ax2.set_title('Offset Distribution by Setpoint/Measure Category (Detailed)')
        ax2.set_ylabel('Offset (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Setpoint/Measure distribution by arm
        sns.boxplot(data=df, x='setpoint_measure_category', y='offset_ms', hue='arm', ax=ax3)
        ax3.set_title('Setpoint/Measure Offsets by Robot Arm')
        ax3.set_ylabel('Offset (ms)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Robot Arm')
        
        # Setpoint vs (Img + Measured) comparison
        # Create a new column for comparison
        df_comparison = df.copy()
        df_comparison['comparison_group'] = df_comparison['setpoint_measure_category'].apply(
            lambda x: 'setpoint' if x == 'setpoint' else 'img_measured'
        )
        
        comparison_data = df_comparison[df_comparison['setpoint_measure_category'].isin(['setpoint', 'measured', 'img'])]
        if len(comparison_data) > 0:
            sns.boxplot(data=comparison_data, x='comparison_group', y='offset_ms', ax=ax4)
            ax4.set_title('Setpoint vs (Img + Measured) Offset Comparison')
            ax4.set_ylabel('Offset (ms)')
            ax4.set_xlabel('Sensor Group')
        else:
            ax4.text(0.5, 0.5, 'No comparison data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Setpoint vs (Img + Measured) Offset Comparison')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'setpoint_measure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_sensors(self, df: pd.DataFrame, plots_dir: Path = None) -> None:
        """Plot individual sensor analysis."""
        if plots_dir is None:
            plots_dir = self.plots_dir
            
        # Get sensors with sufficient data
        sensor_counts = df['sensor'].value_counts()
        significant_sensors = sensor_counts[sensor_counts >= 10].index
        
        if len(significant_sensors) == 0:
            print("No sensors with sufficient data for individual analysis")
            return
            
        # Create subplot grid
        n_sensors = len(significant_sensors)
        n_cols = min(3, n_sensors)
        n_rows = (n_sensors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_sensors == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sensor in enumerate(significant_sensors):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            sensor_data = df[df['sensor'] == sensor]
            
            sns.boxplot(data=sensor_data, x='arm', y='offset_ms', ax=ax)
            ax.set_title(f'{sensor}')
            ax.set_ylabel('Offset (ms)')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for idx in range(n_sensors, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'individual_sensor_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_stability(self, df: pd.DataFrame, plots_dir: Path = None) -> None:
        """Plot temporal stability analysis."""
        if plots_dir is None:
            plots_dir = self.plots_dir
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Rolling mean of offsets over time
        df_sorted = df.sort_values('frame')
        
        # Overall temporal trend
        rolling_mean = df_sorted.groupby('frame')['offset_ms'].apply(lambda x: x.abs().mean()).rolling(window=50, center=True).mean()
        ax1.plot(rolling_mean.index, rolling_mean.values)
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Mean Offset (ms)')
        ax1.set_title('Temporal Stability - Overall Offset Trend')
        ax1.grid(True, alpha=0.3)
        
        # Temporal trend by sensor category
        for category in df['category'].unique():
            category_data = df_sorted[df_sorted['category'] == category]
            rolling_mean_category = category_data.groupby('frame')['offset_ms'].apply(lambda x: x.abs().mean()).rolling(window=50, center=True).mean()
            ax2.plot(rolling_mean_category.index, rolling_mean_category.values, label=category, marker='o', markersize=2)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Mean Offset (ms)')
        ax2.set_title('Temporal Stability by Sensor Category')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Temporal trend by arm
        for arm in df['arm'].unique():
            arm_data = df_sorted[df_sorted['arm'] == arm]
            rolling_mean_arm = arm_data.groupby('frame')['offset_ms'].apply(lambda x: x.abs().mean()).rolling(window=50, center=True).mean()
            ax3.plot(rolling_mean_arm.index, rolling_mean_arm.values, label=arm, marker='o', markersize=2)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Mean Offset (ms)')
        ax3.set_title('Temporal Stability by Robot Arm')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Offset variability over time
        rolling_std = df_sorted.groupby('frame')['offset_ms'].apply(lambda x: x.abs().std()).rolling(window=50, center=True).mean()
        ax4.plot(rolling_std.index, rolling_std.values, color='red')
        ax4.set_xlabel('Frame Number')
        ax4.set_ylabel('Offset Std Dev (ms)')
        ax4.set_title('Temporal Stability - Offset Variability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'temporal_stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sampling_rate_analysis(self, plots_dir: Path = None) -> None:
        """Plot sampling rate analysis."""
        if plots_dir is None:
            plots_dir = self.plots_dir
            
        if not self.temporal_data:
            print("No temporal data available for sampling rate analysis")
            return
            
        temporal_df = pd.DataFrame(self.temporal_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sampling rate by dataset
        dataset_labels = [f"{row['dataset']}\n{row['subfolder']}" for _, row in temporal_df.iterrows()]
        ax1.bar(range(len(temporal_df)), temporal_df['sampling_rate_hz'])
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Sampling Rate (Hz)')
        ax1.set_title('Sampling Rate by Dataset')
        ax1.set_xticks(range(len(temporal_df)))
        ax1.set_xticklabels(dataset_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(temporal_df['sampling_rate_hz']):
            ax1.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
        
        # 2. Duration by dataset
        ax2.bar(range(len(temporal_df)), temporal_df['duration_seconds'])
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Recording Duration by Dataset')
        ax2.set_xticks(range(len(temporal_df)))
        ax2.set_xticklabels(dataset_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(temporal_df['duration_seconds']):
            ax2.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
        
        # 3. Frame count vs duration scatter
        ax3.scatter(temporal_df['duration_seconds'], temporal_df['frame_count'], s=100, alpha=0.7)
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('Frame Count')
        ax3.set_title('Frame Count vs Duration')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(temporal_df['duration_seconds'], temporal_df['frame_count'], 1)
        p = np.poly1d(z)
        ax3.plot(temporal_df['duration_seconds'], p(temporal_df['duration_seconds']), "r--", alpha=0.8)
        
        # 4. Sampling rate distribution
        ax4.hist(temporal_df['sampling_rate_hz'], bins=10, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Sampling Rate (Hz)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Sampling Rate Distribution')
        ax4.axvline(temporal_df['sampling_rate_hz'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {temporal_df["sampling_rate_hz"].mean():.2f} Hz')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'sampling_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self) -> None:
        """Save analysis results to files."""
        print("Saving results...")
        
        if not self.offset_data:
            print("No data to save.")
            return
            
        df = pd.DataFrame(self.offset_data)
        
        # Group data by dataset and subfolder for organized output
        for dataset_info in self.processed_datasets:
            dataset_name = dataset_info['dataset']
            subfolder = dataset_info['subfolder']
            
            # Create output path: analysis_results/output/data_new/dataset_name/subfolder
            output_path = self.base_output_dir / "data_new" / dataset_name / subfolder
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Filter data for this specific dataset/subfolder
            dataset_data = df[(df['dataset'] == dataset_name) & (df['subfolder'] == subfolder)]
            
            if len(dataset_data) == 0:
                print(f"  No data found for {dataset_name}/{subfolder}")
                continue
                
            print(f"  Saving results for {dataset_name}/{subfolder} to {output_path}")
            
            # Calculate statistics for this dataset/subfolder
            dataset_stats = self._calculate_dataset_statistics(dataset_data)
            
            # Save summary statistics
            with open(output_path / 'summary_statistics.json', 'w') as f:
                json.dump(dataset_stats, f, indent=2)
            
            # Save detailed data as CSV
            dataset_data.to_csv(output_path / 'detailed_offset_data.csv', index=False)
            
            # Create plots for this dataset/subfolder
            self._create_dataset_plots(dataset_data, output_path)
            
            # Generate HTML report for this dataset/subfolder
            self._generate_dataset_html_report(dataset_data, dataset_stats, output_path, dataset_name, subfolder)
        
        # Also save overall combined results
        self._save_overall_results(df)
        
        print("Results saved successfully.")
    
    def _calculate_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for a specific dataset/subfolder."""
        stats = {}
        
        # Overall statistics
        stats['overall'] = {
            'count': len(df),
            'mean_offset_ms': df['offset_ms'].abs().mean(),
            'std_offset_ms': df['offset_ms'].abs().std(),
            'min_offset_ms': df['offset_ms'].min(),
            'max_offset_ms': df['offset_ms'].max(),
            'median_offset_ms': df['offset_ms'].median(),
            'q25_offset_ms': df['offset_ms'].quantile(0.25),
            'q75_offset_ms': df['offset_ms'].quantile(0.75),
            'q95_offset_ms': df['offset_ms'].quantile(0.95)
        }
        
        # Statistics by arm
        for arm in df['arm'].unique():
            arm_data = df[df['arm'] == arm]
            stats[f'arm_{arm}'] = {
                'count': len(arm_data),
                'mean_offset_ms': arm_data['offset_ms'].abs().mean(),
                'std_offset_ms': arm_data['offset_ms'].abs().std(),
                'min_offset_ms': arm_data['offset_ms'].min(),
                'max_offset_ms': arm_data['offset_ms'].max(),
                'median_offset_ms': arm_data['offset_ms'].median()
            }
        
        # Statistics by sensor category
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            stats[f'category_{category}'] = {
                'count': len(category_data),
                'mean_offset_ms': category_data['offset_ms'].abs().mean(),
                'std_offset_ms': category_data['offset_ms'].abs().std(),
                'min_offset_ms': category_data['offset_ms'].min(),
                'max_offset_ms': category_data['offset_ms'].max(),
                'median_offset_ms': category_data['offset_ms'].median()
            }
        
        # Statistics by setpoint/measure category
        for setpoint_measure_category in df['setpoint_measure_category'].unique():
            setpoint_measure_data = df[df['setpoint_measure_category'] == setpoint_measure_category]
            stats[f'{setpoint_measure_category}'] = {
                'count': len(setpoint_measure_data),
                'mean_offset_ms': setpoint_measure_data['offset_ms'].abs().mean(),
                'std_offset_ms': setpoint_measure_data['offset_ms'].abs().std(),
                'min_offset_ms': setpoint_measure_data['offset_ms'].min(),
                'max_offset_ms': setpoint_measure_data['offset_ms'].max(),
                'median_offset_ms': setpoint_measure_data['offset_ms'].median()
            }
        
        # Combined img + measured statistics
        img_measured_data = df[df['setpoint_measure_category'].isin(['img', 'measured'])]
        if len(img_measured_data) > 0:
            stats['img_measured_combined'] = {
                'count': len(img_measured_data),
                'mean_offset_ms': img_measured_data['offset_ms'].abs().mean(),
                'std_offset_ms': img_measured_data['offset_ms'].abs().std(),
                'min_offset_ms': img_measured_data['offset_ms'].min(),
                'max_offset_ms': img_measured_data['offset_ms'].max(),
                'median_offset_ms': img_measured_data['offset_ms'].median()
            }
        
        # Statistics by individual sensor
        for sensor in df['sensor'].unique():
            sensor_data = df[df['sensor'] == sensor]
            stats[f'sensor_{sensor}'] = {
                'count': len(sensor_data),
                'mean_offset_ms': sensor_data['offset_ms'].abs().mean(),
                'std_offset_ms': sensor_data['offset_ms'].abs().std(),
                'min_offset_ms': sensor_data['offset_ms'].min(),
                'max_offset_ms': sensor_data['offset_ms'].max(),
                'median_offset_ms': sensor_data['offset_ms'].median()
            }
        
        return stats
    
    def _create_dataset_plots(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create plots for a specific dataset/subfolder."""
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Overall offset distribution
        self._plot_overall_distribution(df, plots_dir)
        
        # 2. Arm comparison
        self._plot_arm_comparison(df, plots_dir)
        
        # 3. Sensor category analysis
        self._plot_sensor_category_analysis(df, plots_dir)
        
        # 4. Setpoint/Measure analysis
        self._plot_setpoint_measure_analysis(df, plots_dir)
        
        # 5. Individual sensor analysis
        self._plot_individual_sensors(df, plots_dir)
        
        # 6. Temporal stability analysis
        self._plot_temporal_stability(df, plots_dir)
        
        # 7. Sampling rate analysis (only for overall analysis)
        if output_path.name == "overall":
            self._plot_sampling_rate_analysis(plots_dir)
    
    def _save_overall_results(self, df: pd.DataFrame) -> None:
        """Save overall combined results."""
        overall_output_dir = self.base_output_dir / "overall"
        overall_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate overall statistics
        overall_stats = self._calculate_dataset_statistics(df)
        
        # Add temporal statistics to overall stats
        if 'temporal_overall' in self.summary_stats:
            overall_stats['temporal_overall'] = self.summary_stats['temporal_overall']
        if 'per_second_analysis' in self.summary_stats:
            overall_stats['per_second_analysis'] = self.summary_stats['per_second_analysis']
        if 'per_second_overall' in self.summary_stats:
            overall_stats['per_second_overall'] = self.summary_stats['per_second_overall']
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        overall_stats = convert_numpy_types(overall_stats)
        
        # Save overall summary statistics
        with open(overall_output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        # Save overall detailed data
        df.to_csv(overall_output_dir / 'detailed_offset_data.csv', index=False)
        
        # Create overall plots
        self._create_dataset_plots(df, overall_output_dir)
        
        # Generate overall HTML report
        self._generate_dataset_html_report(df, overall_stats, overall_output_dir, "Overall", "Combined")
    
    def _generate_dataset_html_report(self, df: pd.DataFrame, stats: Dict, output_path: Path, dataset_name: str, subfolder: str) -> None:
        """Generate an HTML report for a specific dataset/subfolder."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DVRK Timestamp Offset Analysis Report - {dataset_name}/{subfolder}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>DVRK Timestamp Offset Analysis Report</h1>
            <h2>Dataset: {dataset_name} / Subfolder: {subfolder}</h2>
            
            <div class="summary">
                <h2>Analysis Summary</h2>
                <p>This report analyzes timestamp offsets in synchronized multimodal data from the DVRK system.</p>
                <p><strong>Total data points analyzed:</strong> {total_points}</p>
                <p><strong>Dataset:</strong> {dataset_name}</p>
                <p><strong>Subfolder:</strong> {subfolder}</p>
                <p><strong>Robot Arms:</strong> {arms}</p>
                <p><strong>Sensor Categories:</strong> {categories}</p>
                <p><strong>Sampling Rate:</strong> {sampling_rate:.2f} Hz</p>
                <p><strong>Duration:</strong> {duration:.2f} seconds</p>
            </div>
            
            <h2>Key Findings</h2>
            <div class="summary">
                <p><strong>Overall Mean Offset:</strong> {mean_offset:.2f} ms</p>
                <p><strong>Overall Standard Deviation:</strong> {std_offset:.2f} ms</p>
                <p><strong>95th Percentile:</strong> {q95_offset:.2f} ms</p>
            </div>
            
            <h2>Visualizations</h2>
            <div class="plot">
                <h3>Overall Offset Distribution</h3>
                <img src="plots/overall_offset_distribution.png" alt="Overall Distribution">
            </div>
            
            <div class="plot">
                <h3>Robot Arm Comparison</h3>
                <img src="plots/robot_arm_comparison.png" alt="Arm Comparison">
            </div>
            
            <div class="plot">
                <h3>Sensor Category Analysis</h3>
                <img src="plots/sensor_category_analysis.png" alt="Sensor Category">
            </div>
            
            <div class="plot">
                <h3>Individual Sensor Analysis</h3>
                <img src="plots/individual_sensor_analysis.png" alt="Individual Sensors">
            </div>
            
            <div class="plot">
                <h3>Temporal Stability Analysis</h3>
                <img src="plots/temporal_stability_analysis.png" alt="Temporal Stability">
            </div>
            
            <h2>Detailed Statistics</h2>
            <p>For detailed statistical results, please refer to the summary_statistics.json file.</p>
            
        </body>
        </html>
        """
        
        # Fill in the template with actual values
        if len(df) > 0 and 'overall' in stats:
            overall = stats['overall']
            arms = ', '.join(df['arm'].unique())
            categories = ', '.join(df['category'].unique())
            
            # Get sampling rate and duration from temporal data
            sampling_rate = 0
            duration = 0
            for temporal_info in self.temporal_data:
                if temporal_info['dataset'] == dataset_name and temporal_info['subfolder'] == subfolder:
                    sampling_rate = temporal_info['sampling_rate_hz']
                    duration = temporal_info['duration_seconds']
                    break
            
            filled_html = html_content.format(
                dataset_name=dataset_name,
                subfolder=subfolder,
                total_points=overall['count'],
                mean_offset=overall['mean_offset_ms'],
                std_offset=overall['std_offset_ms'],
                q95_offset=overall['q95_offset_ms'],
                arms=arms,
                categories=categories,
                sampling_rate=sampling_rate,
                duration=duration
            )
        else:
            filled_html = html_content.format(
                dataset_name=dataset_name,
                subfolder=subfolder,
                total_points=0,
                mean_offset=0,
                std_offset=0,
                q95_offset=0,
                arms="None",
                categories="None",
                sampling_rate=0,
                duration=0
            )
        
        with open(output_path / 'detailed_analysis_report.html', 'w') as f:
            f.write(filled_html)
    
    def _generate_html_report(self) -> None:
        """Generate an HTML report with analysis results."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DVRK Timestamp Offset Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>DVRK Timestamp Offset Analysis Report</h1>
            
            <div class="summary">
                <h2>Analysis Summary</h2>
                <p>This report analyzes timestamp offsets in synchronized multimodal data from the DVRK system.</p>
                <p><strong>Total data points analyzed:</strong> {total_points}</p>
                <p><strong>Dataset:</strong> CSR Camera (Dataset 2)</p>
                <p><strong>Robot Arms:</strong> ECM, PSM1, PSM2</p>
                <p><strong>Sensor Categories:</strong> Image, Kinematics</p>  # 移除Jaw
            </div>
            
            <h2>Key Findings</h2>
            <div class="summary">
                <p><strong>Overall Mean Offset:</strong> {mean_offset:.2f} ms</p>
                <p><strong>Overall Standard Deviation:</strong> {std_offset:.2f} ms</p>
                <p><strong>95th Percentile:</strong> {q95_offset:.2f} ms</p>
            </div>
            
            <h2>Visualizations</h2>
            <div class="plot">
                <h3>Overall Offset Distribution</h3>
                <img src="plots/overall_offset_distribution.png" alt="Overall Distribution">
            </div>
            
            
            <div class="plot">
                <h3>Robot Arm Comparison</h3>
                <img src="plots/robot_arm_comparison.png" alt="Arm Comparison">
            </div>
            
            <div class="plot">
                <h3>Sensor Category Analysis</h3>
                <img src="plots/sensor_category_analysis.png" alt="Sensor Category">
            </div>
            
            <div class="plot">
                <h3>Individual Sensor Analysis</h3>
                <img src="plots/individual_sensor_analysis.png" alt="Individual Sensors">
            </div>
            
            <div class="plot">
                <h3>Temporal Stability Analysis</h3>
                <img src="plots/temporal_stability_analysis.png" alt="Temporal Stability">
            </div>
            
            <h2>Detailed Statistics</h2>
            <p>For detailed statistical results, please refer to the summary_statistics.json file.</p>
            
        </body>
        </html>
        """
        
        # Fill in the template with actual values
        if self.offset_data and 'overall' in self.summary_stats:
            overall = self.summary_stats['overall']
            filled_html = html_content.format(
                total_points=overall['count'],
                mean_offset=overall['mean_offset_ms'],
                std_offset=overall['std_offset_ms'],
                q95_offset=overall['q95_offset_ms']
            )
        else:
            filled_html = html_content.format(
                total_points=0,
                mean_offset=0,
                std_offset=0,
                q95_offset=0
            )
        
        with open(self.results_dir / 'detailed_analysis_report.html', 'w') as f:
            f.write(filled_html)
    
    def run_full_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        print("Starting DVRK Timestamp Offset Analysis...")
        print("=" * 50)
        
        try:
            self.load_data()
            self.calculate_statistics()
            self.save_results()
            
            print("=" * 50)
            print("Analysis completed successfully!")
            print(f"Results saved to: {self.base_output_dir}")
            
            # Print summary
            if self.offset_data:
                df = pd.DataFrame(self.offset_data)
                print(f"\nQuick Summary:")
                print(f"  Total data points: {len(df)}")
                print(f"  Processed datasets: {len(self.processed_datasets)}")
                print(f"  Mean offset: {df['offset_ms'].abs().mean():.2f} ms")
                print(f"  Std deviation: {df['offset_ms'].abs().std():.2f} ms")
                print(f"  95th percentile: {df['offset_ms'].quantile(0.95):.2f} ms")
                
                # Print temporal summary
                if self.temporal_data:
                    temporal_df = pd.DataFrame(self.temporal_data)
                    print(f"\nTemporal Analysis:")
                    print(f"  Average sampling rate: {temporal_df['sampling_rate_hz'].mean():.2f} Hz")
                    print(f"  Total duration: {temporal_df['duration_seconds'].sum():.2f} seconds")
                    print(f"  Total frames: {temporal_df['frame_count'].sum()}")
                    
                    # Print per-dataset sampling rates
                    print(f"\nSampling Rates by Dataset:")
                    for _, row in temporal_df.iterrows():
                        print(f"  {row['dataset']}/{row['subfolder']}: {row['sampling_rate_hz']:.2f} Hz ({row['duration_seconds']:.1f}s)")
                
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Main function to run the analysis."""
    analyzer = TimestampAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
