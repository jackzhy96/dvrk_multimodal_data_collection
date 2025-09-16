#!/usr/bin/env python3
"""
Timestamp Delay Analysis for DVRK Interpolation Data

This script focuses purely on analyzing timestamp delays without interpolation.
It calculates the delay between sensor timestamps and the baseline timestamp
from time_syn folder's image_stamp_left.

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

class TimestampDelayAnalyzer:
    """Pure timestamp delay analyzer for interpolation data."""
    
    def __init__(self, data_root: str = "../../data/data_new"):
        """
        Initialize the analyzer.
        
        Args:
            data_root: Root directory containing the data_new folder
        """
        self.data_root = Path(data_root)
        self.base_output_dir = Path("output")
        
        # Dataset processing rules for interpolation data - only process data_20250909
        self.dataset_rules = {
            'data_20250909': ['interpolation/1', 'interpolation/2', 'interpolation/3', 'interpolation/4']
        }
        
        # Data containers
        self.delay_data = []
        self.summary_stats = {}
        self.processed_datasets = []
        self.temporal_data = []
        
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
        
        # Setpoint/Measure categorization
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
        
        # Robot arms
        self.robot_arms = ['ECM', 'PSM1', 'PSM2']
        
    def load_data(self) -> None:
        """Load all interpolation JSON files and extract timestamp delays."""
        print("Loading interpolation data for timestamp delay analysis...")
        
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
                
                # Record processed dataset info
                self.processed_datasets.append({
                    'dataset': dataset_name,
                    'subfolder': subfolder
                })
        
        print(f"\nLoaded {len(self.delay_data)} delay measurements from {len(set([d['dataset'] for d in self.delay_data]))} datasets")
    
    def _process_dataset_subfolder(self, dataset_name: str, subfolder: str, subfolder_path: Path) -> None:
        """Process a single dataset subfolder."""
        # Process kinematic data for each arm
        kinematic_path = subfolder_path / "regular" / "kinematic"
        
        if not kinematic_path.exists():
            print(f"    Warning: kinematic folder not found in {subfolder_path}")
            return
            
        for arm in self.robot_arms:
            arm_path = kinematic_path / arm
            if not arm_path.exists():
                print(f"    Skipping {arm} - no data found")
                continue
                
            print(f"    Processing {arm} data...")
            self._process_arm_data(dataset_name, subfolder, arm, arm_path)
        
        # Process time_syn data for sampling rate analysis
        self._process_time_syn_data(dataset_name, subfolder, subfolder_path)
    
    def _process_arm_data(self, dataset_name: str, subfolder: str, arm: str, arm_path: Path) -> None:
        """Process kinematic data for a single arm."""
        # Get all JSON files
        json_files = sorted(glob.glob(str(arm_path / "*.json")))
        
        for json_file in json_files:
            self._process_json_file(json_file, "interpolation", arm, dataset_name, subfolder)
    
    def _process_json_file(self, json_file: str, camera_system: str, arm: str, dataset_name: str, subfolder: str) -> None:
        """Process a single interpolation JSON file and extract timestamp delays."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            frame_num = int(Path(json_file).stem)
            
            # Get the baseline timestamp from time_syn folder
            baseline_timestamp = self._get_baseline_timestamp(dataset_name, subfolder, frame_num)
            if baseline_timestamp is None:
                return
            
            # Process each of the 5 candidate points
            for candidate_idx, candidate_data in enumerate(data):
                # Extract all timestamps from this candidate
                timestamps = self._extract_all_timestamps(candidate_data, arm)
                
                # Calculate delays for each timestamp
                for sensor_key, sensor_timestamp in timestamps.items():
                    if sensor_timestamp is None:
                        continue
                    
                    # Calculate delay: sensor_timestamp - baseline_timestamp
                    delay_ms = (baseline_timestamp - sensor_timestamp) * 1000
                    
                    # Determine sensor category
                    category = self._get_sensor_category(sensor_key)
                    setpoint_measure_category = self._get_setpoint_measure_category(sensor_key)
                    data_type_category = self._get_data_type_category(sensor_key)
                    
                    self.delay_data.append({
                        'frame': frame_num,
                        'candidate': candidate_idx,
                        'dataset': dataset_name,
                        'subfolder': subfolder,
                        'camera_system': camera_system,
                        'arm': arm,
                        'sensor': sensor_key,
                        'category': category,
                        'setpoint_measure_category': setpoint_measure_category,
                        'data_type_category': data_type_category,
                        'delay_ms': delay_ms,
                        'abs_delay_ms': abs(delay_ms),
                        'baseline_timestamp': baseline_timestamp,
                        'sensor_timestamp': sensor_timestamp
                    })
            
                    
        except Exception as e:
            print(f"      Error processing {json_file}: {e}")
            return
    
    def _extract_all_timestamps(self, candidate_data: Dict, arm: str) -> Dict[str, Optional[float]]:
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
    
    def _get_baseline_timestamp(self, dataset_name: str, subfolder: str, frame_num: int) -> Optional[float]:
        """Get the baseline timestamp from time_syn folder for a specific frame."""
        dataset_path = self.data_root / dataset_name / subfolder
        time_syn_path = dataset_path / "regular" / "time_syn"
        
        if not time_syn_path.exists():
            return None
            
        time_syn_file = time_syn_path / f"{frame_num}.json"
        
        if not time_syn_file.exists():
            return None
            
        try:
            with open(time_syn_file, 'r') as f:
                data = json.load(f)
            
            # Use image_stamp_left as baseline
            if 'image_stamp_left' in data and isinstance(data['image_stamp_left'], dict):
                stamp_data = data['image_stamp_left']
                if 'sec' in stamp_data and 'nsec' in stamp_data:
                    return stamp_data['sec'] + stamp_data['nsec'] * 1e-9
            
            # Fallback to image_stamp_right if left is not available
            elif 'image_stamp_right' in data and isinstance(data['image_stamp_right'], dict):
                stamp_data = data['image_stamp_right']
                if 'sec' in stamp_data and 'nsec' in stamp_data:
                    return stamp_data['sec'] + stamp_data['nsec'] * 1e-9
                    
        except Exception as e:
            print(f"      Error reading time_syn file {time_syn_file}: {e}")
            
        return None
    
    def _process_time_syn_data(self, dataset_name: str, subfolder: str, subfolder_path: Path) -> None:
        """Process time_syn data to analyze actual sampling rate."""
        time_syn_path = subfolder_path / "regular" / "time_syn"
        
        if not time_syn_path.exists():
            print(f"    Warning: time_syn folder not found in {subfolder_path}")
            return
            
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
                
                # Parse timestamp from image_stamp_left
                if 'image_stamp_left' in data and isinstance(data['image_stamp_left'], dict):
                    stamp_data = data['image_stamp_left']
                    if 'sec' in stamp_data and 'nsec' in stamp_data:
                        timestamp = stamp_data['sec'] + stamp_data['nsec'] * 1e-9
                        timestamps.append(timestamp)
                        
            except Exception as e:
                continue
        
        if len(timestamps) >= 2:
            duration = (timestamps[-1] - timestamps[0])
            sampling_rate = (len(timestamps) - 1) / duration if duration > 0 else 0
            
            print(f"      Duration: {duration:.2f}s, Sampling rate: {sampling_rate:.2f} Hz")
            
            self.temporal_data.append({
                'dataset': dataset_name,
                'subfolder': subfolder,
                'duration': duration,
                'sampling_rate': sampling_rate,
                'frame_count': len(timestamps)
            })
    
    def _get_sensor_category(self, sensor_key: str) -> str:
        """Determine sensor category based on sensor key."""
        for category, sensors in self.sensor_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def _get_setpoint_measure_category(self, sensor_key: str) -> str:
        """Determine setpoint/measure category based on sensor key."""
        for category, sensors in self.setpoint_measure_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def _get_data_type_category(self, sensor_key: str) -> str:
        """Determine data type category based on sensor key."""
        for category, sensors in self.data_type_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def calculate_delay_statistics(self) -> None:
        """Calculate comprehensive delay statistics."""
        print("Calculating delay statistics...")
        
        if not self.delay_data:
            print("No data loaded. Please run load_data() first.")
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
            'p25_delay_ms': df['delay_ms'].quantile(0.25),
            'p75_delay_ms': df['delay_ms'].quantile(0.75),
            'p95_delay_ms': df['delay_ms'].quantile(0.95),
            'p99_delay_ms': df['delay_ms'].quantile(0.99)
        }
        
        # Statistics by category
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            self.summary_stats[f'category_{category}'] = {
                'count': len(category_data),
                'mean_delay_ms': category_data['abs_delay_ms'].mean(),
                'std_delay_ms': category_data['abs_delay_ms'].std(),
                'min_delay_ms': category_data['delay_ms'].min(),
                'max_delay_ms': category_data['delay_ms'].max(),
                'median_delay_ms': category_data['delay_ms'].median(),
                'abs_median_delay_ms': category_data['abs_delay_ms'].median()
            }
        
        # Statistics by arm
        for arm in df['arm'].unique():
            arm_data = df[df['arm'] == arm]
            self.summary_stats[f'arm_{arm}'] = {
                'count': len(arm_data),
                'mean_delay_ms': arm_data['abs_delay_ms'].mean(),
                'std_delay_ms': arm_data['abs_delay_ms'].std(),
                'min_delay_ms': arm_data['delay_ms'].min(),
                'max_delay_ms': arm_data['delay_ms'].max(),
                'median_delay_ms': arm_data['delay_ms'].median(),
                'abs_median_delay_ms': arm_data['abs_delay_ms'].median()
            }
        
        # Statistics by candidate
        for candidate in df['candidate'].unique():
            candidate_data = df[df['candidate'] == candidate]
            self.summary_stats[f'candidate_{candidate}'] = {
                'count': len(candidate_data),
                'mean_delay_ms': candidate_data['abs_delay_ms'].mean(),
                'std_delay_ms': candidate_data['abs_delay_ms'].std(),
                'min_delay_ms': candidate_data['delay_ms'].min(),
                'max_delay_ms': candidate_data['delay_ms'].max(),
                'median_delay_ms': candidate_data['delay_ms'].median(),
                'abs_median_delay_ms': candidate_data['abs_delay_ms'].median()
            }
        
        print("Delay statistics calculated successfully.")
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations for delay analysis."""
        print("Creating delay visualizations...")
        
        if not self.delay_data:
            print("No data loaded. Please run load_data() first.")
            return
        
        df = pd.DataFrame(self.delay_data)
        
        # Set up plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Create dataset-specific outputs (following strict_match pattern)
        self._create_dataset_specific_outputs(df)
        
        # Create overall analysis
        self._create_overall_analysis(df)
        
        print("Visualizations created successfully.")
    
    def _create_dataset_specific_outputs(self, df: pd.DataFrame) -> None:
        """Create outputs organized by dataset and subfolder (like strict_match)."""
        print("Creating dataset-specific outputs...")
        
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
            stats = self._calculate_dataset_statistics(dataset_data)
            
            # Create plots for this dataset/subfolder
            self._create_dataset_plots(dataset_data, output_path)
            
            # Save CSV data
            dataset_data.to_csv(output_path / "detailed_delay_data.csv", index=False)
            
            # Save statistics
            with open(output_path / "summary_statistics.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate HTML report
            self._generate_dataset_html_report(dataset_data, stats, output_path, dataset_name, subfolder)
    
    def _create_overall_analysis(self, df: pd.DataFrame) -> None:
        """Create overall analysis combining all datasets."""
        print("Creating overall analysis...")
        
        # Create overall output directory
        overall_path = self.base_output_dir / "overall"
        overall_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate overall statistics
        overall_stats = self._calculate_dataset_statistics(df)
        
        # Create overall plots
        self._create_dataset_plots(df, overall_path)
        
        # Save overall CSV data
        df.to_csv(overall_path / "detailed_delay_data.csv", index=False)
        
        # Save overall statistics
        with open(overall_path / "summary_statistics.json", 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        # Generate overall HTML report
        self._generate_dataset_html_report(df, overall_stats, overall_path, "overall", "all_datasets")
    
    def _calculate_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for a specific dataset."""
        stats = {
            'count': len(df),
            'mean_delay_ms': df['delay_ms'].abs().mean(),
            'std_delay_ms': df['delay_ms'].abs().std(),
            'min_delay_ms': df['delay_ms'].min(),
            'max_delay_ms': df['delay_ms'].max(),
            'median_delay_ms': df['delay_ms'].median(),
            'abs_median_delay_ms': df['delay_ms'].abs().median(),
            'p25_delay_ms': df['delay_ms'].quantile(0.25),
            'p75_delay_ms': df['delay_ms'].quantile(0.75),
            'p95_delay_ms': df['delay_ms'].quantile(0.95),
            'p99_delay_ms': df['delay_ms'].quantile(0.99)
        }
        
        # Add category statistics
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            stats[f'category_{category}'] = {
                'count': len(category_data),
                'mean_delay_ms': category_data['delay_ms'].abs().mean(),
                'std_delay_ms': category_data['delay_ms'].abs().std(),
                'median_delay_ms': category_data['delay_ms'].median(),
                'abs_median_delay_ms': category_data['delay_ms'].abs().median()
            }
        
        # Add arm statistics
        for arm in df['arm'].unique():
            arm_data = df[df['arm'] == arm]
            stats[f'arm_{arm}'] = {
                'count': len(arm_data),
                'mean_delay_ms': arm_data['delay_ms'].abs().mean(),
                'std_delay_ms': arm_data['delay_ms'].abs().std(),
                'median_delay_ms': arm_data['delay_ms'].median(),
                'abs_median_delay_ms': arm_data['delay_ms'].abs().median()
            }
        
        # Add sensor statistics
        for sensor in df['sensor'].unique():
            sensor_data = df[df['sensor'] == sensor]
            stats[f'sensor_{sensor}'] = {
                'count': len(sensor_data),
                'mean_delay_ms': sensor_data['delay_ms'].abs().mean(),
                'std_delay_ms': sensor_data['delay_ms'].abs().std(),
                'median_delay_ms': sensor_data['delay_ms'].median(),
                'abs_median_delay_ms': sensor_data['delay_ms'].abs().median()
            }
        
        # Add setpoint/measure category statistics
        for category in df['setpoint_measure_category'].unique():
            category_data = df[df['setpoint_measure_category'] == category]
            stats[f'{category}'] = {
                'count': len(category_data),
                'mean_delay_ms': category_data['delay_ms'].abs().mean(),
                'std_delay_ms': category_data['delay_ms'].abs().std(),
                'median_delay_ms': category_data['delay_ms'].median(),
                'abs_median_delay_ms': category_data['delay_ms'].abs().median()
            }
        
        # Add data type category statistics
        for category in df['data_type_category'].unique():
            category_data = df[df['data_type_category'] == category]
            stats[f'data_type_{category}'] = {
                'count': len(category_data),
                'mean_delay_ms': category_data['delay_ms'].abs().mean(),
                'std_delay_ms': category_data['delay_ms'].abs().std(),
                'median_delay_ms': category_data['delay_ms'].median(),
                'abs_median_delay_ms': category_data['delay_ms'].abs().median()
            }
        
        return stats
    
    def _create_dataset_plots(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create plots for a specific dataset."""
        plots_dir = output_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Overall delay distribution
        self._plot_overall_distribution(df, plots_dir)
        
        # 2. Delay by candidate
        self._plot_candidate_analysis(df, plots_dir)
        
        # 3. Delay by sensor category
        self._plot_category_analysis(df, plots_dir)
        
        # 4. Delay by arm
        self._plot_arm_analysis(df, plots_dir)
        
        # 5. Frame-level analysis
        self._plot_frame_analysis(df, plots_dir)
    
    def _generate_dataset_html_report(self, df: pd.DataFrame, stats: Dict, output_path: Path, dataset_name: str, subfolder: str) -> None:
        """Generate HTML report for a specific dataset."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DVRK Interpolation Delay Analysis - {dataset_name}/{subfolder}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats {{ margin: 20px 0; }}
                .stat-item {{ margin: 10px 0; }}
                .stat-label {{ font-weight: bold; }}
                .stat-value {{ color: #0066cc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DVRK Interpolation Delay Analysis</h1>
                <h2>Dataset: {dataset_name}/{subfolder}</h2>
                <p>Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <h3>Summary Statistics</h3>
                <div class="stat-item">
                    <span class="stat-label">Total Measurements:</span>
                    <span class="stat-value">{stats['count']:,}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Mean Delay:</span>
                    <span class="stat-value">{stats['mean_delay_ms']:.3f} ms</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Std Deviation:</span>
                    <span class="stat-value">{stats['std_delay_ms']:.3f} ms</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">95th Percentile:</span>
                    <span class="stat-value">{stats['p95_delay_ms']:.3f} ms</span>
                </div>
            </div>
            
            <div class="stats">
                <h3>Generated Plots</h3>
                <p>Check the 'plots' folder for detailed visualizations.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path / "detailed_analysis_report.html", 'w') as f:
            f.write(html_content)
    
    def _plot_overall_distribution(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot overall delay distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(df['delay_ms'].abs(), bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Delay (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Delay Distribution (Absolute Values)')
        ax1.axvline(df['delay_ms'].abs().mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["delay_ms"].abs().mean():.2f} ms')
        ax1.legend()
        
        # Box plot by candidate
        sns.boxplot(data=df, x='candidate', y='delay_ms', ax=ax2)
        ax2.set_xlabel('Candidate Index')
        ax2.set_ylabel('Delay (ms)')
        ax2.set_title('Delay Distribution by Candidate')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'overall_delay_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_candidate_analysis(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot delay analysis by candidate."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mean delay by candidate
        candidate_stats = df.groupby('candidate')['delay_ms'].agg(['mean', 'std', 'count']).reset_index()
        candidate_stats['mean'] = candidate_stats['mean'].abs()
        candidate_stats['std'] = candidate_stats['std'].abs()
        ax1.bar(candidate_stats['candidate'], candidate_stats['mean'], 
                yerr=candidate_stats['std'], capsize=5, alpha=0.7)
        ax1.set_xlabel('Candidate Index')
        ax1.set_ylabel('Mean Delay (ms)')
        ax1.set_title('Mean Delay by Candidate (Absolute Values)')
        
        # Delay distribution by candidate
        for candidate in sorted(df['candidate'].unique()):
            candidate_data = df[df['candidate'] == candidate]['delay_ms'].abs()
            ax2.hist(candidate_data, alpha=0.6, label=f'Candidate {candidate}', bins=30)
        ax2.set_xlabel('Delay (ms)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Delay Distribution by Candidate (Absolute Values)')
        ax2.legend()
        
        # Delay range by candidate
        candidate_ranges = df.groupby('candidate')['delay_ms'].agg(['min', 'max']).reset_index()
        candidate_ranges['range'] = candidate_ranges['max'] - candidate_ranges['min']
        ax3.bar(candidate_ranges['candidate'], candidate_ranges['range'], alpha=0.7)
        ax3.set_xlabel('Candidate Index')
        ax3.set_ylabel('Delay Range (ms)')
        ax3.set_title('Delay Range by Candidate')
        
        # Delay stability (std dev) by candidate
        ax4.bar(candidate_stats['candidate'], candidate_stats['std'], alpha=0.7)
        ax4.set_xlabel('Candidate Index')
        ax4.set_ylabel('Delay Std Dev (ms)')
        ax4.set_title('Delay Stability by Candidate (Absolute Values)')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'candidate_delay_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_analysis(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot delay analysis by sensor category."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot by category
        sns.boxplot(data=df, x='category', y='delay_ms', ax=ax1)
        ax1.set_xlabel('Sensor Category')
        ax1.set_ylabel('Delay (ms)')
        ax1.set_title('Delay Distribution by Sensor Category')
        
        # Mean delay by category
        category_stats = df.groupby('category')['delay_ms'].agg(['mean', 'std', 'count']).reset_index()
        category_stats['mean'] = category_stats['mean'].abs()
        category_stats['std'] = category_stats['std'].abs()
        ax2.bar(category_stats['category'], category_stats['mean'], 
                yerr=category_stats['std'], capsize=5, alpha=0.7)
        ax2.set_xlabel('Sensor Category')
        ax2.set_ylabel('Mean Delay (ms)')
        ax2.set_title('Mean Delay by Sensor Category (Absolute Values)')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'category_delay_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_arm_analysis(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot delay analysis by robot arm."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot by arm
        sns.boxplot(data=df, x='arm', y='delay_ms', ax=ax1)
        ax1.set_xlabel('Robot Arm')
        ax1.set_ylabel('Delay (ms)')
        ax1.set_title('Delay Distribution by Robot Arm')
        
        # Mean delay by arm
        arm_stats = df.groupby('arm')['delay_ms'].agg(['mean', 'std', 'count']).reset_index()
        arm_stats['mean'] = arm_stats['mean'].abs()
        arm_stats['std'] = arm_stats['std'].abs()
        ax2.bar(arm_stats['arm'], arm_stats['mean'], 
                yerr=arm_stats['std'], capsize=5, alpha=0.7)
        ax2.set_xlabel('Robot Arm')
        ax2.set_ylabel('Mean Delay (ms)')
        ax2.set_title('Mean Delay by Robot Arm (Absolute Values)')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'arm_delay_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_frame_analysis(self, df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot frame-level delay analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Delay over time (frames)
        frame_stats = df.groupby('frame')['delay_ms'].agg(['mean', 'std']).reset_index()
        ax1.plot(frame_stats['frame'], frame_stats['mean'], label='Mean Delay', alpha=0.8)
        ax1.fill_between(frame_stats['frame'], 
                        frame_stats['mean'] - frame_stats['std'],
                        frame_stats['mean'] + frame_stats['std'],
                        alpha=0.3, label='±1 Std Dev')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Delay (ms)')
        ax1.set_title('Delay Over Time (Frames)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Delay stability over time
        ax2.plot(frame_stats['frame'], frame_stats['std'], color='red', alpha=0.8)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Delay Std Dev (ms)')
        ax2.set_title('Delay Stability Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'frame_delay_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_analysis(self) -> None:
        """Run the complete timestamp delay analysis pipeline."""
        print("=" * 60)
        print("DVRK Timestamp Delay Analysis")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        if not self.delay_data:
            print("No data loaded. Exiting.")
            return
        
        # Calculate statistics
        self.calculate_delay_statistics()
        
        # Create visualizations
        self.create_visualizations()
        
        # Print summary
        self._print_summary()
        
        print("\n" + "=" * 60)
        print("Timestamp delay analysis completed successfully!")
        print("=" * 60)
        print(f"Results saved to: {self.base_output_dir}")
    
    def _print_summary(self) -> None:
        """Print summary of analysis results."""
        print("\nTimestamp Delay Analysis Summary:")
        print("-" * 40)
        
        if 'overall' in self.summary_stats:
            stats = self.summary_stats['overall']
            print(f"Total measurements: {stats['count']:,}")
            print(f"Mean delay: {stats['mean_delay_ms']:.3f} ms")
            print(f"Std deviation: {stats['std_delay_ms']:.3f} ms")
            print(f"Min delay: {stats['min_delay_ms']:.3f} ms")
            print(f"Max delay: {stats['max_delay_ms']:.3f} ms")
            print(f"Median delay: {stats['median_delay_ms']:.3f} ms")
            print(f"95th percentile: {stats['p95_delay_ms']:.3f} ms")
        
        # Temporal analysis
        if self.temporal_data:
            total_duration = sum(td['duration'] for td in self.temporal_data)
            avg_sampling_rate = np.mean([td['sampling_rate'] for td in self.temporal_data])
            total_frames = sum(td['frame_count'] for td in self.temporal_data)
            
            print(f"\nTemporal Analysis:")
            print(f"Total duration: {total_duration:.2f} seconds")
            print(f"Average sampling rate: {avg_sampling_rate:.2f} Hz")
            print(f"Total frames: {total_frames:,}")

if __name__ == "__main__":
    analyzer = TimestampDelayAnalyzer()
    analyzer.run_full_analysis()
