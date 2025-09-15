#!/usr/bin/env python3
"""
Recorder Comparison Analysis for DVRK Multimodal Data Collection

This script compares the two types of recorders:
1. Strict Match Recorder (first type) - precise alignment with 10ms error margin
2. Interpolation Recorder (second type) - stable 10Hz video with 5 candidate points

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

class RecorderComparisonAnalyzer:
    """Main class for comparing the two types of recorders."""
    
    def __init__(self, data_root: str = "../../data/data_new"):
        """
        Initialize the analyzer.
        
        Args:
            data_root: Root directory containing the data_new folder
        """
        self.data_root = Path(data_root)
        self.base_output_dir = Path("output")
        
        # Dataset processing rules
        self.dataset_rules = {
            'data_20250909': {
                'strict_match': ['strict_match/1', 'strict_match/2', 'strict_match/4'],
                'interpolation': ['interpolation/3']
            },
            'data_20250911': {
                'strict_match': ['suturing/strict_match/1', 'suturing/strict_match/2'],
                'interpolation': ['suturing/interpolation/3', 'suturing/interpolation/4']
            }
        }
        
        # Data containers
        self.strict_match_data = []
        self.interpolation_data = []
        self.comparison_stats = {}
        
        # Sensor categorization
        self.sensor_categories = {
            'image': ['header_img_left', 'header_img_right', 'header_img_side', 'image_stamp_left', 'image_stamp_right'],
            'kinematics': [
                'header_cp_set', 'header_cv', 'header_js_set', 'header_lcp', 'header_measure_cp',
                'measured_cp_stamp', 'measured_cv_stamp', 'measured_js_stamp',
                'setpoint_cp_stamp', 'setpoint_js_stamp', 'header_cv'
            ]
        }
        
        # Robot arms
        self.robot_arms = ['ECM', 'PSM1', 'PSM2', 'PSM3']
        
    def load_data(self) -> None:
        """Load data from both recorder types."""
        print("Loading data from both recorder types...")
        
        if not self.data_root.exists():
            print(f"Error: Data root directory {self.data_root} not found!")
            return
            
        # Process each dataset
        for dataset_folder in self.data_root.iterdir():
            if not dataset_folder.is_dir():
                continue
                
            dataset_name = dataset_folder.name
            print(f"\nFound dataset: {dataset_name}")
            
            if dataset_name not in self.dataset_rules:
                print(f"  Skipping {dataset_name} - no processing rules defined")
                continue
                
            # Process strict_match data
            for subfolder in self.dataset_rules[dataset_name]['strict_match']:
                subfolder_path = dataset_folder / subfolder
                if subfolder_path.exists():
                    print(f"  Processing strict_match: {subfolder}")
                    self._process_strict_match_data(dataset_name, subfolder, subfolder_path)
                else:
                    print(f"  Warning: {subfolder} not found")
            
            # Process interpolation data
            for subfolder in self.dataset_rules[dataset_name]['interpolation']:
                subfolder_path = dataset_folder / subfolder
                if subfolder_path.exists():
                    print(f"  Processing interpolation: {subfolder}")
                    self._process_interpolation_data(dataset_name, subfolder, subfolder_path)
                else:
                    print(f"  Warning: {subfolder} not found")
        
        print(f"\nLoaded {len(self.strict_match_data)} strict_match data points")
        print(f"Loaded {len(self.interpolation_data)} interpolation data points")
    
    def _process_strict_match_data(self, dataset_name: str, subfolder: str, subfolder_path: Path) -> None:
        """Process strict_match data (first type recorder)."""
        regular_path = subfolder_path / "regular"
        if not regular_path.exists():
            return
            
        for arm in self.robot_arms:
            arm_path = regular_path / "kinematic" / arm
            if not arm_path.exists():
                continue
                
            json_files = sorted(glob.glob(str(arm_path / "*.json")))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    frame_num = int(Path(json_file).stem)
                    header = data.get('header', {})
                    
                    # Use header_js_meas as baseline
                    baseline_data = header.get('header_js_meas', {})
                    if 'sec' not in baseline_data or 'nsec' not in baseline_data:
                        continue
                        
                    baseline_timestamp = baseline_data['sec'] + baseline_data['nsec'] * 1e-9
                    
                    # Calculate offsets for each sensor timestamp
                    for sensor_key, sensor_data in header.items():
                        if sensor_key == 'header_js_meas':
                            continue
                            
                        if isinstance(sensor_data, dict) and 'sec' in sensor_data and 'nsec' in sensor_data:
                            sensor_timestamp = sensor_data['sec'] + sensor_data['nsec'] * 1e-9
                            offset_ms = (baseline_timestamp - sensor_timestamp) * 1000
                            
                            category = self._get_sensor_category(sensor_key)
                            
                            self.strict_match_data.append({
                                'frame': frame_num,
                                'dataset': dataset_name,
                                'subfolder': subfolder,
                                'recorder_type': 'strict_match',
                                'arm': arm,
                                'sensor': sensor_key,
                                'category': category,
                                'offset_ms': offset_ms,
                                'baseline_timestamp': baseline_timestamp,
                                'sensor_timestamp': sensor_timestamp
                            })
                            
                except Exception as e:
                    print(f"      Error processing {json_file}: {e}")
                    continue
    
    def _process_interpolation_data(self, dataset_name: str, subfolder: str, subfolder_path: Path) -> None:
        """Process interpolation data (second type recorder)."""
        regular_path = subfolder_path / "regular"
        if not regular_path.exists():
            return
            
        for arm in self.robot_arms:
            arm_path = regular_path / "kinematic" / arm
            if not arm_path.exists():
                continue
                
            json_files = sorted(glob.glob(str(arm_path / "*.json")))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    frame_num = int(Path(json_file).stem)
                    
                    # Get the baseline timestamp from time_syn folder
                    baseline_timestamp = self._get_baseline_timestamp(dataset_name, subfolder, frame_num)
                    if baseline_timestamp is None:
                        continue
                    
                    # Process each of the 5 candidate points
                    for candidate_idx, candidate_data in enumerate(data):
                        
                        # Extract all timestamps from this candidate
                        timestamps = self._extract_all_timestamps(candidate_data)
                        
                        # Calculate offsets for each timestamp
                        for sensor_key, sensor_timestamp in timestamps.items():
                            if sensor_timestamp is None:
                                continue
                                
                            offset_ms = (sensor_timestamp - baseline_timestamp) * 1000
                            category = self._get_sensor_category(sensor_key)
                            
                            self.interpolation_data.append({
                                'frame': frame_num,
                                'candidate': candidate_idx,
                                'dataset': dataset_name,
                                'subfolder': subfolder,
                                'recorder_type': 'interpolation',
                                'arm': arm,
                                'sensor': sensor_key,
                                'category': category,
                                'offset_ms': offset_ms,
                                'baseline_timestamp': baseline_timestamp,
                                'sensor_timestamp': sensor_timestamp
                            })
                            
                except Exception as e:
                    print(f"      Error processing {json_file}: {e}")
                    continue
    
    def _extract_all_timestamps(self, candidate_data: Dict) -> Dict[str, Optional[float]]:
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
        
        # Header timestamps
        header = candidate_data.get('header', {})
        for key in ['header_cv']:
            if key in header and isinstance(header[key], dict):
                if 'sec' in header[key] and 'nsec' in header[key]:
                    timestamps[key] = header[key]['sec'] + header[key]['nsec'] * 1e-9
                else:
                    timestamps[key] = None
            else:
                timestamps[key] = None
        
        return timestamps
    
    def _get_baseline_timestamp(self, dataset_name: str, subfolder: str, frame_num: int) -> Optional[float]:
        """Get the baseline timestamp from time_syn folder for a specific frame."""
        # Find the dataset path
        dataset_path = self.data_root / dataset_name / subfolder
        time_syn_path = dataset_path / "regular" / "time_syn"
        
        if not time_syn_path.exists():
            return None
            
        # Get the specific time_syn file for this frame
        time_syn_file = time_syn_path / f"{frame_num}.json"
        
        if not time_syn_file.exists():
            return None
            
        try:
            with open(time_syn_file, 'r') as f:
                data = json.load(f)
            
            # Use image_stamp_left as baseline (as suggested by user)
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
    
    def _get_sensor_category(self, sensor_key: str) -> str:
        """Determine the category of a sensor based on its name."""
        for category, sensors in self.sensor_categories.items():
            if sensor_key in sensors:
                return category
        return 'other'
    
    def calculate_comparison_statistics(self) -> None:
        """Calculate comprehensive comparison statistics."""
        print("Calculating comparison statistics...")
        
        if not self.strict_match_data or not self.interpolation_data:
            print("Insufficient data for comparison. Please run load_data() first.")
            return
            
        strict_df = pd.DataFrame(self.strict_match_data)
        interp_df = pd.DataFrame(self.interpolation_data)
        
        # Overall statistics comparison
        self.comparison_stats['overall'] = {
            'strict_match': {
                'count': len(strict_df),
                'mean_offset_ms': strict_df['offset_ms'].abs().mean(),
                'std_offset_ms': strict_df['offset_ms'].abs().std(),
                'min_offset_ms': strict_df['offset_ms'].min(),
                'max_offset_ms': strict_df['offset_ms'].max(),
                'median_offset_ms': strict_df['offset_ms'].median(),
                'q95_offset_ms': strict_df['offset_ms'].quantile(0.95)
            },
            'interpolation': {
                'count': len(interp_df),
                'mean_offset_ms': interp_df['offset_ms'].abs().mean(),
                'std_offset_ms': interp_df['offset_ms'].abs().std(),
                'min_offset_ms': interp_df['offset_ms'].min(),
                'max_offset_ms': interp_df['offset_ms'].max(),
                'median_offset_ms': interp_df['offset_ms'].median(),
                'q95_offset_ms': interp_df['offset_ms'].quantile(0.95)
            }
        }
        
        # Category-wise comparison
        for category in ['image', 'kinematics']:
            strict_category = strict_df[strict_df['category'] == category]
            interp_category = interp_df[interp_df['category'] == category]
            
            if len(strict_category) > 0 and len(interp_category) > 0:
                self.comparison_stats[f'category_{category}'] = {
                    'strict_match': {
                        'count': len(strict_category),
                        'mean_offset_ms': strict_category['offset_ms'].abs().mean(),
                        'std_offset_ms': strict_category['offset_ms'].abs().std(),
                        'median_offset_ms': strict_category['offset_ms'].median()
                    },
                    'interpolation': {
                        'count': len(interp_category),
                        'mean_offset_ms': interp_category['offset_ms'].abs().mean(),
                        'std_offset_ms': interp_category['offset_ms'].abs().std(),
                        'median_offset_ms': interp_category['offset_ms'].median()
                    }
                }
        
        # Interpolation method comparison (mean of 5 candidates vs raw)
        if len(interp_df) > 0:
            # Calculate frame-level means for interpolation
            interp_frame_means = interp_df.groupby(['frame', 'sensor'])['offset_ms'].mean().reset_index()
            interp_frame_means['recorder_type'] = 'interpolation_mean'
            
            self.comparison_stats['interpolation_methods'] = {
                'raw_5_candidates': {
                    'count': len(interp_df),
                    'mean_offset_ms': interp_df['offset_ms'].abs().mean(),
                    'std_offset_ms': interp_df['offset_ms'].abs().std(),
                    'median_offset_ms': interp_df['offset_ms'].median()
                },
                'frame_mean': {
                    'count': len(interp_frame_means),
                    'mean_offset_ms': interp_frame_means['offset_ms'].abs().mean(),
                    'std_offset_ms': interp_frame_means['offset_ms'].abs().std(),
                    'median_offset_ms': interp_frame_means['offset_ms'].median()
                }
            }
        
        print("Comparison statistics calculated successfully.")
    
    def create_comparison_visualizations(self) -> None:
        """Create comprehensive comparison visualizations."""
        print("Creating comparison visualizations...")
        
        if not self.strict_match_data or not self.interpolation_data:
            print("Insufficient data for visualization. Please run load_data() first.")
            return
            
        strict_df = pd.DataFrame(self.strict_match_data)
        interp_df = pd.DataFrame(self.interpolation_data)
        
        # Create output directory
        plots_dir = self.base_output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Overall comparison
        self._plot_overall_comparison(strict_df, interp_df, plots_dir)
        
        # 2. Category-wise comparison
        self._plot_category_comparison(strict_df, interp_df, plots_dir)
        
        # 3. Distribution comparison
        self._plot_distribution_comparison(strict_df, interp_df, plots_dir)
        
        # 4. Temporal stability comparison
        self._plot_temporal_comparison(strict_df, interp_df, plots_dir)
        
        # 5. Interpolation methods comparison
        self._plot_interpolation_methods_comparison(interp_df, plots_dir)
        
        print("Comparison visualizations created successfully.")
    
    def _plot_overall_comparison(self, strict_df: pd.DataFrame, interp_df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot overall comparison between recorder types."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot comparison
        combined_data = []
        for _, row in strict_df.iterrows():
            combined_data.append({
                'offset_ms': row['offset_ms'],
                'recorder_type': 'Strict Match',
                'category': row['category']
            })
        for _, row in interp_df.iterrows():
            combined_data.append({
                'offset_ms': row['offset_ms'],
                'recorder_type': 'Interpolation',
                'category': row['category']
            })
        
        combined_df = pd.DataFrame(combined_data)
        
        sns.boxplot(data=combined_df, x='recorder_type', y='offset_ms', ax=ax1)
        ax1.set_title('Overall Offset Comparison')
        ax1.set_ylabel('Offset (ms)')
        
        # Violin plot for detailed distribution
        sns.violinplot(data=combined_df, x='recorder_type', y='offset_ms', ax=ax2)
        ax2.set_title('Offset Distribution Comparison (Detailed)')
        ax2.set_ylabel('Offset (ms)')
        
        # Category comparison
        sns.boxplot(data=combined_df, x='category', y='offset_ms', hue='recorder_type', ax=ax3)
        ax3.set_title('Offset Comparison by Category')
        ax3.set_ylabel('Offset (ms)')
        ax3.legend(title='Recorder Type')
        
        # Statistical summary
        stats_data = {
            'Metric': ['Mean', 'Std Dev', 'Median', '95th Percentile'],
            'Strict Match': [
                strict_df['offset_ms'].abs().mean(),
                strict_df['offset_ms'].abs().std(),
                strict_df['offset_ms'].median(),
                strict_df['offset_ms'].abs().quantile(0.95)
            ],
            'Interpolation': [
                interp_df['offset_ms'].abs().mean(),
                interp_df['offset_ms'].abs().std(),
                interp_df['offset_ms'].median(),
                interp_df['offset_ms'].abs().quantile(0.95)
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.set_index('Metric').plot(kind='bar', ax=ax4)
        ax4.set_title('Statistical Comparison')
        ax4.set_ylabel('Offset (ms)')
        ax4.tick_params(axis='x', rotation=0)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_comparison(self, strict_df: pd.DataFrame, interp_df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot category-wise comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Image category comparison
        strict_img = strict_df[strict_df['category'] == 'image']
        interp_img = interp_df[interp_df['category'] == 'image']
        
        if len(strict_img) > 0 and len(interp_img) > 0:
            img_data = []
            for _, row in strict_img.iterrows():
                img_data.append({'offset_ms': row['offset_ms'], 'recorder_type': 'Strict Match'})
            for _, row in interp_img.iterrows():
                img_data.append({'offset_ms': row['offset_ms'], 'recorder_type': 'Interpolation'})
            
            img_df = pd.DataFrame(img_data)
            sns.boxplot(data=img_df, x='recorder_type', y='offset_ms', ax=ax1)
            ax1.set_title('Image Category Comparison')
            ax1.set_ylabel('Offset (ms)')
        
        # Kinematics category comparison
        strict_kin = strict_df[strict_df['category'] == 'kinematics']
        interp_kin = interp_df[interp_df['category'] == 'kinematics']
        
        if len(strict_kin) > 0 and len(interp_kin) > 0:
            kin_data = []
            for _, row in strict_kin.iterrows():
                kin_data.append({'offset_ms': row['offset_ms'], 'recorder_type': 'Strict Match'})
            for _, row in interp_kin.iterrows():
                kin_data.append({'offset_ms': row['offset_ms'], 'recorder_type': 'Interpolation'})
            
            kin_df = pd.DataFrame(kin_data)
            sns.boxplot(data=kin_df, x='recorder_type', y='offset_ms', ax=ax2)
            ax2.set_title('Kinematics Category Comparison')
            ax2.set_ylabel('Offset (ms)')
        
        # Arm-wise comparison
        arm_data = []
        for _, row in strict_df.iterrows():
            arm_data.append({
                'offset_ms': row['offset_ms'],
                'recorder_type': 'Strict Match',
                'arm': row['arm']
            })
        for _, row in interp_df.iterrows():
            arm_data.append({
                'offset_ms': row['offset_ms'],
                'recorder_type': 'Interpolation',
                'arm': row['arm']
            })
        
        arm_df = pd.DataFrame(arm_data)
        sns.boxplot(data=arm_df, x='arm', y='offset_ms', hue='recorder_type', ax=ax3)
        ax3.set_title('Arm-wise Comparison')
        ax3.set_ylabel('Offset (ms)')
        ax3.legend(title='Recorder Type')
        
        # Sensor-wise comparison (top sensors)
        sensor_counts = pd.concat([strict_df['sensor'], interp_df['sensor']]).value_counts()
        top_sensors = sensor_counts.head(6).index
        
        sensor_data = []
        for _, row in strict_df[strict_df['sensor'].isin(top_sensors)].iterrows():
            sensor_data.append({
                'offset_ms': row['offset_ms'],
                'recorder_type': 'Strict Match',
                'sensor': row['sensor']
            })
        for _, row in interp_df[interp_df['sensor'].isin(top_sensors)].iterrows():
            sensor_data.append({
                'offset_ms': row['offset_ms'],
                'recorder_type': 'Interpolation',
                'sensor': row['sensor']
            })
        
        sensor_df = pd.DataFrame(sensor_data)
        sns.boxplot(data=sensor_df, x='sensor', y='offset_ms', hue='recorder_type', ax=ax4)
        ax4.set_title('Top Sensors Comparison')
        ax4.set_ylabel('Offset (ms)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Recorder Type')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'category_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_comparison(self, strict_df: pd.DataFrame, interp_df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot distribution comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram comparison
        ax1.hist(strict_df['offset_ms'], bins=50, alpha=0.7, label='Strict Match', density=True)
        ax1.hist(interp_df['offset_ms'], bins=50, alpha=0.7, label='Interpolation', density=True)
        ax1.set_xlabel('Offset (ms)')
        ax1.set_ylabel('Density')
        ax1.set_title('Offset Distribution Comparison')
        ax1.legend()
        
        # Absolute value histogram
        ax2.hist(strict_df['offset_ms'].abs(), bins=50, alpha=0.7, label='Strict Match', density=True)
        ax2.hist(interp_df['offset_ms'].abs(), bins=50, alpha=0.7, label='Interpolation', density=True)
        ax2.set_xlabel('Absolute Offset (ms)')
        ax2.set_ylabel('Density')
        ax2.set_title('Absolute Offset Distribution Comparison')
        ax2.legend()
        
        # Q-Q plot for distribution comparison (simplified without scipy)
        strict_sample = np.random.choice(strict_df['offset_ms'], size=min(1000, len(strict_df)), replace=False)
        interp_sample = np.random.choice(interp_df['offset_ms'], size=min(1000, len(interp_df)), replace=False)
        
        # Simple histogram comparison instead of Q-Q plot
        ax3.hist(strict_sample, bins=30, alpha=0.7, label='Strict Match Sample', density=True)
        ax3.set_title('Strict Match Sample Distribution')
        ax3.set_xlabel('Offset (ms)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.hist(interp_sample, bins=30, alpha=0.7, label='Interpolation Sample', density=True)
        ax4.set_title('Interpolation Sample Distribution')
        ax4.set_xlabel('Offset (ms)')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_comparison(self, strict_df: pd.DataFrame, interp_df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot temporal stability comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Frame-level mean comparison
        strict_frame_means = strict_df.groupby('frame')['offset_ms'].mean()
        interp_frame_means = interp_df.groupby('frame')['offset_ms'].mean()
        
        ax1.plot(strict_frame_means.index, strict_frame_means.values, label='Strict Match', alpha=0.7)
        ax1.plot(interp_frame_means.index, interp_frame_means.values, label='Interpolation', alpha=0.7)
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Mean Offset (ms)')
        ax1.set_title('Temporal Stability - Mean Offset')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frame-level std comparison
        strict_frame_std = strict_df.groupby('frame')['offset_ms'].std()
        interp_frame_std = interp_df.groupby('frame')['offset_ms'].std()
        
        ax2.plot(strict_frame_std.index, strict_frame_std.values, label='Strict Match', alpha=0.7)
        ax2.plot(interp_frame_std.index, interp_frame_std.values, label='Interpolation', alpha=0.7)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Offset Std Dev (ms)')
        ax2.set_title('Temporal Stability - Offset Variability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling mean comparison
        window = 50
        strict_rolling = strict_df.sort_values('frame').groupby('frame')['offset_ms'].mean().rolling(window=window, center=True).mean()
        interp_rolling = interp_df.sort_values('frame').groupby('frame')['offset_ms'].mean().rolling(window=window, center=True).mean()
        
        ax3.plot(strict_rolling.index, strict_rolling.values, label='Strict Match', alpha=0.7)
        ax3.plot(interp_rolling.index, interp_rolling.values, label='Interpolation', alpha=0.7)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Rolling Mean Offset (ms)')
        ax3.set_title(f'Temporal Stability - Rolling Mean (window={window})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Stability metrics comparison
        stability_metrics = {
            'Metric': ['Mean Std Dev', 'Max Std Dev', 'Temporal Variance'],
            'Strict Match': [
                strict_frame_std.mean(),
                strict_frame_std.max(),
                strict_frame_std.var()
            ],
            'Interpolation': [
                interp_frame_std.mean(),
                interp_frame_std.max(),
                interp_frame_std.var()
            ]
        }
        
        stability_df = pd.DataFrame(stability_metrics)
        stability_df.set_index('Metric').plot(kind='bar', ax=ax4)
        ax4.set_title('Temporal Stability Metrics')
        ax4.set_ylabel('Value')
        ax4.tick_params(axis='x', rotation=0)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'temporal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_interpolation_methods_comparison(self, interp_df: pd.DataFrame, plots_dir: Path) -> None:
        """Plot interpolation methods comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Raw 5 candidates vs frame mean
        interp_frame_means = interp_df.groupby(['frame', 'sensor'])['offset_ms'].mean().reset_index()
        
        # Distribution comparison
        ax1.hist(interp_df['offset_ms'], bins=50, alpha=0.7, label='Raw (5 candidates)', density=True)
        ax1.hist(interp_frame_means['offset_ms'], bins=50, alpha=0.7, label='Frame Mean', density=True)
        ax1.set_xlabel('Offset (ms)')
        ax1.set_ylabel('Density')
        ax1.set_title('Interpolation Methods Distribution')
        ax1.legend()
        
        # Box plot comparison
        methods_data = []
        for _, row in interp_df.iterrows():
            methods_data.append({'offset_ms': row['offset_ms'], 'method': 'Raw (5 candidates)'})
        for _, row in interp_frame_means.iterrows():
            methods_data.append({'offset_ms': row['offset_ms'], 'method': 'Frame Mean'})
        
        methods_df = pd.DataFrame(methods_data)
        sns.boxplot(data=methods_df, x='method', y='offset_ms', ax=ax2)
        ax2.set_title('Interpolation Methods Comparison')
        ax2.set_ylabel('Offset (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Candidate-wise analysis
        sns.boxplot(data=interp_df, x='candidate', y='offset_ms', ax=ax3)
        ax3.set_title('Offset Distribution by Candidate')
        ax3.set_xlabel('Candidate Index')
        ax3.set_ylabel('Offset (ms)')
        
        # Statistical comparison
        stats_data = {
            'Method': ['Raw (5 candidates)', 'Frame Mean'],
            'Mean': [interp_df['offset_ms'].abs().mean(), interp_frame_means['offset_ms'].abs().mean()],
            'Std': [interp_df['offset_ms'].abs().std(), interp_frame_means['offset_ms'].abs().std()],
            'Median': [interp_df['offset_ms'].median(), interp_frame_means['offset_ms'].median()]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.set_index('Method').plot(kind='bar', ax=ax4)
        ax4.set_title('Interpolation Methods Statistics')
        ax4.set_ylabel('Offset (ms)')
        ax4.tick_params(axis='x', rotation=0)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'interpolation_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_analysis(self) -> None:
        """Run the complete comparison analysis pipeline."""
        print("Starting DVRK Recorder Comparison Analysis...")
        print("=" * 60)
        
        try:
            self.load_data()
            self.calculate_comparison_statistics()
            self.create_comparison_visualizations()
            
            print("=" * 60)
            print("Comparison analysis completed successfully!")
            
            # Print summary
            if self.strict_match_data and self.interpolation_data:
                strict_df = pd.DataFrame(self.strict_match_data)
                interp_df = pd.DataFrame(self.interpolation_data)
                
                print(f"\nComparison Summary:")
                print(f"  Strict Match data points: {len(strict_df)}")
                print(f"  Interpolation data points: {len(interp_df)}")
                print(f"  Strict Match mean offset: {strict_df['offset_ms'].abs().mean():.2f} ms")
                print(f"  Interpolation mean offset: {interp_df['offset_ms'].abs().mean():.2f} ms")
                print(f"  Strict Match std deviation: {strict_df['offset_ms'].abs().std():.2f} ms")
                print(f"  Interpolation std deviation: {interp_df['offset_ms'].abs().std():.2f} ms")
                
                # Print key findings
                print(f"\nKey Findings:")
                if strict_df['offset_ms'].abs().mean() < interp_df['offset_ms'].abs().mean():
                    print(f"  ✓ Strict Match has lower mean offset")
                else:
                    print(f"  ✓ Interpolation has lower mean offset")
                
                if strict_df['offset_ms'].abs().std() < interp_df['offset_ms'].abs().std():
                    print(f"  ✓ Strict Match has lower variability")
                else:
                    print(f"  ✓ Interpolation has lower variability")
                
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Main function to run the comparison analysis."""
    analyzer = RecorderComparisonAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
