# DVRK Multimodal Data Collection - Analysis Suite

This directory contains comprehensive analysis tools for DVRK multimodal data collection, supporting two different data recorder formats.

## Overview

The analysis suite includes three main components:

1. **Original Timestamp Analysis** - Analyzes data from the original recorder (strict_match format)
2. **Interpolation Data Analysis** - Analyzes data from the interpolation recorder (array-based format)  
3. **Recorder Comparison** - Compares performance between the two recorders

## Data Formats

### Original Recorder (strict_match)
- **Location**: `data/data_new/data_20250909/strict_match/`
- **Format**: Each JSON file contains a single object with `header` field containing sensor timestamps
- **Reference**: `header_js_meas` timestamp used as baseline
- **Structure**: Flat hierarchy with all timestamps in header

### Interpolation Recorder (interpolation)
- **Location**: `data/data_new/data_20250909/interpolation/`
- **Format**: Each JSON file contains an array of data points with multiple timestamps
- **Reference**: `measured_js_stamp` timestamp used as baseline for each data point
- **Structure**: Nested hierarchy with timestamps in different sections (arm, header, jaw)

## Analysis Scripts

### 1. Original Timestamp Analysis
**File**: `timestamp_offset_analysis.py`

Analyzes timestamp offsets in the original data format.

**Features**:
- Calculates timestamp offsets relative to baseline
- Analyzes by robot arm, sensor category, and individual sensors
- Temporal stability analysis
- Comprehensive visualizations

**Usage**:
```bash
python timestamp_offset_analysis.py
```

**Output**: `output/data_new/`

### 2. Interpolation Data Analysis
**File**: `interpolation_timestamp_analysis.py`

Analyzes timestamp patterns in the interpolation data format.

**Features**:
- Multi-point timestamp analysis per frame
- Frame-based data density analysis
- Sensor category analysis (measured, setpoint, header, jaw, image)
- Temporal stability across frames
- Data completeness analysis

**Usage**:
```bash
python interpolation_timestamp_analysis.py
```

**Output**: `output/interpolation_analysis/`

### 3. Recorder Comparison
**File**: `recorder_comparison_analysis.py`

Compares performance between the two data recorders.

**Features**:
- Side-by-side statistical comparison
- Offset distribution comparison
- Temporal stability comparison
- Data density analysis
- Performance metrics comparison

**Usage**:
```bash
python recorder_comparison_analysis.py
```

**Output**: `output/recorder_comparison/`

### 4. Run All Analyses
**File**: `run_all_analysis.py`

Executes all analysis scripts in sequence.

**Usage**:
```bash
python run_all_analysis.py
```

## Output Structure

```
output/
├── data_new/                    # Original recorder analysis
│   ├── data_20250908/
│   ├── data_20250909/
│   └── overall/
├── interpolation_analysis/      # Interpolation recorder analysis
│   ├── plots/
│   ├── summary_statistics.json
│   ├── detailed_timestamp_data.csv
│   └── detailed_analysis_report.html
└── recorder_comparison/         # Comparison analysis
    ├── plots/
    ├── comparison_statistics.json
    ├── original_recorder_data.csv
    ├── interpolation_recorder_data.csv
    └── comparison_analysis_report.html
```

## Key Analysis Metrics

### Timestamp Accuracy
- **Mean Offset**: Average timestamp offset in milliseconds
- **Standard Deviation**: Consistency of timestamp offsets
- **95th Percentile**: Upper bound for 95% of offsets
- **Range**: Min/max offset values

### Data Quality
- **Completeness**: Percentage of expected data present
- **Consistency**: Stability of timestamp patterns over time
- **Density**: Number of data points per frame/arm/sensor

### Temporal Stability
- **Rolling Mean**: Moving average of offsets over time
- **Rolling Std**: Moving standard deviation of offsets
- **Trend Analysis**: Long-term patterns in timestamp accuracy

## Visualization Types

1. **Distribution Plots**: Histograms and box plots of timestamp offsets
2. **Temporal Plots**: Time-series analysis of offset trends
3. **Comparison Plots**: Side-by-side comparisons between recorders
4. **Category Analysis**: Breakdown by sensor type and robot arm
5. **Density Analysis**: Data volume and completeness metrics

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- pathlib

## Usage Examples

### Quick Start
```bash
# Run all analyses
python run_all_analysis.py
```

### Individual Analysis
```bash
# Analyze original data only
python timestamp_offset_analysis.py

# Analyze interpolation data only  
python interpolation_timestamp_analysis.py

# Compare both recorders
python recorder_comparison_analysis.py
```

### Custom Analysis
You can modify the analysis parameters by editing the respective script files:

- **Data paths**: Modify `data_root` parameter in each script
- **Robot arms**: Update `robot_arms` list to include/exclude specific arms
- **Sensor categories**: Modify `sensor_categories` dictionary
- **Visualization settings**: Adjust matplotlib/seaborn parameters

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure data paths are correct and data exists
2. **Memory errors**: For large datasets, consider processing in chunks
3. **Empty results**: Check that JSON files contain expected timestamp fields
4. **Visualization errors**: Ensure output directories exist and are writable

### Data Validation

Before running analysis, verify:
- JSON files are valid and readable
- Expected timestamp fields are present
- Data structure matches expected format
- File naming convention is consistent

## Results Interpretation

### Good Performance Indicators
- Low mean offset (< 10ms)
- Low standard deviation (< 5ms)
- Stable temporal patterns
- High data completeness (> 95%)

### Potential Issues
- High offset variance may indicate synchronization problems
- Missing data patterns may suggest recording issues
- Temporal drift may indicate clock synchronization problems
- Large differences between recorders may indicate system changes

## Contributing

To add new analysis features:
1. Follow the existing class structure
2. Add comprehensive error handling
3. Include visualization methods
4. Update this README with new features
5. Test with sample data before deployment
