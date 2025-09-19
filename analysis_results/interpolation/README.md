# DVRK Interpolation Analysis

This folder contains the **timestamp delay analysis** script for the interpolation-based DVRK data recorder.

## Files

- `timestamp_delay_analysis.py` - **Main script** for timestamp delay analysis

## Output

The analysis generates outputs organized by dataset (following strict_match pattern):

### Structure
```
output/
├── data_new/                    # Dataset-specific outputs
│   ├── data_20250909/
│   │   └── interpolation/
│   │       └── 3/
│   │           ├── plots/       # Dataset-specific plots
│   │           ├── detailed_delay_data.csv
│   │           ├── summary_statistics.json
│   │           └── detailed_analysis_report.html
│   └── data_20250911/
│       └── suturing/
│           └── interpolation/
│               ├── 3/           # Same structure as above
│               └── 4/           # Same structure as above
└── overall/                     # Combined analysis
    ├── plots/                   # Overall plots
    ├── detailed_delay_data.csv
    ├── summary_statistics.json
    └── detailed_analysis_report.html
```

## Usage

### Run Complete Analysis Pipeline
```bash
python run_analysis.py
```

### Run Individual Analysis
```bash
# Timestamp delay analysis (main focus)
python timestamp_delay_analysis.py

# Recorder comparison analysis
python recorder_comparison_analysis.py
```

## Data Sources

The analysis processes data from:
- `data_20250909/interpolation/3`
- `data_20250911/suturing/interpolation/3`
- `data_20250911/suturing/interpolation/4`

## Key Features

1. **Timestamp Delay Analysis**: Analyzes the 5-candidate interpolation data structure
2. **Baseline Correction**: Uses `time_syn` folder's `image_stamp_left` as baseline
3. **Organized Output**: Dataset-specific outputs following strict_match pattern
4. **Comprehensive Visualizations**: Distribution plots, frame-level analysis, sensor comparisons
5. **Recorder Comparison**: Direct comparison with strict_match recorder performance
6. **Temporal Analysis**: Sampling rate analysis and time stability assessment

## Results Summary

- **Total data points**: 237,690
- **Mean delay**: -0.088 ms
- **Std deviation**: 1.606 ms
- **95th percentile**: 2.392 ms
- **Sampling rate**: 57.21 Hz (average)
- **Total duration**: 119.70 seconds
- **Total frames**: 2,641
