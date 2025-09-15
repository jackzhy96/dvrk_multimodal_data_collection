# DVRK Timestamp Offset Analysis

This analysis examines timestamp offsets in synchronized multimodal data from the DVRK (da Vinci Research Kit) system.

## Overview

This repository contains analysis tools for two types of DVRK data recorders:
- **Strict Match Recorder**: High precision, low latency, variable frame rate (3-8Hz)
- **Interpolation Recorder**: Fixed 10Hz frame rate, stable video stream, 5-candidate interpolation

Each JSON file represents a synchronized frame containing timestamps from different sensors. The main timestamp serves as the reference, and we calculate offsets for each sensor timestamp relative to this reference.

## Data Structure

- **Dataset 1**: dVRK Camera system (792 frames)
- **Dataset 2**: CSR Camera system (458 frames)
- **Robot Arms**: ECM, PSM1, PSM2
- **Sensor Categories**:
  - **Image**: Left/Right/Side camera timestamps
  - **Jaw**: Gripper measurement/setpoint timestamps
  - **Robot Control**: Joint space, Cartesian space, and local CP timestamps

## Analysis Components

### 1. Statistical Analysis
- Mean, standard deviation, min, max, median offsets
- 25th, 75th, 95th percentiles
- Comparison across camera systems, robot arms, and sensor types

### 2. Visualizations
- Overall offset distributions
- Camera system comparisons
- Robot arm comparisons
- Sensor category analysis
- Individual sensor analysis
- Temporal stability analysis

### 3. Output Files
- `summary_statistics.json`: Detailed statistical results
- `detailed_offset_data.csv`: Raw offset data
- `detailed_analysis_report.html`: Comprehensive HTML report
- `plots/`: Directory containing all visualization plots

## Folder Structure

- `strict_match/` - Analysis tools for strict_match recorder data
- `interpolation/` - Analysis tools for interpolation recorder data

## Usage

### Strict Match Analysis
```bash
cd analysis_results/strict_match
python run_analysis.py
```

### Interpolation Analysis
```bash
cd analysis_results/interpolation
python run_analysis.py
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- pathlib

## Results Interpretation

- **Positive offsets**: Sensor timestamp is earlier than main timestamp
- **Negative offsets**: Sensor timestamp is later than main timestamp
- **Small absolute values**: Better synchronization
- **Large standard deviations**: Higher timing jitter

## Key Metrics

- **Mean Offset**: Systematic bias in timing
- **Standard Deviation**: Timing jitter/variability
- **95th Percentile**: Worst-case timing error
- **Temporal Stability**: Consistency over time
