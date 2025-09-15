#!/usr/bin/env python3
"""
Run Interpolation Analysis

This script runs the complete interpolation analysis pipeline including:
1. Interpolation timestamp analysis
2. Recorder comparison analysis

Usage:
    python run_analysis.py
"""

import sys
import os
from pathlib import Path

# Add current directory to path so we can import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from interpolation_timestamp_analysis import InterpolationTimestampAnalyzer
from recorder_comparison_analysis import RecorderComparisonAnalyzer

def main():
    """Run the complete interpolation analysis pipeline."""
    print("=" * 60)
    print("DVRK Interpolation Analysis Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Run interpolation timestamp analysis
        print("\n1. Running Interpolation Timestamp Analysis...")
        print("-" * 50)
        interpolation_analyzer = InterpolationTimestampAnalyzer()
        interpolation_analyzer.run_full_analysis()
        
        # Step 2: Run recorder comparison analysis
        print("\n2. Running Recorder Comparison Analysis...")
        print("-" * 50)
        comparison_analyzer = RecorderComparisonAnalyzer()
        comparison_analyzer.run_full_analysis()
        
        print("\n" + "=" * 60)
        print("Analysis Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"Results saved to: {current_dir / 'output'}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
