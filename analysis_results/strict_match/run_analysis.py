#!/usr/bin/env python3
"""
Simple runner script for the timestamp offset analysis.
This script provides a clean interface to run the analysis.
"""

import sys
from pathlib import Path

# Add current directory to path to import the main analysis module
sys.path.append(str(Path(__file__).parent))

from timestamp_offset_analysis import TimestampAnalyzer

def main():
    """Run the timestamp offset analysis."""
    print("DVRK Timestamp Offset Analysis Runner")
    print("=" * 40)
    
    # Check if data directory exists (try both relative paths)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up two levels to reach project root
    
    # Try data path from current directory first, then from project root
    data_root = None
    if (current_dir / "data" / "data_new").exists():
        data_root = str(current_dir / "data" / "data_new")
    elif (project_root / "data" / "data_new").exists():
        data_root = str(project_root / "data" / "data_new")
    else:
        print("Error: Data directory 'data/data_new' not found!")
        print("Please ensure you're running this from the project root directory or analysis_results directory.")
        print("Expected structure: data/data_new/data_20250908/2/regular/kinematic/...")
        return 1
    
    try:
        # Create analyzer and run analysis
        analyzer = TimestampAnalyzer(data_root=data_root)
        analyzer.run_full_analysis()
        
        print("\nAnalysis completed successfully!")
        print("Check the 'analysis_results/results/' directory for outputs.")
        
        return 0
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
