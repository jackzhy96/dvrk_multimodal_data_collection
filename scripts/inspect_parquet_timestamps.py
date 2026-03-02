#!/usr/bin/env python3
"""
Quick script to inspect timestamps in parquet files.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def inspect_parquet_timestamps(parquet_file: Path):
    """Inspect timestamp column in a parquet file."""
    print(f"\n{'='*70}")
    print(f"Inspecting: {parquet_file.name}")
    print(f"{'='*70}")

    # Read parquet file
    df = pd.read_parquet(parquet_file)

    # Check if timestamp column exists
    if 'timestamp' not in df.columns:
        print("❌ No 'timestamp' column found!")
        return

    # Extract timestamps
    timestamps = df['timestamp'].values

    # Handle array-stored values (extract scalar if needed)
    if hasattr(timestamps[0], '__len__') and not isinstance(timestamps[0], str):
        timestamps = np.array([t[0] if len(t) > 0 else 0.0 for t in timestamps])

    print(f"\n📊 Timestamp Statistics:")
    print(f"  Number of frames: {len(timestamps)}")
    print(f"  First timestamp:  {timestamps[0]:.6f} seconds")
    print(f"  Last timestamp:   {timestamps[-1]:.6f} seconds")
    print(f"  Duration:         {timestamps[-1] - timestamps[0]:.6f} seconds")
    print(f"  Min timestamp:    {np.min(timestamps):.6f} seconds")
    print(f"  Max timestamp:    {np.max(timestamps):.6f} seconds")

    # Check if it starts from 0
    if abs(timestamps[0]) < 0.001:  # Within 1ms of zero
        print(f"  ✅ Starts from ~0 (relative to episode start)")
    else:
        print(f"  ⚠️  Does NOT start from 0 (starts from {timestamps[0]:.6f})")

    # Show first few and last few timestamps
    print(f"\n📝 First 5 timestamps:")
    for i, ts in enumerate(timestamps[:5]):
        print(f"    Frame {i}: {ts:.6f} seconds")

    if len(timestamps) > 5:
        print(f"\n📝 Last 5 timestamps:")
        for i, ts in enumerate(timestamps[-5:], start=len(timestamps)-5):
            print(f"    Frame {i}: {ts:.6f} seconds")

    # Calculate time deltas between frames
    if len(timestamps) > 1:
        deltas = np.diff(timestamps)
        print(f"\n⏱️  Frame-to-frame time deltas:")
        print(f"  Mean delta:   {np.mean(deltas):.6f} seconds ({1/np.mean(deltas):.2f} Hz)")
        print(f"  Min delta:    {np.min(deltas):.6f} seconds")
        print(f"  Max delta:    {np.max(deltas):.6f} seconds")
        print(f"  Std delta:    {np.std(deltas):.6f} seconds")

    # Also show other metadata columns if they exist
    print(f"\n📋 Available columns in parquet:")
    meta_cols = [col for col in df.columns if 'meta' in col or col in ['timestamp', 'frame_id', 'episode_index']]
    for col in sorted(meta_cols):
        print(f"  - {col}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_parquet_timestamps.py <parquet_file>")
        print("\nExample:")
        print("  python inspect_parquet_timestamps.py data/output/data/case-000/episode_000000.parquet")
        sys.exit(1)

    parquet_path = Path(sys.argv[1])

    if not parquet_path.exists():
        print(f"❌ File not found: {parquet_path}")
        sys.exit(1)

    inspect_parquet_timestamps(parquet_path)
    print(f"\n{'='*70}\n")
