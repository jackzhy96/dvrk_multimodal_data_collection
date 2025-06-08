# SurgPoseV1
Upgraded Version of SurgPose: A Dataset for Articulated Robotic Surgical Tool Pose Estimation, Tracking and Tool-Tissue On-Contact

# Folder structure

## Optical Flow 

Calculates optical flow between consecutive frames.

### What it does
- Reads image sequences from left/right camera folders
- Calculates motion between consecutive frames
- Saves optical flow data and visualizations

### Setup
1. Put your images in: `data/dataset_name/normal/left_frames/` and `right_frames/`
2. Configure `dataset_path` in the script
3. Run: `python optical_flow_script.py`

### Output
Files saved to: `output/dataset_name/normal_optical_flow/`
- `.npy` files: Raw optical flow data
- `visualization/`: Flow visualization images

## Kinematic Heatmap Generator 

Generates heatmaps from robot motion data.

### What it does
- Reads JSON files with robot position/velocity data
- Creates heatmaps showing current position + predicted next position
- Saves PNG images and NPY data files

### Setup
1. Put your JSON files in: `Data/dataset_name/api_cp_files/`
2. Configure `data_file_name` in the script
3. Run: `python heatmap_script.py`

### Output
Files saved to: `output/dataset_name/kinematic_heatmap/`
