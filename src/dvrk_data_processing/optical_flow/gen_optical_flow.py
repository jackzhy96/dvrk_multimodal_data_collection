import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import re

# ---------- Configurable parameters ----------
# dataset_path: Dataset subfolder path (starting from data directory)
# Full path: {current script directory}/../data/Dataset8/normal/
# Modify this list to point to different datasets, e.g.:
#   ['Dataset9', 'normal'] → ../data/Dataset9/normal/

dataset_path = ['Dataset8', 'normal']  # Dataset subfolder path
output_suffix = 'optical_flow'  # Output folder suffix
flow_format = 'npy'  # Output format: npy/flo
enable_visualization = True  # Generate visualization images

# ---------- Optical Flow parameters ----------
pyramid_scale = 0.5  # Pyramid scale factor
pyramid_levels = 3  # Number of pyramid levels
window_size = 15  # Window size
iterations = 3  # Number of iterations
poly_n = 5  # Size of pixel neighborhood for polynomial expansion
poly_sigma = 1.2  # Standard deviation for Gaussian
flags = 0  # Additional flags

# ---------- Image preprocessing parameters ----------
bilateral_d = 9  # Bilateral filter diameter
bilateral_sigma_color = 75  # Bilateral filter sigma color
bilateral_sigma_space = 75  # Bilateral filter sigma space
gaussian_kernel_size = (5, 5)  # Gaussian blur kernel size
gaussian_sigma = 1.2  # Gaussian blur sigma

# ---------- Paths ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
base_dir = os.path.join(parent_dir, 'data', *dataset_path)

output_dir = os.path.join(parent_dir, 'output', dataset_path[0], f"{dataset_path[1]}_{output_suffix}")


print(f"Base directory: {base_dir}")
print(f"Output directory: {output_dir}")


def sort_by_frame_number(file_path):
    """Extract frame number for sorting"""
    basename = os.path.basename(file_path)
    match = re.search(r'frame(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0


def calculate_optical_flow(img1_path, img2_path, output_path=None, format="npy"):
    """Calculate optical flow between two frames and save to file"""
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise ValueError(f"Could not read images: {img1_path}, {img2_path}")

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Step 1: Reduce speckle noise
    gray1 = cv2.bilateralFilter(gray1, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    gray2 = cv2.bilateralFilter(gray2, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    # Step 2: Light Gaussian smoothing
    gray1 = cv2.GaussianBlur(gray1, gaussian_kernel_size, gaussian_sigma)
    gray2 = cv2.GaussianBlur(gray2, gaussian_kernel_size, gaussian_sigma)

    # Calculate optical flow using configurable parameters
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,  # Initial flow estimate, None means start from zero
        pyramid_scale,  # Pyramid scale factor
        pyramid_levels,  # Number of pyramid levels
        window_size,  # Window size
        iterations,  # Number of iterations
        poly_n,  # Size of pixel neighborhood for polynomial expansion
        poly_sigma,  # Standard deviation for Gaussian
        flags  # Additional flags
    )

    # Save flow data
    if output_path:
        if format.lower() == "npy":
            np.save(output_path, flow)
        elif format.lower() == "flo":
            # Write .flo file (Middlebury format)
            with open(output_path, 'wb') as f:
                np.array([202021.25], dtype=np.float32).tofile(f)  # Magic number
                np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)  # Width, Height
                flow.astype(np.float32).tofile(f)
        # Add more formats as needed

    return flow


def visualize_flow(flow, original_img, output_path=None):
    """Visualize optical flow and save the result"""
    # Calculate magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV representation of flow
    hsv = np.zeros_like(original_img)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2  # Map angle to hue
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Map magnitude to value

    # Convert to BGR for display
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if output_path:
        cv2.imwrite(output_path, flow_img)

    return flow_img


def process_camera_sequence(frames, camera_name, output_dir, vis_dir=None):
    """Process a single camera sequence and calculate optical flows"""
    print(f"Calculating {camera_name} camera optical flow...")

    for i in tqdm(range(len(frames) - 1), desc=f"{camera_name} camera"):
        frame_t_path = frames[i]
        frame_t1_path = frames[i + 1]

        # Extract frame numbers
        frame_t_num = str(int(os.path.splitext(os.path.basename(frame_t_path))[0].replace("frame", ""))).zfill(3)
        frame_t1_num = str(int(os.path.splitext(os.path.basename(frame_t1_path))[0].replace("frame", ""))).zfill(3)

        # Set output path
        flow_output_path = os.path.join(output_dir, f"{camera_name}_flow_{frame_t_num}_{frame_t1_num}.{flow_format}")

        try:
            # Calculate and save optical flow
            flow = calculate_optical_flow(frame_t_path, frame_t1_path, flow_output_path, flow_format)

            # Visualize flow (optional)
            if enable_visualization and vis_dir:
                original_img = cv2.imread(frame_t_path)
                vis_output_path = os.path.join(vis_dir, f"{camera_name}_flow_{frame_t_num}_{frame_t1_num}.png")
                visualize_flow(flow, original_img, vis_output_path)

        except Exception as e:
            print(f"Error processing {camera_name} frames {frame_t_num}-{frame_t1_num}: {e}")
            continue

    return len(frames) - 1


def process_sequence():
    """Process the entire image sequence and calculate all optical flows"""
    # Verify base directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = None
    if enable_visualization:
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)

    # Process left camera sequence
    left_dir = os.path.join(base_dir, "left_frames")
    if not os.path.exists(left_dir):
        print(f"Warning: Left frames directory not found: {left_dir}")
        left_frames = []
    else:
        left_frames = sorted(glob.glob(os.path.join(left_dir, "frame*.png")), key=sort_by_frame_number)

    # Process right camera sequence
    right_dir = os.path.join(base_dir, "right_frames")
    if not os.path.exists(right_dir):
        print(f"Warning: Right frames directory not found: {right_dir}")
        right_frames = []
    else:
        right_frames = sorted(glob.glob(os.path.join(right_dir, "frame*.png")), key=sort_by_frame_number)

    print(f"Found {len(left_frames)} left camera frames and {len(right_frames)} right camera frames")

    # Process sequences
    left_flow_count = 0
    right_flow_count = 0

    if left_frames:
        left_flow_count = process_camera_sequence(left_frames, "left", output_dir, vis_dir)

    if right_frames:
        right_flow_count = process_camera_sequence(right_frames, "right", output_dir, vis_dir)

    # Summary
    total_flows = left_flow_count + right_flow_count
    print(f"\nProcessing complete! Generated {total_flows} optical flow files")
    print(f"Left camera flows: {left_flow_count}")
    print(f"Right camera flows: {right_flow_count}")
    print(f"Flow files saved to: {output_dir}")
    if enable_visualization:
        print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("OPTICAL FLOW PROCESSING")
    print("=" * 60)
    print(f"Dataset: {' -> '.join(dataset_path)}")
    print(f"Output format: {flow_format}")
    print(f"Visualization: {'Enabled' if enable_visualization else 'Disabled'}")
    print(f"Optical flow parameters:")
    print(f"  - Pyramid scale: {pyramid_scale}")
    print(f"  - Pyramid levels: {pyramid_levels}")
    print(f"  - Window size: {window_size}")
    print(f"  - Iterations: {iterations}")
    print("=" * 60)

    try:
        process_sequence()
        print("\n All processing completed successfully!")
    except Exception as e:
        print(f"\n Error during processing: {e}")
        raise