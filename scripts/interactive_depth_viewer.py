import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

depth_data = None

def gen_depth_from_disp(disp: np.float32)->float:
    '''
    Generate depth from disparity
    disp : raw disparity value
    return : depth value in meters

    Hard-coded fx and baseline
    '''
    disp_float = float(disp)
    depth = 393.3557365890108 * 0.004344699 / disp_float * 1000
    return depth


def load_depth_file(depth_dir:str, idx_frame:int)->None:
    '''
    depth_dir : directory of the parent folder
    idx_frame : index of the selected frame
    Load the depth file
    '''
    global depth_data
    depth_path = os.path.join(depth_dir, 'disparity', f'{str(idx_frame)}.npy')
    depth_data = np.load(depth_path)



def custom_coord(x: float, y: float) -> str:
    '''
    x : x coordinate
    y : y coordinate
    depth_map: given depth map
    redefine the matplot coordinate formate
    '''
    x = int(x)
    y = int(y)
    if 0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]:
        z = gen_depth_from_disp(depth_data[y, x])
        return f'x={x}, y={y}, depth={z: .2f}mm'
    else:
        return f'x={x}, y={y}'


def plot_depth_map(idx_frame:int)->None:
    '''
    idx_frame : index of the selected frame
    Plot the depth map
    '''
    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.imshow(depth_data, cmap='plasma', vmin=np.percentile(depth_data, 1), vmax=np.percentile(depth_data, 99))
    plt.colorbar(img, label='Depth (mm)')
    ax.set_title(f'Interactive Depth Viewer Frame {idx_frame}')
    ax.axis('off')
    ax.format_coord = custom_coord
    plt.show()


if __name__ == '__main__':
    test_path = '/home/jackzhy/dvrk_multimodal_data_collection/data/output/data_20250828/suturing1/7/depth_estimation'
    parser = argparse.ArgumentParser(description="Interactive Depth Viewer")
    parser.add_argument("--depth_dir", type=str, help="Path of the depth estimation generated folder", default=test_path)
    parser.add_argument("--idx_frame", type=str, help="frame index", default=340)
    args = parser.parse_args()

    depth_dir = args.depth_dir
    idx_frame = int(args.idx_frame)

    try:
        load_depth_file(depth_dir, idx_frame)
        plot_depth_map(idx_frame)
    except Exception as e:
        print(e)

