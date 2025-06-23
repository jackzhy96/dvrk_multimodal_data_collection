import os, sys
import glob
from tqdm import tqdm
import re
import argparse
import time
dynamic_path = os.path.abspath(__file__+"/../../")
# print(dynamic_path)
sys.path.append(dynamic_path)
# code_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


def extract_frame_number(filename):
    """Extract the frame number from a filename"""
    match = re.search(r'frame(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return -1


if __name__ == "__main__":
    # code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_dir',
                        default=os.path.join(dynamic_path, 'yaml','assets','data8_rectified','left'),
                        type=str, help='directory containing left images')
    parser.add_argument('--right_dir',
                        default=os.path.join(dynamic_path, 'yaml','assets','data8_rectified','right'),
                        type=str, help='directory containing right images')
    parser.add_argument('--intrinsic_file', default=os.path.join(dynamic_path, 'yaml','assets','data8_rectified','K.txt'), type=str,
                        help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=os.path.join(dynamic_path, 'pretrained_models','23-51-11','model_best_bp2.pth'), type=str,
                        help='pretrained model path')
    parser.add_argument('--out_dir', default=os.path.join(dynamic_path, 'yaml','assets','data8_rectified','output'), type=str,
                        help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int,
                        help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int,
                        help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30,
                        help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    parser.add_argument('--start_frame', type=int, default=100, help='first frame to process')
    parser.add_argument('--end_frame', type=int, default=101, help='last frame to process')
    args = parser.parse_args()

    start_time = time.time()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()

    # Get all left image files
    all_left_files = glob.glob(os.path.join(args.left_dir, "frame*"))

    # Sort by numeric frame number instead of lexicographically
    left_files = sorted(all_left_files, key=extract_frame_number)

    # Filter frames to only include those in the specified range
    filtered_left_files = []
    for file in left_files:
        frame_num = extract_frame_number(file)
        if args.start_frame <= frame_num <= args.end_frame:
            filtered_left_files.append(file)

    logging.info(f"Found {len(filtered_left_files)} frames in range {args.start_frame}-{args.end_frame} to process")

    # Preload camera intrinsic matrix (if point cloud generation is needed)
    if args.get_pc:
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
            baseline = float(lines[1])

    # Process each frame in the filtered list
    for left_file in tqdm(filtered_left_files):
        # Get frame number from filename
        frame_num = extract_frame_number(left_file)

        # Construct corresponding right image file path
        right_file = os.path.join(args.right_dir, os.path.basename(left_file))

        # Check if right file exists
        if not os.path.exists(right_file):
            logging.warning(f"Right image {right_file} does not exist, skipping this frame")
            continue

        # Create output directory for current frame
        frame_out_dir = os.path.join(args.out_dir, f"frame{frame_num}")
        os.makedirs(frame_out_dir, exist_ok=True)

        # Read images
        img0 = imageio.imread(left_file)
        img1 = imageio.imread(right_file)

        # Resize
        scale = args.scale
        assert scale <= 1, "scale must be <=1"
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        # Convert to tensors and pad
        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        # Model inference
        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        # Save visualization results
        vis = vis_disparity(disp)
        vis = np.concatenate([img0_ori, vis], axis=1)
        imageio.imwrite(f'{frame_out_dir}/vis_frame{frame_num}.png', vis)

        # Handle invisible points
        if args.remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        # Generate point cloud (if needed)
        if args.get_pc:
            K_scaled = K.copy()
            K_scaled[:2] *= scale
            depth = K_scaled[0, 0] * baseline / disp
            np.save(f'{frame_out_dir}/depth_meter_frame{frame_num}.npy', depth)

            xyz_map = depth2xyzmap(depth, K_scaled)
            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
            keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)
            o3d.io.write_point_cloud(f'{frame_out_dir}/cloud_frame{frame_num}.ply', pcd)

            # Denoise point cloud (if needed)
            if args.denoise_cloud:
                cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
                inlier_cloud = pcd.select_by_index(ind)
                o3d.io.write_point_cloud(f'{frame_out_dir}/cloud_denoise_frame{frame_num}.ply', inlier_cloud)

        logging.info(f"Processed frame {frame_num}, results saved to {frame_out_dir}")

    logging.info(f"Batch processing complete, all results saved to {args.out_dir}")

    print(f'Done! Total Runtime: {time.time()-start_time} seconds')