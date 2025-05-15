import os, sys
import glob
from tqdm import tqdm
import re
import time
import argparse
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
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_dir',
                        default=os.path.join(dynamic_path, 'yaml','assets','data8_rectified','left'),
                        type=str, help='directory containing left images')
    parser.add_argument('--right_dir',
                        default=os.path.join(dynamic_path, 'yaml','assets','data8_rectified','right'),
                        type=str, help='directory containing right images')
    parser.add_argument('--ckpt_dir', default=os.path.join(dynamic_path, 'pretrained_models','23-51-11','model_best_bp2.pth'), type=str,
                        help='pretrained model path')
    parser.add_argument('--out_dir', default=os.path.join(dynamic_path, 'yaml','assets','data8_rectified','output_light'), type=str,
                        help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=1, type=int,
                        help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--save_depth', type=int, default=1, help='save depth map output')
    parser.add_argument('--start_frame', type=int, default=100, help='first frame to process')
    parser.add_argument('--end_frame', type=int, default=101, help='last frame to process')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load configuration and model
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

    # Initialize timing statistics
    total_start_time = time.time()
    frame_times = []

    # Process each frame in the filtered list
    for left_file in tqdm(filtered_left_files):
        frame_start_time = time.time()

        # Get frame number from filename
        frame_num = extract_frame_number(left_file)

        # Construct corresponding right image file path
        right_file = os.path.join(args.right_dir, os.path.basename(left_file))

        # Check if right file exists
        if not os.path.exists(right_file):
            logging.warning(f"Right image {right_file} does not exist, skipping this frame")
            continue

        # Read images
        img0 = imageio.imread(left_file)
        img1 = imageio.imread(right_file)

        # Resize
        scale = args.scale
        assert scale <= 1, "scale must be <=1"

        resize_start = time.time()
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
        resize_time = time.time() - resize_start

        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        # Convert to tensors and pad
        tensor_start = time.time()
        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        tensor_time = time.time() - tensor_start

        # Model inference
        inference_start = time.time()
        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        inference_time = time.time() - inference_start

        post_start = time.time()
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)
        post_time = time.time() - post_start

        # Save visualization results
        save_start = time.time()
        vis = vis_disparity(disp)
        vis = np.concatenate([img0_ori, vis], axis=1)

        # Create output directory for current frame
        frame_out_dir = os.path.join(args.out_dir, f"frame{frame_num}")
        os.makedirs(frame_out_dir, exist_ok=True)
        imageio.imwrite(f'{frame_out_dir}/vis_frame{frame_num}.png', vis)

        # Optionally save depth map
        if args.save_depth:
            np.save(f'{frame_out_dir}/disp_frame{frame_num}.npy', disp)

        save_time = time.time() - save_start

        # Clear GPU memory
        cleanup_start = time.time()
        torch.cuda.empty_cache()
        cleanup_time = time.time() - cleanup_start

        # Log frame processing time
        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)

        logging.info(f"Frame {frame_num} processed in {frame_time:.2f}s (resize: {resize_time:.2f}s, "
                     f"tensor: {tensor_time:.2f}s, inference: {inference_time:.2f}s, "
                     f"post: {post_time:.2f}s, save: {save_time:.2f}s, cleanup: {cleanup_time:.2f}s)")

    # Log overall statistics
    total_time = time.time() - total_start_time
    avg_time = sum(frame_times) / len(frame_times) if frame_times else 0

    logging.info(f"")
    logging.info(f"Batch processing complete. All results saved to {args.out_dir}")
    logging.info(f"Total time: {total_time:.2f}s for {len(filtered_left_files)} frames")
    logging.info(f"Average time per frame: {avg_time:.2f}s ({1 / avg_time:.2f} FPS)")
    logging.info(f"Min frame time: {min(frame_times):.2f}s, Max frame time: {max(frame_times):.2f}s")