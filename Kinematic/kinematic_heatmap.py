import os, re, json
import numpy as np
import cv2
from tqdm import tqdm

# ---------- Configurable parameters ----------
# data_file_name: Data subfolder path (starting from Data directory)
# Full path: {current script directory}/../Data/dataset_updated/2/replay/api_cp_files/
# Modify this list to point to different data folders, e.g.:
#   ['new_experiment'] → ../Data/new_experiment/api_cp_files/
#   ['exp1', 'session2'] → ../Data/exp1/session2/api_cp_files/

data_file_name = ['dataset_updated', '2', 'replay']   # sub-folder
arm_key        = 'PSM1'                               # PSM1 / PSM2
img_h, img_w   = 1200, 1920
sigma_px       = 10
fps            = 30                                   # for dt = 1/fps

# Camera intrinsic matrix K
K = np.array([[1645.60012,     0.0,     1282.90462],
              [    0.0,     1645.60012,  610.1012],
              [    0.0,        0.0,        1.0]])

# ---------- Paths ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
single_dir  = os.path.join(parent_dir, 'Data', *data_file_name)
cp_dir      = os.path.join(single_dir, 'api_cp_files')

cp_files = sorted(
    [f for f in os.listdir(cp_dir) if f.startswith('frame') and f.endswith('.json')],
    key=lambda x: int(re.findall(r'\d+', x)[0])
)

# ---------- Utility ----------
def project_xyz_to_uv(xyz):
    """Project a 3-D point [x,y,z] to pixel coords (u,v)."""
    uvw = K @ xyz
    return uvw[0] / uvw[2], uvw[1] / uvw[2]

def d_weight(s):
    """d(s) = 1 / (1000 * s), safe-clipped to [0,1]."""
    s_min = 1e-4               # meters
    d = 1.0 / (1000.0 * max(s, s_min))
    return np.clip(d, 0.0, 1.0)

def make_heatmap(u, v, du, dv, s):
    """Two-peak Gaussian heat-map -> (H,W) float32."""
    y, x = np.mgrid[0:img_h, 0:img_w]

    primary   = np.exp(-((x - u)**2 + (y - v)**2) / sigma_px**2)
    secondary = np.exp(-((x - (u + du))**2 + (y - (v + dv))**2) / sigma_px**2)

    return primary + d_weight(s) * secondary

# ---------- Main ----------
dt = 1.0 / fps
out_dir = os.path.join(current_dir, 'heatmaps')
os.makedirs(out_dir, exist_ok=True)

for fname in tqdm(cp_files, desc="Processing frames"):
    with open(os.path.join(cp_dir, fname), 'r') as f:
        data = json.load(f)

    # 1) current 3-D pos & its pixel coords
    xyz = np.array(data[arm_key]['t'])        # meters
    u, v = project_xyz_to_uv(xyz)

    # 2) velocity -> predict next point -> (Δu,Δv)
    vel      = np.array(data[f'{arm_key}_cv']['linear'])   # m/s
    xyz_pred = xyz + vel * dt
    u_pred, v_pred = project_xyz_to_uv(xyz_pred)
    du, dv = u_pred - u, v_pred - v

    # 3) distance s for weighting
    s = np.linalg.norm(xyz) - 0.05            # subtract 5 cm offset

    # 4) generate heat-map
    heat = make_heatmap(u, v, du, dv, s).astype(np.float32)

    # 5) save
    png_path = os.path.join(out_dir, fname.replace('.json', '.png'))
    npy_path = os.path.join(out_dir, fname.replace('.json', '.npy'))

    heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(png_path, heat_norm)
    np.save(npy_path, heat)
