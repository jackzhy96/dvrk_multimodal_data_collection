import json, re, sys, ctypes, os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2, numpy as np, yaml
from tqdm import tqdm

import pathlib
proj_root = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(str(proj_root / "src"))

from dvrk_data_processing.utils.utility import create_folder, glob_sorted_frame

#config
def find_config_kp() -> Path:
    for parent in Path(__file__).resolve().parents:
        p = parent / "config" / "config_kp.yaml"
        if p.exists():
            return p
    sys.exit("[Error] config/config_kp.yaml not found")

def parse_path_config(cfg_kp: Path) -> Dict[str, str]:
    kp_yaml = yaml.safe_load(cfg_kp.read_text("utf-8"))
    pc_name = next((d["path_config"] for d in kp_yaml.get("defaults", [])
                    if "path_config" in d), None)
    if pc_name is None:
        sys.exit("[Error] defaults.path_config missing in config_kp.yaml")

    pc_file = cfg_kp.parent / "path_config" / f"{pc_name}.yaml"
    if not pc_file.exists():
        sys.exit(f"[Error] path_config file missing: {pc_file}")

    raw = yaml.safe_load(pc_file.read_text("utf-8"))
    patt = re.compile(r"\$\{\.(\w+)\}")

    def resolve(v: str):
        while isinstance(v, str) and patt.search(v):
            v = patt.sub(lambda m: str(raw[m.group(1)]), v)
        return v

    return {k: resolve(v) for k, v in raw.items()}

#UI helpers
SIDEBAR_W = 360                      # pixels

def build_sidebar(label: str,
                  frame_txt: str,
                  ds_txt: str,
                  height: int,
                  font_scale: float = 0.8) -> np.ndarray:
    """Return a white sidebar with instructions + meta info."""
    sb = 255 * np.ones((height, SIDEBAR_W, 3), dtype=np.uint8)
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 1
    lines = [
        "dVRK Contact Annotation",
        "-" * 30,
        frame_txt,
        ds_txt,
        f"Label: {label}",
        "",
        "C : contact (1)",
        "N : no-contact (0)",
        "S : skip this arm",
        "Q : quit",
    ]
    y = 34
    for ln in lines:
        cv2.putText(sb, ln, (10, y), font, font_scale,
                    (0, 0, 0), th, cv2.LINE_AA)
        y += int(46 * font_scale)
    return sb

def pad_to_equal(imgs: List[np.ndarray], axis: int) -> List[np.ndarray]:
    """Pad images so they share same width (axis=1) or height (axis=0)."""
    if axis == 1:                             # equalise widths
        tgt = max(im.shape[1] for im in imgs)
        out = []
        for im in imgs:
            d = tgt - im.shape[1]
            l, r = d // 2, d - d // 2
            out.append(cv2.copyMakeBorder(im, 0, 0, l, r,
                                          cv2.BORDER_CONSTANT, value=0))
        return out
    else:                                     # equalise heights
        tgt = max(im.shape[0] for im in imgs)
        out = []
        for im in imgs:
            d = tgt - im.shape[0]
            t, b = d // 2, d - d // 2
            out.append(cv2.copyMakeBorder(im, t, b, 0, 0,
                                          cv2.BORDER_CONSTANT, value=0))
        return out

# main annotator
def annotate(raw_dir: Path,
             vertical: bool,
             auto_fit: bool,
             data_name: str,
             data_index: str):

    # camera folders
    img_root = raw_dir / "regular" / "image"
    left_dir, right_dir, side_dir = [img_root / c for c in ("left", "right", "side")]
    for d in (left_dir, right_dir, side_dir):
        if not d.exists():
            sys.exit(f"[Error] missing camera folder: {d}")

    out_dir = raw_dir / "regular" / "kinematic" / "contact"
    create_folder(out_dir)
    frames = glob_sorted_frame(left_dir)

    # screen size (Windows)
    user32 = ctypes.windll.user32
    scr_w, scr_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    cv2.namedWindow("Annot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Annot", cv2.WND_PROP_TOPMOST, 1)

    arms: List[Tuple[str, str]] = [
        ("PSM1", "right arm (PSM1)"),
        ("PSM2", "left arm  (PSM2)"),
    ]

    for idx, f in enumerate(frames, start=1):
        stem = f.stem
        imgs = [
            cv2.imread(str(left_dir  / (stem + f.suffix))),
            cv2.imread(str(right_dir / (stem + f.suffix))),
            cv2.imread(str(side_dir  / (stem + f.suffix))),
        ]
        if any(im is None for im in imgs):
            continue

        if vertical:                                          # 3-stack
            cam_cat = cv2.vconcat(pad_to_equal(imgs, axis=1))
        else:                                                 # 2-column
            col_left  = cv2.vconcat(pad_to_equal([imgs[0], imgs[2]], axis=1))
            cams      = pad_to_equal([col_left, imgs[1]], axis=0)
            cam_cat   = cv2.hconcat(cams)

        #auto-fit
        scale = 0.5
        if auto_fit:
            max_w = scr_w - SIDEBAR_W - 40
            max_h = scr_h - 120
            scale = min(1.0,
                        max_w / cam_cat.shape[1],
                        max_h / cam_cat.shape[0])
            if scale < 1.0:
                cam_cat = cv2.resize(cam_cat, None,
                                     fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)

        # load or init json for this frame
        lbl_path = out_dir / f"{stem}.json"
        labels   = json.load(open(lbl_path)) if lbl_path.exists() else {}

        frame_txt   = f"Frame: {idx}"
        dataset_txt = f"Dataset: {data_name} / {data_index}"

        # loop over arms (PSM1 → PSM2)
        for key, desc in arms:
            if key in labels:
                continue

            base_font = 0.15
            font = base_font / scale if auto_fit else base_font

            sidebar = build_sidebar(desc,
                                    frame_txt,
                                    dataset_txt,
                                    cam_cat.shape[0],
                                    font_scale=font)
            canvas = cv2.hconcat([cam_cat, sidebar])

            cv2.setWindowTitle("Annot", f"{stem} | {key}")
            cv2.imshow("Annot", canvas)

            while True:
                k = cv2.waitKey(0) & 0xFF
                if k in (ord('c'), ord('C')):
                    labels[key] = 1; break
                if k in (ord('n'), ord('N')):
                    labels[key] = 0; break
                if k in (ord('s'), ord('S')):
                    labels[key] = None; break
                if k in (ord('q'), ord('Q')):
                    cv2.destroyAllWindows()
                    print("\nQuit – progress saved.")
                    return

        json.dump(labels, open(lbl_path, "w"), indent=2)

    cv2.destroyAllWindows()

# CLI 
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("dVRK 3-camera, two-arm annotator")
    ap.add_argument("--horizontal", action="store_true",
                    help="use 2-column layout (left|side, then right)")
    ap.add_argument("--no-fit", action="store_true",
                    help="disable auto fit-to-screen")
    args = ap.parse_args()

    cfg_kp   = find_config_kp()
    cfg_data = parse_path_config(cfg_kp)
    raw_dir  = Path(cfg_data["raw_dir"])
    if not raw_dir.exists():
        sys.exit(f"[Error] raw_dir not found: {raw_dir}")

    print(f"raw_dir : {raw_dir}")
    print("output  : regular/kinematic/contact/*.json")

    annotate(raw_dir,
             vertical=not args.horizontal,
             auto_fit=not args.no_fit,
             data_name=cfg_data["data_name"],
             data_index=cfg_data["data_index"])

    print("All frames done!")