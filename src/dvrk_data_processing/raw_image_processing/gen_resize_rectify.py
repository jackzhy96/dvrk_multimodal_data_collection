from dataclasses import dataclass
from typing import Union, Tuple
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import numpy as np
from dvrk_data_processing.utils.hydra_config import PathConfig, KinematicMapConfig
from dvrk_data_processing.utils.utility import load_stereo_proj_mtx, create_folder, clear_folder, load_json_cp, \
    glob_sorted_frame, load_camera_param_yaml
# from dvrk_data_processing.utils.data_load_config import CameraInfo, KinematicInfo, datacls_from_dict
from tqdm import tqdm
import cv2




if __name__ == '__main__':
    # main()
    # print('Done!')
    from hydra import compose, initialize

    with initialize(version_base=None, config_path='../../../config'):
        cfg = compose(config_name="config_kp")