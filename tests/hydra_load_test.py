# import os
# import sys
# dynamic_path = os.path.abspath(__file__+"/../../")
# print(dynamic_path)
# sys.path.append(dynamic_path)
from pathlib import Path
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dvrk_data_processing.utils.utility import clear_folder, create_folder

@dataclass
class PathConfig:
    raw_dir: str
    data_name: str
    intermediate_dir: str
    processed_dir: str

@dataclass
class ProcessedConfig:
    stage: str
    folder_initialize: bool = False

@dataclass
class AppConfig:
    path_config: PathConfig
    preprocess: ProcessedConfig
    workspace: str

cs = ConfigStore.instance()
cs.store(name="hydra_load_test", node=AppConfig)

# set config path
p_config = Path.cwd().parent / 'config'
@hydra.main(
    version_base=None,
    config_path= str(p_config),
    config_name="config"
)
def main(cfg: AppConfig):
    print(OmegaConf.to_yaml(cfg))
    dir_paths = {k: Path(v) for k, v in cfg.path_config.items()}
    assert dir_paths["raw_dir"].exists(), 'Incorrect Raw Data Path!'
    for new_path in (dir_paths['intermediate_dir'], dir_paths['processed_dir']):
        if not new_path.exists():
            create_folder(new_path)
        else:
            if cfg.preprocess.folder_initialize:
                clear_folder(new_path)


if __name__ == '__main__':
    # # test loading
    from hydra import compose, initialize
    with initialize(version_base=None, config_path='../config'):
        cfg = compose(config_name="config")
    # dir_paths = {k: Path(v) for k, v in cfg.path_config.items()}
    main()
