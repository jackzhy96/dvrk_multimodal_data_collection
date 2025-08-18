# dVRK Multi-Modal Data Collection
This repository contains scripts for post-collection data processing for multi-modal data collected by dVRK.
The input modalities are:
- Stereo Camera Images, RGB
- Robot motion data
  - measured_cp
  - measured_cv
  - measured_js
  - setpoint_cp
  - setpoint_js
  - Only for PSMs: jaw/measured_js, jaw/setpoint_js
  - Only for PSMs: measured_frequency
- Side Camera Images (Optional)

#  structure

(will be updated later)

```bash
.
├── src/
│   ├── dvrk_data_processing
│   │   └── depth_estimation
│   │   │   └── gen_depth_estimate.py
│   │   └── ...
│   └── ...
├── data
│   └── ...
├── README.md
└── ...
```

## Configration, local path

Local path configration is required. You may create your own configration file. The file should be named `<your name>_local.yaml` and should be located in the `config/path_config` folder

```bash
data_dir: <your data save folder>, e.g. "/home/jackzhy/dvrk_multimodal_data_collection/data"
data_name: <your data name, ususally named as the experiment date>, e.g. "data_20250808"
data_index: <your selected subset of the data, usually int, 0-n>, e.g. "1"
raw_dir: "${.data_dir}/${.data_name}/${.data_index}" [the raw data folder, if not following the default folder structure, you may need to change this]
intermediate_dir: "${.data_dir}/interm/${.data_name}/${.data_index}" [the folder for intermediate data, usually for resize and rectify outputs]
processed_dir:  "${.data_dir}/output/${.data_name}/${.data_index}" [the fodler for processed data, usually for all the other post-collection data processing]
```