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

# Clone the repository

*** **Important: Please include the submodules when cloning the repository.** ***



#  Folder Structure

(will be updated later)

```
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

Local path configration is required. You may create your own configration file. The file should be named `<your name>_local.yaml` and should be located in the `config/path_config` folder.

I'll take `jackzhy_local.yaml` as an example.

|        Name        |                              Meaning                              |           Default / Given Values            |                     Notes                     |
|:------------------:|:-----------------------------------------------------------------:|:-------------------------------------------:|:---------------------------------------------:|
|     `data_dir`     |                    the downloaded data folder                     | /home/jack/dvrk_multimodal_data_collection  | You may need to change it for your own config |
|    `data_name`     |                   name of your selected dataset                   |               data_202050808                | You may need to change it for your own config |
|    `data_index`    |                index of your selected sub-dataset                 |                      1                      | You may need to change it for your own config |
|     `raw_dir`      |                          raw data folder                          | \${.data_dir}/\$.{data_name}/$.{data_index} |                       -                       |
| `intermediate_dir` | intermediate output folder, usually for resize and rectify output |     \${.data_dir}/interm/$.{data_index}     |                       -                       |
|  `processed_dir`   |    processed data output folder, for all the other operations     |     \${.data_dir}/output/$.{data_index}     |                       -                       |


# Post-collection data processing

## Required: Resize and Rectify

For this step, you will implement the image resize and rectification for iamges. You will also generate a new, scaled camera calibration file based on the given size.

### Configuration file

```
stage: resize_rectify
folder_initialize: false

# custom configurations
resize_config:
  original_size: [1920, 1080]
  new_size: [640, 480]
  enable_resize: true

input_folder: '${path_config.raw_dir}/regular'
output_folder: "${path_config.intermediate_dir}/regular"

enable_rectify: true
```

### How to Run


