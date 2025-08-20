# dVRK Multi-Modal Data Collection

Table of contents:
- [Introduction](#introduction)
- [Clone the repository](#clone-the-repository)
- [Install the local package, Required both for local and virtual environment](#install-the-local-package-required-both-for-local-and-virtual-environment)
- [Folder Structure](#folder-structure)
  - [Configration, local path](#configration-local-path)
- [Post-collection data processing](#post-collection-data-processing)
  - [Running Configuration file](#running-configuration-file)
  - [Required: Resize and Rectify](#required-resize-and-rectify)
  - [Kinematic Mapping](#kinematic-mapping)
  - [Depth Estimation](#depth-estimation)
  - [Optical Flow](#optical-flow)

# Introduction

This repository contains scripts for post-collection data processing for multi-modal data collected by dVRK.

The **input modalities** are:
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
- Contact Sensor Data / Contact Annotation (Optional)


The **output modalities after post-collection processing** contains:
- Kinematic Mapping Masked Images
  - based on dVRK software measurement
  - based on hand-eye calibration
- Disparity Map (Depth Map)
- Resized and Rectified Images
- Optical Flow Images

Expect for the resized and rectified images, all the other modalities also included the raw processed data save in `*.npy` format.

# Clone the repository

*** **Important: Please include the submodules when cloning the repository.** ***

Go to the directory where you want to clone the repository. We would suggest the home directory since it can save you sometime on editing configuration files.

## HTTPS

Run the following command:

```bash
git clone --recursive https://github.com/jackzhy96/dvrk_multimodal_data_collection.git
```

## SSH

**Note: This approach could be deprecated after converting to be a public repository.**

Run the following command:

```bash
git clone --recursive git@github.com:jackzhy96/dvrk_multimodal_data_collection.git
```

# Install the local package, Required both for local and virtual environment

Run the following command:

```bash
cd <your cloned dvrk_multimodal_data_collection repo>
pip install -e .
```

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

I'll take `config/path_config/jackzhy_local.yaml` as an example.

|        Name        |                              Meaning                              |            Default / Given Values            |                     Notes                     |
|:------------------:|:-----------------------------------------------------------------:|:--------------------------------------------:|:---------------------------------------------:|
|     `data_dir`     |                    the downloaded data folder                     |  /home/jack/dvrk_multimodal_data_collection  | You may need to change it for your own config |
|    `data_name`     |                   name of your selected dataset                   |                data_202050808                | You may need to change it for your own config |
|    `data_index`    |                index of your selected sub-dataset                 |                      1                       | You may need to change it for your own config |
|     `raw_dir`      |                          raw data folder                          | \${.data_dir}/\$.{data_name}/\$.{data_index} |                       -                       |
| `intermediate_dir` | intermediate output folder, usually for resize and rectify output |     \${.data_dir}/interm/\$.{data_index}     |                       -                       |
|  `processed_dir`   |    processed data output folder, for all the other operations     |     \${.data_dir}/output/\$.{data_index}     |                       -                       |


# Post-collection data processing

## Running Configuration file

The running configuration file is located in `config/config_<operation>_<your name>.yaml`. You may create your own running configuration file. The file should follow the above name convention.

I'll take `config_kp_jack.yaml` as an example.

|           Name            |                                  Meaning                                  |                         Default / Given Values                          |                                                            Notes                                                            |
|:-------------------------:|:-------------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------:|
|        `workspace`        |                          your selected workspace                          |            /home/jackzhy/dvrk\_multimodal\_data\_collection             |                            The place where you clone the repository, you may need to change it.                             |
|      `camera_names`       |                     the name of the selected cameras                      |                            ['left', 'right']                            |   name of your cameras, for stereo camera, use the one I am showing. Mono camera could be supported but not validated yet   |
| `camera_calibration_path` |                         camera calibration files                          |           \${path_config.intermediate_dir}/camera_calibration           |                            the path of your camera calibration files, you may need to change it                             |
|  `defaults/path_config`   |                 the name of your local path configuration                 |                               jack_local                                |                                                 your path config file name                                                  |
|   `defaults/preprocess`   |                       selected processing operation                       |                              kinematic_map                              | processing operation name, currently support: kinematic\_mapping, optical\_flow(\_raft), resize\_rectify, depth\_estimation |
|     `default/_self_`      |           enable the inheritance among the configuration files            |                                    -                                    |                                        just included the term in the running config                                         |
|      `camera_offset`      | manual camera offset to compensate different camera base frame definition |           [-1.0, 0.0, 0.0;   0.0, -1.0, 0.0;   0.0, 0.0, 1.0]           |                             Optional, only used for kinematic mapping, only for the dVRK system                             |
|   `handeye_calib_path`    |                      hand-eye calibration file path                       | \${path_config.data_dir}/\${path_config.data_name}/hand_eye_calibration | Optional, only used for kinematic mapping, only for the dVRK system, can change if you put the files in a different folder  |


## Required: Resize and Rectify

For this step, you will implement the image resize and rectification for images. You will also generate a new, scaled camera calibration file based on the given size.

### Configuration file

The hyper-parameters of resizing and rectification are stored in `config/preprocess/resize_rectify.yaml`:

|             Name              |                 Meaning                 |          Default / Given Values          |                            Note                             |
|-------------------------------|:---------------------------------------:|:----------------------------------------:|:-----------------------------------------------------------:|
|            `stage`            |     the processing operation to do      |              resize_rectify              |               You **CANNOT** change this term               |
|      `folder_initialize`      |  empty the output folder if it exists   |                  False                   | Usually keep to be False in case of forced removal of files |
| `resize_config/original_size` |       original size of the images       |               [1920, 1080]               |        You may need to change it based on your input        |
|   `reszie_config/new_size`    |        output size of the images        |                [640, 480]                |   You may need to change it based on your desired output    |
| `resize_config/enable_resize` |   whether to enable resizing feature    |                   True                   |                              -                              |
|       `enable_rectify`        | whether to enable rectification feature |                   True                   |                              -                              |
|        `input_folder`         |     the folder of the input images      |     \${path_config.raw_dir}/regular      |                              -                              |
|        `output_folder`        |  the folder to save the output images   | \${path_config.intermediate_dir}/regular |                              -                              |

If you have to change the arguments in the hyper-parameter config file, please create a new file named `<your name>_resize_rectify.yaml` and put it in the `config/preprocess` folder.

After settling the hyper-parameters, you can create your own running configuration file. The file should be named `<your name>_local.yaml` and should be located in the `config` folder.

### How to Run

(will be updated later)

## Kinematic Mapping

(will be updated later)

### Configuration file

(will be updated later)

### How to Run

(will be updated later)

## Depth Estimation

(will be updated later)

### Configuration file

(will be updated later)

### How to Run

(will be updated later)

## Optical Flow

(will be updated later)

### Configuration file

(will be updated later)

### How to Run

(will be updated later)

