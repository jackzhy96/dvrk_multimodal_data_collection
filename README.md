# dVRK Multi-Modal Data Collection

Table of contents:
- [Introduction](#introduction)
- [Clone the repository](#clone-the-repository)
- [Install the local package](#install-the-local-package)
- [Folder Structure](#folder-structure)
  - [Configration, local path](#configration-local-path)
- [Post-collection data processing](#post-collection-data-processing)
  - [Running Configuration file](#running-configuration-file)
  - [Required: Resize and Rectify](#required-resize-and-rectify)
  - [Kinematic Mapping](#kinematic-mapping)
  - [Depth Estimation](#depth-estimation)
  - [Optical Flow](#optical-flow)
- [Others](#others)

If you have any further questions, please feel free to contact Haoying (Jack) Zhou for more information.
- Email
  - hzhou6@wpi.edu
  - hzhou62@jh.edu
- Github: https://github.com/jackzhy96

Alternatively, you can also create an issue in this repository.

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

# Install the local package

*** **Important !!! : Installing the local package is required both in the local and virtual environment.** ***

Run the following command:

```bash
cd <your cloned dvrk_multimodal_data_collection repo>
pip install -e .
```

If you want to run the depth estimation code, you may also need to install the conda environment for [FoundationStereo](https://github.com/jackzhy96/FoundationStereo?tab=readme-ov-file#installation):

```
@article{wen2025stereo,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={Bowen Wen and Matthew Trepte and Joseph Aribido and Jan Kautz and Orazio Gallo and Stan Birchfield},
  journal={CVPR},
  year={2025}
}
```


#  Folder Structure

You may find the overall folder structure as follows:

(the `interm` folder is for intermediate output, the `output` folder is for processed data output, the `data` folder is for the raw data)

```
.
├── config
    ├── path_config
    ├── preprocess
    └── ...
├── dvrk_config
    ├── contact_sensor
    └── ...
├── video_launch
    ├── gscam_v4l.launch
    └── ...
├── src
    └── dvrk_data_processing
        ├── Annotation
            └── annotate_contact.py
        ├── depth_estimation
            └── gen_depth_estimate.py
        ├── kinematic_mapping
            ├── gen_kinematic_heatmap_dVRK.py
            └── gen_kinematic_heatmap_handeye.py
        ├── optical_flow
            ├── gen_optical_flow.py
            └── gen_optical_flow_raft.py
        ├── raw_image_processing
            └── gen_resize_rectify.py
        └── utils
├── data
    ├── camera_calibration
        ├── left.yaml
        └── right.yaml
    ├── data_test
        ├── 1
            ├── regular
                ├── image
                    ├── left
                        ├── 0.png
                        └── ...
                    ├── right
                    └── ...
                ├── kinematic
                    ├── ECM
                    ├── PSM1
                        ├── 0.json
                        └── ...
                    └── ...
                └── time_sync
            └── annotation
                ├── contact_detection
                    ├── 0.json
                    └── ...
                ├── events
                ├── skill_assessment
                └── ...
        ├── ...
        ├── hand_eye_calibration
            ├── PSM1-registration-dVRK.json
            └── ...
        └── SUJ_measured_cp_output.json
    ├── ...
    ├── interm
        ├── data_test
            ├── 1
                 ├── camera_calibration
                 └── regular
                     ├── image
                     ├── kinematic
                     └── time_sync 
            └── ...
        └── ...
    └── output
        ├── data_test
            ├── 1
                ├── depth_estimation
                    ├── combined_image
                    ├── depth_image
                    └── disparity
                ├── kinematic_map
                    ├── PSM1
                        ├── left
                            ├── image
                            └── heatmap
                                ├── 0.npy
                                └── ...
                        └── right
                    └── ...
                └── optical_flow
                    ├── left
                        ├── image
                        └── optical_flow
                            ├── 0.npy
                            └── ...
                    └── right
            └── ...
        └── ...
├── replay
    ├── dvrk_bag_replay.py
    ├── run_two_arm.sh
    └── ...
├── rosbag_record
    └── ...
├── README.md
├── pyproject.toml
└── ...
```

(will be updated later)

## Configration, local path

Local path configration is required. You may create your own configration file. The file should be named `<your name>_local.yaml` and should be located in the `config/path_config` folder.

I'll take `config/path_config/jackzhy_local.yaml` as an example.

|        Name        |                              Meaning                              |           Default / Given Values            |       Notes        |
|:------------------:|:-----------------------------------------------------------------:|:-------------------------------------------:|:------------------:|
|     `data_dir`     |                    the downloaded data folder                     | /home/jack/dvrk_multimodal_data_collection  | change if required |
|    `data_name`     |                   name of your selected dataset                   |               data_202050808                | change if required |
|    `data_index`    |                index of your selected sub-dataset                 |                      1                      | change if required |
|     `raw_dir`      |                          raw data folder                          | `${.data_dir}/$.{data_name}/$.{data_index}` |         -          |
| `intermediate_dir` | intermediate output folder, usually for resize and rectify output |    `${.data_dir}/interm/$.{data_index}`     |         -          |
|  `processed_dir`   |    processed data output folder, for all the other operations     |    `${.data_dir}/output/$.{data_index}`     |         -          |


# Post-collection data processing

## Running Configuration file

The running configuration file is located in `config/config_<operation>_<your name>.yaml`. You may create your own running configuration file. The file should follow the above name convention.

I'll take `config_kp_jack.yaml` as an example.

|           Name            |                                  Meaning                                  |                                  Default / Given Values                                   |                                                             Notes                                                              |
|:-------------------------:|:-------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|
|        `workspace`        |                          your selected workspace                          |                     /home/jackzhy/dvrk\_multimodal\_data\_collection                      |                             usually the place where you clone the repository, change if required.                              |
|      `camera_names`       |                     the name of the selected cameras                      |                                     ['left', 'right']                                     |       name of your cameras, for stereo camera, use the current one. Mono camera could be supported but not validated yet       |
| `camera_calibration_path` |                         camera calibration files                          |                    \${path_config.intermediate_dir}/camera_calibration                    |                                 the path of your camera calibration files, change if required                                  |
|  `defaults/path_config`   |                 the name of your local path configuration                 |                                        jack_local                                         |                                                       change if required                                                       |
|   `defaults/preprocess`   |                       selected processing operation                       |                                       kinematic_map                                       | processing operation name, currently support: `kinematic_mapping`, `optical_flow(_raft)`, `resize_rectify`, `depth_estimation` |
|     `defaults/_self_`     |           enable the inheritance among the configuration files            |                                             -                                             |                                      just need to include the term in the running config                                       |
|      `camera_offset`      | manual camera offset to compensate different camera base frame definition | $$\begin{bmatrix} -1.0 & 0.0 & 0.0 \\ 0.0 & -1.0 & 0.0 \\ 0.0 & 0.0 & 1.0\end{bmatrix} $$ |                              Optional, only used for kinematic mapping, only for the dVRK system                               |
|   `handeye_calib_path`    |                      hand-eye calibration file path                       |          \${path_config.data_dir}/\${path_config.data_name}/hand_eye_calibration          |                    Optional, only used for kinematic mapping, only for the dVRK system, change if required                     |

After settling the hyper-parameters for your selected procedure, you can create your own running configuration file. The file should be named `<your name>_local.yaml` and should be located in the `config` folder.

## Required: Resize and Rectify

For this step, you will implement the image resize and rectification for images. You will also generate a new, scaled camera calibration file based on the given size.

### Configuration file

The hyper-parameters of resizing and rectification are stored in `config/preprocess/resize_rectify.yaml`:

|             Name              |                 Meaning                 |          Default / Given Values          |                            Note                             |
|-------------------------------|:---------------------------------------:|:----------------------------------------:|:-----------------------------------------------------------:|
|            `stage`            |     the processing operation to do      |              resize_rectify              |               you **CANNOT** change this term               |
|      `folder_initialize`      |  empty the output folder if it exists   |                  False                   | usually keep to be False in case of forced removal of files |
| `resize_config/original_size` |       original size of the images       |               [1920, 1080]               |           change based on your input if required            |
|   `reszie_config/new_size`    |        output size of the images        |                [640, 480]                |       change based on your desired output if required       |
| `resize_config/enable_resize` |   whether to enable resizing feature    |                   True                   |                              -                              |
|       `enable_rectify`        | whether to enable rectification feature |                   True                   |                              -                              |
|        `input_folder`         |         the folder of the input         |     \${path_config.raw_dir}/regular      |                     change if required                      |
|        `output_folder`        |      the folder to save the output      | \${path_config.intermediate_dir}/regular |                     change if required                      |

If you have to change the arguments in the hyper-parameter config file, please create a new file named `<your name>_resize_rectify.yaml` and put it in the `config/preprocess` folder.

### How to Run

Assume that you have been in the folder where you cloned the repository.

Make sure that you have created your own running configuration file and the name in [line #139](https://github.com/jackzhy96/dvrk_multimodal_data_collection/blob/main/src/dvrk_data_processing/raw_image_processing/gen_resize_rectify.py#L139) has been changed to your running config file.

Run the following command:

```bash
cd src/dvrk_data_processing/raw_data_processing
python gen_resize_rectify.py
```

## Kinematic Mapping

You need to run the resize and rectify step before running the kinematic mapping.

(will be updated later)

### Configuration file

The hyper-parameters of kinematic mapping are stored in `config/preprocess/kinematic_map.yaml`:

|              Name               |                       Meaning                       |          Default/ Given Values           |                            Notes                            |
|:-------------------------------:|:---------------------------------------------------:|:----------------------------------------:|:-----------------------------------------------------------:|
|             `stage`             |           the processing operation to do            |            kinematic_mapping             |               you **CANNOT** change this term               |
|       `folder_initialize`       |        empty the output folder if it exists         |                  False                   | usually keep to be False in case of forced removal of files |
|           `img_size`            |             the size of the image input             |                [640, 480]                |      change to your own selection (the resized output)      |
|           `arm_name`            |      the name(s) for the selected robot arm(s)      |             ['PSM1', 'PSM2']             |             Currently support: PSM1, PSM2, PSM3             |
|         `input_folder`          |               the folder of the input               | \${path_config.intermediate_dir}/regular |                     change if required                      |
|         `output_folder`         |            the folder to save the output            | \${path_config.processed_dir}/\${.stage} |                     change if required                      |
|     `weight_config/sigma_x`     |       proportional to your x spanning radius        |                    60                    |                              -                              |
|     `weight_config/sigma_y`     |       proportional to your y spanning radius        |                    60                    |                              -                              |
| `weight_config/advanced_weight` |     whether to use the advanced weight function     |                   True                   |                              -                              |
|    `weight_config/tol_dist`     |  tolerance of the closest distance to your camera   |                   0.05                   |                        the unit is m                        |
|        `enable_overlay`         | whether to overlay the grey-scale image to the mask |                   True                   |                              -                              |

If you have to change the arguments in the hyper-parameter config file, please create a new file named `<your name>_kinematic_map.yaml` and put it in the `config/preprocess` folder.

### How to Run

Assume that you have been in the folder where you cloned the repository.

#### mapping using dVRK measurement

Make sure that you have created your own running configuration file and the name in [line #112](https://github.com/jackzhy96/dvrk_multimodal_data_collection/blob/main/src/dvrk_data_processing/kinematic_mapping/gen_kinematic_heatmap_dVRK.py#L112) has been changed to your running config file.

Run the following command:

```bash
cd src/dvrk_data_processing/kinematic_mapping
python gen_kinematic_heatmap_dVRK.py
```

#### mapping using hand-eye calibration results

Make sure that you have created your own running configuration file and the name in [line #113](https://github.com/jackzhy96/dvrk_multimodal_data_collection/blob/main/src/dvrk_data_processing/kinematic_mapping/gen_kinematic_heatmap_handeye.py#L113) has been changed to your running config file.

Run the following command:

```bash
cd src/dvrk_data_processing/kinematic_mapping
python gen_kinematic_heatmap_handeye.py
```

** ***Note: this approach is selected as the default approach*** **

## Depth Estimation

This step is to estimate the depth of the images. We are using [FoundationStereo](https://github.com/NVlabs/FoundationStereo). You may create your own pipeline.

### Configuration file

The hyper-parameters of depth estimation are stored in `config/preprocess/depth_estimation.yaml`:

|           Name           |                        Meaning                        |                            Default / Given Values                            |                            Notes                            |
|:------------------------:|:-----------------------------------------------------:|:----------------------------------------------------------------------------:|:-----------------------------------------------------------:|
|         `stage`          |            the processing operation to do             |                               depth_estimation                               |               you **CANNOT** change this term               |
|   `folder_initialize`    |         empty the output folder if it exists          |                                    False                                     | usually keep to be False in case of forced removal of files |
|      `input_folder`      |                the folder of the input                |                   \${path_config.intermediate_dir}/regular                   |                     change if required                      |
|     `output_folder`      |             the folder to save the output             |                   \${path_config.processed_dir}/\${.stage}                   |                     change if required                      |
| `pretrained_model_path`  |           the path of the pretrained weight           | \${workspace}/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth |                     change if required                      |
|         `scale`          |          scale implement to the input images          |                                     1.0                                      |                     change if required                      |
| `hierarchical_inference` |         whether to use hierarchical inference         |                                     True                                     |                              -                              |
|      `valid_iters`       |   number of flow-field updates during forward pass    |                                      32                                      |                     change if required                      |
|       `save_depth`       | whether to save the raw depth output in `*.npy` files |                                     True                                     |                              -                              |
|   `save_visualization`   |           whether to save the depth images            |                                     True                                     |                              -                              |
|      `start_frame`       |          the start index of depth estimation          |                                      -1                                      |                -1 means selecting all frames                |
|       `end_frame`        |           the end index of depth estimation           |                                      -1                                      |                -1 means selecting all frames                |

If you have to change the arguments in the hyper-parameter config file, please create a new file named `<your name>_depth_estimation.yaml` and put it in the `config/preprocess` folder.

### How to Run

Assume that you have been in the folder where you cloned the repository. Make sure that your local/virtual environment is activated and fulfills the requirements of running FoundationStereo.

Make sure that you have created your own running configuration file and the name in [line #348](https://github.com/jackzhy96/dvrk_multimodal_data_collection/blob/main/src/dvrk_data_processing/depth_estimation/gen_depth_estimate.py#L348) has been changed to your running config file.

Run the following command:

```bash
cd src/dvrk_data_processing/depth_estimation
python gen_depth_estimate.py
```

## Optical Flow

This step is to estimate the optical flow of the images. You may create your own pipeline.

### Traditional Approach

For the traditional approach, we use openCV to implement the optical flow. The code has been tested under openCV 4.5.3. You may create your own pipeline.

#### Configuration file

The hyper-parameters of optical flow are stored in `config/preprocess/optical_flow.yaml`:

|                 Name                  |                                         Meaning                                          |          Default / Given Values          |                                         Notes                                         |
|:-------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------:|:-------------------------------------------------------------------------------------:|
|                `stage`                |                              the processing operation to do                              |             depth_estimation             |                            you **CANNOT** change this term                            |
|          `folder_initialize`          |                           empty the output folder if it exists                           |                  False                   |              usually keep to be False in case of forced removal of files              |
|            `input_folder`             |                                 the folder of the input                                  | \${path_config.intermediate_dir}/regular |                                  change if required                                   |
|            `output_folder`            |                              the folder to save the output                               | \${path_config.processed_dir}/\${.stage} |                                  change if required                                   |
|             `flow_format`             |                          the output format of raw optical flow                           |                   npy                    |                                  change if required                                   |
|        `enable_visualization`         |                          whether to save the optical flow image                          |                   True                   |                                           -                                           |
|        `enable_preprocessing`         |                          whether to enable preprocessing filter                          |                   True                   |                                           -                                           |
|      `filter_config/bilateral_d`      |               the diameter of each pixel neighborhood used when filtering                |                    9                     |                   change if required, 0 or -1 means auto-selection                    |
| `filter_config/bilateral_sigma_color` |                 the variance of the Gaussian filter in the color domain                  |                   75.0                   |                                  change if required                                   |
| `filter_config/bilateral_sigma_space` |                the variance of the Gaussian filter in the spatial domain                 |                   75.0                   |                                  change if required                                   |
| `filter_config/gaussian_kernel_size`  |                                 the Gaussian kernel size                                 |                  [5, 5]                  |                                  change if required                                   |
|    `filter_config/gaussian_sigma`     |                           the variance of the Gaussian kernel                            |                   1.2                    |                                  change if required                                   |
|   `algorithm_config/pyramid_scale`    |                        the image ratio between each pyramid level                        |                   0.5                    | allow the algorithm to track motion from a coarse to a fine level, change if required |
|   `algorithm_config/pyramid_level`    |                            the total number of pyramid layers                            |                    3                     |                                  change if required                                   |
|    `algorithm_config/window_size`     |                   the size of the search window at each pyramid level                    |                    15                    |                                  change if required                                   |
|     `algorithm_config/iterations`     |            number of iterations the algorithm performs at each pyramid level             |                    3                     |                                  change if required                                   |
|               `poly_n`                |    the size of the pixel neighborhood used to find polynomial expansion in each pixel    |                    5                     |                                  change if required                                   |
|             `poly_sigma`              | the standard deviation of the Gaussian filter to smooth polynomial expansion derivatives |                   1.2                    |                                  change if required                                   |
|                `flags`                |                                     additional flags                                     |                    0                     |                                    add if required                                    |

If you have to change the arguments in the hyper-parameter config file, please create a new file named `<your name>_optical_flow.yaml` and put it in the `config/preprocess` folder.

#### How to Run

Assume that you have been in the folder where you cloned the repository.

Make sure that you have created your own running configuration file and the name in [line #401](https://github.com/jackzhy96/dvrk_multimodal_data_collection/blob/main/src/dvrk_data_processing/optical_flow/gen_optical_flow.py#L401) has been changed to your running config file.

Run the following command:

```bash
cd src/dvrk_data_processing/optical_flow
python gen_optical_flow.py
```

### Deep Learning Approach

We are using [RAFT](https://github.com/princeton-vl/RAFT) with [Pytorch-integrated functions](https://docs.pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html). You may create your own pipeline.

The code has been tested in my local environment, which fulfills the following requirements:

```
CUDA==12.4
torch==2.3.1
torchvision==0.18.1
numpy==1.23.5
opencv==4.5.3
```

** ***Note: this approach is selected as the default approach*** **

#### Configuration file

The hyper-parameters of optical flow are stored in `config/preprocess/optical_flow_raft.yaml`:

|              Name              |                          Meaning                          |          Default / Given Values          |                                     Notes                                     |
|:------------------------------:|:---------------------------------------------------------:|:----------------------------------------:|:-----------------------------------------------------------------------------:|
|            `stage`             |              the processing operation to do               |             depth_estimation             |                        you **CANNOT** change this term                        |
|      `folder_initialize`       |           empty the output folder if it exists            |                  False                   |          usually keep to be False in case of forced removal of files          |
|         `input_folder`         |                  the folder of the input                  | \${path_config.intermediate_dir}/regular |                              change if required                               |
|        `output_folder`         |               the folder to save the output               | \${path_config.processed_dir}/\${.stage} |                              change if required                               |
|         `flow_format`          |         the output format of the raw optical flow         |                   npy                    |                              change if required                               |
|     `enable_visualization`     |          whether to save the optical flow images          |                   True                   |                                       -                                       |
|       `save_confidence`        | whether to save the confidence information when inference |                  False                   |                                       -                                       |
| `model_config/use_pretrained`  |           whether to use the pretrained weights           |                   True                   |      change if required, highly recommend to use the pretrained weights       |
|  `model_config/model_variant`  |            selection of the pretrained models             |                  large                   |       change if required, large is more accurate while small is faster        |
|     `model_config/device`      |       the device used for deep learning evaluation        |                   auto                   |        auto will use GPU if available, can also take `cuda` and `cpu`         |
|   `model_config/batch_size`    |             number of frame pairs to process              |                    1                     |                              change if required                               |
| `model_config/mixed_precision` |       whether to use mixed precision for inference        |                   True                   | using mixed precision may accelerate inference rate and decrease memory usage |
| `model_config/num_flow_update` |                 number of RAFT iterations                 |                    -1                    |                         -1 means using all iterations                         |

If you have to change the arguments in the hyper-parameter config file, please create a new file named `<your name>_optical_flow_raft.yaml` and put it in the `config/preprocess` folder.

** Note: the dimensions of the resized images should be a multiple of 8 for the best performance.

#### How to Run

Assume that you have been in the folder where you cloned the repository. Make sure that your local/virtual environment is activated and fulfills the requirements of running FoundationStereo.

Make sure that you have created your own running configuration file and the name in [line #548](https://github.com/jackzhy96/dvrk_multimodal_data_collection/blob/main/src/dvrk_data_processing/optical_flow/gen_optical_flow_raft.py#L548) has been changed to your running config file.

Run the following command:

```bash
cd src/dvrk_data_processing/optical_flow
python gen_optical_flow_raft.py
```

# Others

(will be updated later)