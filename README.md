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
