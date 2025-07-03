## Data

You can find the default configuration file for datasets in `datasets/configs_dataset`, which can be modified as needed.

### DSEC

The DSEC dataset can be downloaded [here](https://dsec.ifi.uzh.ch/dsec-datasets/download/).

The file structure should be like this:

```
DSEC/
├── cache/
├── train_calibration/
├── test_calibration/
|   ├── interlaken_00_a
|   |   └── calibration
|   |       ├── cam_to_cam.yaml
|   |       └── cam_to_lidar.yaml
|   └── ...
├── test_disparity_timestamps/
|   ├── interlaken_00_a.csv
|   └── ...
├── test_forward_optical_flow_timestamps/
|   ├── interlaken_00_a.csv
|   └── ...
├── train_events/
├── test_events/
|   ├── interlaken_00_a/
|   |   └── events/
|   |       ├── left/
|   |       |   ├── events.h5
|   |       |   └── rectify_map.h5
|   |       └── right/
|   └── ...
├── train_images/
├── test_images/
|   ├── interlaken_00_a/
|   |   └── images/
|   |       ├── left/
|   |       |   ├── rectified/
|   |       |   |   ├── 000000.png
|   |       |   |   ├── 000001.png
|   |       |   |   └── ...
|   |       |   └── exposure_timestamps.txt
|   |       ├── right/
|   |       └── timestamps.txt
|   └── ...
├── train_disparity/
|   ├── interlaken_00_c/
|   |   └── disparity/
|   |       ├── event/
|   |       |   ├── 000000.png
|   |       |   ├── 000002.png
|   |       |   └── ...
|   |       ├── image/
|   |       |   ├── 000000.png
|   |       |   ├── 000002.png
|   |       |   └── ...
|   |       └── timestamps.txt
|   └── ...
└── train_optical_flow/
    ├── thun_00_a/
    |   └── flow/
    |       ├── backward/
    |       |   ├── 000002.png
    |       |   ├── 000004.png
    |       |   └── ...
    |       ├── forward/
    |       ├── backward_timestamps.txt
    |       └── forward_timestamps.txt
    └── ...
```

**(not necessary)** You can execute the preprocessing code in `tools/DSEC_preprocess` to preprocess events and convert them into `npz` formatted voxels.

### MVSEC

The MVSEC dataset can be downloaded [here](https://daniilidis-group.github.io/mvsec/download/#hdf5-files).

The file structure should be like this:

```
MVSEC/
└── data_hdf5/
    ├── indoor_flying1_data.hdf5
    ├── indoor_flying1_gt.hdf5
    ├── indoor_flying2_data.hdf5
    ├── indoor_flying2_gt.hdf5
    ├── indoor_flying3_data.hdf5
    ├── indoor_flying3_gt.hdf5
    ├── outdoor_day1_data.hdf5
    ├── outdoor_day1_gt.hdf5
    ├── outdoor_day2_data.hdf5
    └── outdoor_day2_gt.hdf5
```
