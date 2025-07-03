## Preprocess for DSEC

You can use this code to preprocess events and save them as voxels in npz format.

For `events_to_npz.py`, the following arguments can be set:

- --split : 'test' or 'train'
- --root_path : location of DSEC dataset
- --save_path : NPZ file storage location
- --dt : default is 100 (ms)
- --bins : default is 15


Here is an example：

```
python events_to_npz.py \
--split test \
--root_path data/DSEC \
--save_path data/DSEC/cache/DSECSequence_ematch_flow/test/left/voxel_dt100_bins15_us/

python events_to_npz.py \
--split train \
--root_path data/DSEC \
--save_path data/DSEC/cache/DSECSequence_ematch_flow/train/left/voxel_dt100_bins15_us/
```
