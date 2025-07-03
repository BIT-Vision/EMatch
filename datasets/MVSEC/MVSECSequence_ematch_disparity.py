import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from .utils.EventToVoxel import events_to_voxel
from .preLoader.mvsec import MVSEC

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# For test we use same frames as
# "Realtime Time Synchronized Event-based Stereo"
# by Alex Zhu et al. for consistency of test results.
FRAMES_FILTER_FOR_TEST = {
    'indoor_flying1': [140, 1201],
    'indoor_flying2': [120, 1421],
    'indoor_flying3': [73, 1616],
    'indoor_flying4': [190, 290]
}

# For the training we use different frames, since we found
# that frames recomended by "Realtime Time Synchronized
# Event-based Stereo" by Alex Zhu include some still frames.
FRAMES_FILTER_FOR_TRAINING = {
    'indoor_flying1': [80, 1260],
    'indoor_flying2': [160, 1580],
    'indoor_flying3': [125, 1815],
    'indoor_flying4': [190, 290],
    
    'outdoor_day2': [50, 12000]
}


class MVSECSequence_ematch_disparity(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, cfgs, name):
        self.root_path = cfgs['root_path']
        self.name = name
        self.mvsec_left = MVSEC(os.path.join(self.root_path, 'data_hdf5'), name, 'left')
        self.mvsec_right = MVSEC(os.path.join(self.root_path, 'data_hdf5'), name, 'right')

        self.crop_size = cfgs['crop_size']
        self.voxel_bins = cfgs['voxel_bins']
        self.dt = cfgs['dt']
        self.augmentor = Augmentor(self.crop_size, do_flip=True) if cfgs['augment'] else None

        self.split = cfgs['split']
        if self.split == 'train':
            self.start_no = FRAMES_FILTER_FOR_TRAINING[name][0]
            self.data_len = FRAMES_FILTER_FOR_TRAINING[name][1] - FRAMES_FILTER_FOR_TRAINING[name][0] + 1
        else:
            self.start_no = FRAMES_FILTER_FOR_TEST[name][0]
            self.data_len = FRAMES_FILTER_FOR_TEST[name][1] - FRAMES_FILTER_FOR_TEST[name][0] + 1

    def __getitem__(self, index):
        no = self.start_no + index

        # 1. Get Events
        T_end = self.mvsec_left.get_time_ofDisparity(no)
        T_start = T_end - self.dt * 0.001
        E_start = self.mvsec_left.find_ts_index(T_start)
        E_end = self.mvsec_left.find_ts_index(T_end)
        events_left = self.mvsec_left.get_events(E_start, E_end, rectify=True)
        voxel_0 = events_to_voxel(events_left, self.voxel_bins, 260, 346).transpose(1,2,0) # Trans to Voxel

        T_end = self.mvsec_left.get_time_ofDisparity(no)
        T_start = T_end - self.dt * 0.001
        E_start = self.mvsec_right.find_ts_index(T_start)
        E_end = self.mvsec_right.find_ts_index(T_end)
        events_right = self.mvsec_right.get_events(E_start, E_end, rectify=True)
        voxel_1 = events_to_voxel(events_right, self.voxel_bins, 260, 346).transpose(1,2,0) # Trans to Voxel

        # 2. Get Disparity
        disparity, valid = self.mvsec_left.get_disparityAndValid(no)
        # flow = flow.transpose(2,0,1)

        if self.name == 'outdoor_day1' or self.name == 'outdoor_day2':
            valid[193:,:]=False

        # 4. Augment
        if self.augmentor is not None:
            voxel_0, voxel_1, disparity, valid = self.augmentor(voxel_0, voxel_1, disparity, valid)
        else:
            crop_height = self.crop_size[0]
            crop_width = self.crop_size[1]
            height = voxel_0.shape[0]
            width = voxel_0.shape[1]
            if height != crop_height or width != crop_width:
                assert crop_height <= height and crop_width <= width
                # 中心裁剪
                start_y = (height - crop_height) // 2
                start_x = (width - crop_width) // 2
                voxel_0 = voxel_0[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
                voxel_1 = voxel_1[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
                disparity = disparity[start_y:start_y+crop_height, start_x:start_x+crop_width]
                valid = valid[start_y:start_y+crop_height, start_x:start_x+crop_width]

        # 5. To Tensor
        voxel_0 = torch.from_numpy(voxel_0).permute(2,0,1).float()
        voxel_1 = torch.from_numpy(voxel_1).permute(2,0,1).float()
        disparity = torch.from_numpy(disparity).float()
        valid = torch.from_numpy(valid).float()
        batch = {'voxel_0': voxel_0,
                 'voxel_1': voxel_1,
                 'target': disparity,
                 'valid': valid}

        return batch

    def __len__(self):
        return self.data_len


class Augmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        # crop augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0
        self.v_flip_prob = 0.1

    def resize_sparse_disparity_map(self, disparity, valid, fx=1.0, fy=1.0):
        ht, wd = disparity.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        disparity = disparity.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=0.5]
        disparity0 = disparity[valid>=0.5]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        disparity1 = disparity0 * (fx)

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disparity1 = disparity1[v]

        disparity_img = np.zeros([ht1, wd1], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        disparity_img[yy, xx] = disparity1
        valid_img[yy, xx] = 1

        return disparity_img, valid_img

    def spatial_transform(self, img1, img2, disparity, valid):
        # randomly sample scale
        ht, wd = img1.shape[:2]

        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # flow = flow * [scale_x, scale_y]
            # valid = cv2.resize(valid, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disparity, valid = self.resize_sparse_disparity_map(disparity, valid, fx=scale_x, fy=scale_y)

        # flip
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                valid = valid[:, ::-1]
                disparity = disparity[:, ::-1] * (-1.0)

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                valid = valid[::-1, :]
                disparity = disparity[::-1, :]

        # random crop
        y0 = 0 if img1.shape[0] == self.crop_size[0] else np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = 0 if img1.shape[1] == self.crop_size[1] else np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        disparity = disparity[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, disparity, valid

    def __call__(self, img1, img2, disparity, valid):
        # img   : (h,w,c)
        # disparity  : (h,w)
        # valid : (h,w)
        img1, img2, disparity, valid = self.spatial_transform(img1, img2, disparity, valid)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        disparity = np.ascontiguousarray(disparity)
        valid = np.ascontiguousarray(valid)

        return img1, img2, disparity, valid
