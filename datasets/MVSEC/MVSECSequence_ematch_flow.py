import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from .utils.EventToVoxel import events_to_voxel
from .preLoader.mvsec import MVSEC

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

Valid_Index_ForTrain = {
    'indoor_flying1': [314, 2199],
    'indoor_flying2': [314, 2199],
    'indoor_flying3': [314, 2199],
    'outdoor_day2': [1950,28500]    # DCEI:[4375, 7002]
}
Valid_Index_ForTest = {
    'indoor_flying1': [314, 2199],
    'indoor_flying2': [314, 2199],
    'indoor_flying3': [314, 2199],
    'indoor_flying4': [196, 570],
    'outdoor_day1': [10167, 10954],  # ERAFT,TMA,IDNet:[10167,10954]   # DCEI:[245, 3000]
}

class MVSECSequence_ematch_flow(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, cfgs, name):
        self.root_path = cfgs['root_path']
        self.location = cfgs['location']
        self.name = name
        self.mvsec = MVSEC(os.path.join(self.root_path, 'data_hdf5'), name, self.location)

        self.crop_size = cfgs['crop_size']
        self.voxel_bins = cfgs['voxel_bins']
        self.dt = cfgs['dt']
        self.augmentor = Augmentor(self.crop_size, do_flip=True) if cfgs['augment'] else None

        self.split = cfgs['split']
        if self.split == 'train':
            self.start_no = Valid_Index_ForTrain[name][0]
            self.data_len = Valid_Index_ForTrain[name][1] - Valid_Index_ForTrain[name][0] - self.dt
        else:
            self.start_no = Valid_Index_ForTest[name][0]
            self.data_len = Valid_Index_ForTest[name][1] - Valid_Index_ForTest[name][0] - self.dt

    def __getitem__(self, index):
        no = self.start_no + index
        # 0. Get Images
        # pre_image = self.mvsec.get_image(no)
        # next_image = self.mvsec.get_image(no + self.dt)
        # # Dim changes: Gray to 3Gray
        # if len(pre_image.shape) == 2:
        #     pre_image = np.tile(pre_image[..., None], (1, 1, 3))
        #     next_image = np.tile(next_image[..., None], (1, 1, 3))

        # 1. Get Events
        E_pre = self.mvsec.get_idx_imageToevent(no - self.dt)
        E_mid = self.mvsec.get_idx_imageToevent(no)
        E_next = self.mvsec.get_idx_imageToevent(no + self.dt)
        events = self.mvsec.get_events(E_pre, E_mid)
        voxel_0 = events_to_voxel(events, self.voxel_bins, 260, 346).transpose(1,2,0) # Trans to Voxel
        events = self.mvsec.get_events(E_mid, E_next)
        voxel_1 = events_to_voxel(events, self.voxel_bins, 260, 346).transpose(1,2,0) # Trans to Voxel

        # 2. Get Flow
        T_start = self.mvsec.get_time_ofimage(no)
        T_end = self.mvsec.get_time_ofimage(no + self.dt)
        flow = self.mvsec.estimate_flow(T_start, T_end)
        # flow = flow.transpose(2,0,1)

        # 3. Get Valid
        valid = np.logical_and(np.linalg.norm(x=flow, ord=2, axis=2, keepdims=False) > 0,
                               np.logical_and(np.absolute(flow[..., 0]) < 1000, 
                                              np.absolute(flow[..., 1]) < 1000)).astype(np.float64)
        if self.name == 'outdoor_day1' or self.name == 'outdoor_day2':
            valid[193:,:]=False

        # 4. Augment
        if self.augmentor is not None:
            voxel_0, voxel_1, flow,  valid = self.augmentor(voxel_0, voxel_1, flow, valid)
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
                flow = flow[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
                valid = valid[start_y:start_y+crop_height, start_x:start_x+crop_width]

        # 5. To Tensor
        voxel_0 = torch.from_numpy(voxel_0).permute(2,0,1).float()
        voxel_1 = torch.from_numpy(voxel_1).permute(2,0,1).float()
        flow = torch.from_numpy(flow).permute(2,0,1).float()
        valid = torch.from_numpy(valid).float()
        batch = {'voxel_0': voxel_0,
                 'voxel_1': voxel_1,
                 'target': flow,
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
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=0.5]
        flow0 = flow[valid>=0.5]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, voxel_0, voxel_1, flow, valid):
        # randomly sample scale
        ht, wd = voxel_0.shape[:2]

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
            voxel_0 = cv2.resize(voxel_0, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            voxel_1 = cv2.resize(voxel_1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # flow = flow * [scale_x, scale_y]
            # valid = cv2.resize(valid, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        # flip
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                voxel_0 = voxel_0[:, ::-1]
                voxel_1 = voxel_1[:, ::-1]
                valid = valid[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                voxel_0 = voxel_0[::-1, :]
                voxel_1 = voxel_1[::-1, :]
                valid = valid[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # random crop
        y0 = 0 if voxel_0.shape[0] == self.crop_size[0] else np.random.randint(0, voxel_0.shape[0] - self.crop_size[0])
        x0 = 0 if voxel_0.shape[1] == self.crop_size[1] else np.random.randint(0, voxel_0.shape[1] - self.crop_size[1])

        voxel_0 = voxel_0[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        voxel_1 = voxel_1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return voxel_0, voxel_1, flow, valid

    def __call__(self, voxel_0, voxel_1, flow, valid):
        # img   : (h,w,c)
        # flow  : (h,w,c)
        # voxel : (h,w,c)
        # valid : (h,w)
        voxel_0, voxel_1, flow, valid = self.spatial_transform(voxel_0, voxel_1, flow, valid)
        voxel_0 = np.ascontiguousarray(voxel_0)
        voxel_1 = np.ascontiguousarray(voxel_1)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return voxel_0, voxel_1, flow, valid
