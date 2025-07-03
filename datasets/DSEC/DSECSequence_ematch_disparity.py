import os
import numpy as np
import cv2
import torch.utils.data as data
import torch

from .preLoader.Event import dsecPreLoader_Event
from .preLoader.Disparity import dsecPreLoader_Disparity
from .preLoader.CSVtimestamp import CSVtimestamp
from .utils.EventToVoxel import events_to_voxel


class DSECSequence_ematch_disparity(data.Dataset):
    def __init__(self, cfgs, name):
        super(DSECSequence_ematch_disparity).__init__()
        # File
        self.root_path = cfgs['root_path']
        self.split = cfgs['split']
        self.name = name

        # Data
        self.dt = cfgs['dt']
        # self.interval = cfgs['interval']
        self.voxel_bins = cfgs['voxel_bins']
        self.crop_size = cfgs['crop_size']
        self.augmentor = Augmentor(self.crop_size) if self.split == 'train' else None

        # Cache
        self.Cache_path = cfgs['Cache_path']

        # Data Loading
        if self.Cache_path is None:
            self.preLoader_event_left = dsecPreLoader_Event(root_path=self.root_path, name=name, split=self.split, location='left', ToMemeory=False)
            self.preLoader_event_right = dsecPreLoader_Event(root_path=self.root_path, name=name, split=self.split, location='right', ToMemeory=False)
        if self.split == 'train':
            self.preLoader_disparity = dsecPreLoader_Disparity(root_path=self.root_path, name=name, split=self.split, ToMemeory=False)
            self.len = self.preLoader_disparity.get_len()
        else:
            self.timestamps = CSVtimestamp(self.root_path, name, task='disparity')
            self.len = len(self.timestamps)

    def __getitem__(self, index):
        # 1. Load Events
        if self.Cache_path is not None:
            voxel_0 = np.load( os.path.join(self.Cache_path, self.name, ('%07d' % index) + '_voxel_0.npz'), allow_pickle=True)['arr_0']
            voxel_1 = np.load( os.path.join(self.Cache_path, self.name, ('%07d' % index) + '_voxel_1.npz'), allow_pickle=True)['arr_0']
        else:
            voxel_0, voxel_1 = self.GetEvents(index)

        if self.split == 'train':
            # 2. Disparity
            disparity, valid = self.preLoader_disparity.get_disparity(index)

            # 3. Augment
            if self.augmentor is not None:
                voxel_0, voxel_1, disparity, valid = self.augmentor(voxel_0, voxel_1, disparity, valid)

            # 4. To Tensor
            voxel_0 = torch.from_numpy(voxel_0).permute(2, 0, 1).float()
            voxel_1 = torch.from_numpy(voxel_1).permute(2, 0, 1).float()
            disparity = torch.from_numpy(disparity).float()
            valid = torch.from_numpy(valid).float()
            batch = {
                'voxel_0': voxel_0,
                'voxel_1': voxel_1,
                'target': disparity,
                'valid': valid,
            }
        else:
            # 2. File index
            file_index = int(self.timestamps[index][1])

            # 3. To Tensor
            voxel_0 = torch.from_numpy(voxel_0).permute(2, 0, 1).float()
            voxel_1 = torch.from_numpy(voxel_1).permute(2, 0, 1).float()
            batch = {
                'voxel_0': voxel_0,
                'voxel_1': voxel_1,
                'file_index': file_index
            }

        return batch
    
    def GetEvents(self, index):
        # 1. Get T1 T2 (from Flow Timestamps)
        if self.split == 'train':
            timestamps_disparity = self.preLoader_disparity.get_t(index)
        else:
            timestamps_disparity = self.timestamps[index][0]

        # 2. Voxel
        events_0 = self.preLoader_event_left.search_events_fromT(timestamps_disparity - self.dt * 1000, timestamps_disparity)
        voxel_0 = events_to_voxel(events_0, self.voxel_bins, 480, 640, pos=0, normalize=False, standardize=True).transpose(1,2,0) # (B,H,W) -> (H,W,B)
        events_1 = self.preLoader_event_right.search_events_fromT(timestamps_disparity - self.dt * 1000, timestamps_disparity)
        voxel_1 = events_to_voxel(events_1, self.voxel_bins, 480, 640, pos=0, normalize=False, standardize=True).transpose(1,2,0) # (B,H,W) -> (H,W,B)
        return voxel_0, voxel_1
    
    def __len__(self):
        return self.len
    
        
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
