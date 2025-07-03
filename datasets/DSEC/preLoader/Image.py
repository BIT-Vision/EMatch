import glob
import imageio
import numpy as np
import os
import argparse
from pathlib import Path
import cv2
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rot

class Transform:
    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        assert self._translation.size == 3
        self._rotation = rotation

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    def R_matrix(self):
        return self._rotation.as_matrix()

    def R(self):
        return self._rotation

    def t(self):
        return self._translation

    def T_matrix(self) -> np.ndarray:
        return self._T_matrix_from_tR(self._translation, self._rotation.as_matrix())

    def q(self):
        # returns (x, y, z, w)
        return self._rotation.as_quat()

    def euler(self):
        return self._rotation.as_euler('xyz', degrees=True)

    def __matmul__(self, other):
        # a (self), b (other)
        # returns a @ b
        #
        # R_A | t_A   R_B | t_B   R_A @ R_B | R_A @ t_B + t_A
        # --------- @ --------- = ---------------------------
        # 0   | 1     0   | 1     0         | 1
        #
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)

    def inverse(self):
        #           R_AB  | A_t_AB
        # T_AB =    ------|-------
        #           0     | 1
        #
        # to be converted to
        #
        #           R_BA  | B_t_BA    R_AB.T | -R_AB.T @ A_t_AB
        # T_BA =    ------|------- =  -------|-----------------
        #           0     | 1         0      | 1
        #
        # This is numerically more stable than matrix inversion of T_AB
        rotation = self._rotation.inv()
        translation = - rotation.apply(self._translation)
        return Transform(translation, rotation)


class dsecPreLoader_Image():
    def __init__(self, root_path, name, split='train', location='left', ToMemeory=False):
        self.ToMemory = ToMemeory
        # Locate root path
        assert split == 'train' or split == 'test'
        Rootpath = 'train_images' if split == 'train' else 'test_images'
        Rootpath_img = os.path.join(root_path, Rootpath, name, 'images', location, 'rectified')
        Rootpath_img_t = os.path.join(root_path, Rootpath, name, 'images', 'timestamps.txt')

        # Load images
        if ToMemeory:
            # Load Image
            self.images = []
            for image_path in sorted(glob.glob(os.path.join(Rootpath_img, '*')), key=lambda x: int(str(x).split('.')[0].split(os.path.sep)[-1])): 
                img = imageio.imread(image_path)
                self.images.append(img)
        else:
            # Load Image path
            self.images_path = []
            for image_path in sorted(glob.glob(os.path.join(Rootpath_img, '*')), key=lambda x: int(str(x).split('.')[0].split(os.path.sep)[-1])): 
                self.images_path.append(image_path)

        # Load image ts
        self.images_timestamp = np.loadtxt(Rootpath_img_t, delimiter=',')
        
        # Length
        self.images_len = len(self.images) if ToMemeory else len(self.images_path)

        # Get mapping for this sequence:
        Rootpath_conf = 'train_calibration' if split == 'train' else 'test_calibration' 
        confpath = os.path.join(root_path, Rootpath_conf , name, 'calibration', 'cam_to_cam.yaml')
        conf = OmegaConf.load(confpath)

        K_r0 = np.eye(3)
        K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect0']['camera_matrix']
        K_r1 = np.eye(3)
        K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect1']['camera_matrix']

        R_r0_0 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect0']))
        R_r1_1 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect1']))

        T_r0_0 = Transform.from_rotation(R_r0_0)
        T_r1_1 = Transform.from_rotation(R_r1_1)
        T_1_0 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_10']))

        T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
        R_r1_r0_matrix = T_r1_r0.R().as_matrix()
        P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)

        ht = 480
        wd = 640
        # coords: ht, wd, 2
        coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
        # coords_hom: ht, wd, 3
        coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
        # mapping: ht, wd, 3
        mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
        # mapping: ht, wd, 2
        mapping = (mapping/mapping[..., -1][..., None])[..., :2]
        self.mapping = mapping.astype('float32')

        # T to idx
        self.t_idx = {}
        for i in range(len(self.images_timestamp)):
            t = self.images_timestamp[i]
            self.t_idx[t]=i

    def get_image(self, index:int, align_to_event:bool=False) -> imageio.core.util.Array:
        """
        :return rgb: [1080,1440,3] or [480,640,3] if align_to_event is True
        """
        if self.ToMemory:
            img = self.images[index]
        else:
            img = imageio.imread(self.images_path[index])

        if align_to_event:
            img = cv2.remap(img, self.mapping, None, interpolation=cv2.INTER_CUBIC)

        return img

    def get_t(self, index:int) -> int:
        return self.images_timestamp[index]
    
    def t_to_idx(self, t):
        return self.t_idx[t]

    def get_len(self):
        return self.images_len
    