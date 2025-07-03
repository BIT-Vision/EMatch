import os
import numpy as np
import glob
import cv2


class dsecPreLoader_Disparity():
    def __init__(self, root_path, name, split='train', ToMemeory=False):
        self.ToMemory = ToMemeory
        # Locate root path
        assert split == 'train' or split == 'test'
        Rootpath = 'train_disparity' if split == 'train' else 'test_disparity'
        Rootpath_disparity = os.path.join(root_path, Rootpath, name, 'disparity', 'event')
        Rootpath_disparity_t = os.path.join(root_path, Rootpath, name, 'disparity', 'timestamps.txt')

        if ToMemeory:
            self.disparities = []
            self.valids = []
            for disparity_path in sorted(glob.glob(os.path.join(Rootpath_disparity, '*')), key=lambda x: int(str(x).split('.')[0].split(os.path.sep)[-1])): 
                disp_16bit = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH)
                disparity = disp_16bit.astype(np.float32) / 256
                mask = (disparity > 0).astype(np.float32)
                self.disparities.append(disparity)
                self.valids.append(mask)
        else:
            # Load Flow paths
            self.disparities_path = []
            for disparity_path in sorted(glob.glob(os.path.join(Rootpath_disparity, '*')), key=lambda x: int(str(x).split('.')[0].split(os.path.sep)[-1])): 
                self.disparities_path.append(disparity_path)

        # Load timestamps
        self.timestamps_disparity = np.loadtxt(Rootpath_disparity_t)

        # Len
        self.disparities_len = len(self.disparities) if ToMemeory else len(self.disparities_path)

    def get_disparity(self, index):
        """
        :return disparity: (H,W)
        """
        disp_16bit = cv2.imread(self.disparities_path[index], cv2.IMREAD_ANYDEPTH)
        disparity = disp_16bit.astype(np.float32) / 256.
        mask = (disparity > 0).astype(np.float32)

        return disparity, mask

    def get_t(self, index):
        return self.timestamps_disparity[index]
    
    def get_len(self):
        return self.disparities_len
    

if __name__ == '__main__':
    preLoader_disparity = dsecPreLoader_Disparity(root_path='/ssd/zhangpengjie/DSEC', name='interlaken_00_c', split='train', ToMemeory=False)
    disparity = preLoader_disparity.get_disparity(100)
