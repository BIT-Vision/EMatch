import os
import numpy as np
import glob
import imageio


class dsecPreLoader_Flow():
    def __init__(self, root_path, name, split='train', location='left', direction='forward', ToMemeory=False):
        self.ToMemory = ToMemeory
        # Locate root path
        assert split == 'train' or split == 'test'
        assert location == 'left'
        Rootpath = 'train_optical_flow' if split == 'train' else 'test_optical_flow'
        Rootpath_flow = os.path.join(root_path, Rootpath, name, 'flow', direction)
        Rootpath_flow_t = os.path.join(root_path, Rootpath, name, 'flow', direction+'_timestamps.txt')

        if ToMemeory:
            self.flows = []
            self.valids = []
            for flow_path in sorted(glob.glob(os.path.join(Rootpath_flow, '*')), key=lambda x: int(str(x).split('.')[0].split(os.path.sep)[-1])): 
                flow_16bit = imageio.imread(flow_path, format='PNG-FI')
                gt_flow, valid2D = self.__flow_16bit_to_float(flow_16bit)
                valid2D = np.asarray(valid2D).astype(np.float64)
                self.flows.append(gt_flow)
                self.valids.append(valid2D)
        else:
            # Load Flow paths
            self.flows_path = []
            for flow_path in sorted(glob.glob(os.path.join(Rootpath_flow, '*')), key=lambda x: int(str(x).split('.')[0].split(os.path.sep)[-1])): 
                self.flows_path.append(flow_path)

        # Load timestamps
        self.timestamps_flow = np.loadtxt(Rootpath_flow_t, delimiter=',')

        # Len
        self.flows_len = len(self.flows) if ToMemeory else len(self.flows_path)


    def get_flow(self, index):
        """
        :returns gt_flow: (H,W,2)
        :returns valid2D: (H,W)
        """
        if self.ToMemory:
            gt_flow = self.flows[index]
            valid2D = self.valids[index]
        else:
            # Load Flow And Valid
            flow_16bit = imageio.imread(self.flows_path[index], format='PNG-FI')
            gt_flow, valid2D = self.__flow_16bit_to_float(flow_16bit)
            valid2D = np.asarray(valid2D).astype(np.float64)

        return gt_flow, valid2D

    def __flow_16bit_to_float(self, flow_16bit: np.ndarray):
        assert flow_16bit.dtype == np.uint16
        assert flow_16bit.ndim == 3
        h, w, c = flow_16bit.shape
        assert c == 3

        valid2D = flow_16bit[..., 2] == 1
        assert valid2D.shape == (h, w)
        assert np.all(flow_16bit[~valid2D, -1] == 0)
        valid_map = np.where(valid2D)

        # to actually compute something useful:
        flow_16bit = flow_16bit.astype('float')

        flow_map = np.zeros((h, w, 2))
        flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
        flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
        return flow_map, valid2D
    
    def get_t(self, index):
        return self.timestamps_flow[index][0], self.timestamps_flow[index][1]
    
    def get_len(self):
        return self.flows_len