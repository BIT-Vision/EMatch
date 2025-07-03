import numpy as np
import os


def CSVtimestamp(rootpath, name, task='flow'):
    if task == 'flow':
        timestamps = np.loadtxt(open(os.path.join(rootpath, 'test_forward_optical_flow_timestamps', name + '.csv'), "rb"),delimiter=",",skiprows=1,usecols=[0,1,2])
    elif task == 'disparity':
        timestamps = np.loadtxt(open(os.path.join(rootpath, 'test_disparity_timestamps', name + '.csv'), "rb"), delimiter=",",skiprows=1,usecols=[0,1])
    else:
        raise Exception("task输入错误, 仅支持flow / disparity")
    
    return timestamps
