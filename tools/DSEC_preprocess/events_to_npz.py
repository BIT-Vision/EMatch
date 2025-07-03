import argparse
import os
import numpy as np
from tqdm import tqdm

from preLoader.Event import dsecPreLoader_Event
from preLoader.Flow import dsecPreLoader_Flow
from preLoader.CSVtimestamp import CSVtimestamp

from utils.EventToVoxel import events_to_voxel

# SplitNames
names_opticalFlow_Train = ['thun_00_a', 'zurich_city_01_a', 'zurich_city_02_a', 'zurich_city_02_c',
                        'zurich_city_02_d', 'zurich_city_02_e', 'zurich_city_03_a', 'zurich_city_05_a',
                        'zurich_city_05_b', 'zurich_city_06_a', 'zurich_city_07_a', 'zurich_city_08_a',
                        'zurich_city_09_a', 'zurich_city_10_b', 'zurich_city_10_a', 'zurich_city_11_a', 
                        'zurich_city_11_b', 'zurich_city_11_c']
names_opticalFlow_Test = ['interlaken_00_b','interlaken_01_a','thun_01_a','thun_01_b',
                        'zurich_city_12_a','zurich_city_14_c','zurich_city_15_a']

def convert_npz(preLoader_event, start, end, dt, bins, dir_path, idx):
    # 读取事件
    events_0 = preLoader_event.search_events_fromT(start-dt, start)
    events_1 = preLoader_event.search_events_fromT(end-dt, end)

    # 转化为体素
    voxel_0 = events_to_voxel(events_0, bins, 480, 640, pos=0, normalize=False, standardize=True).transpose(1,2,0) # (B,H,W) -> (H,W,B)
    voxel_1 = events_to_voxel(events_1, bins, 480, 640, pos=0, normalize=False, standardize=True).transpose(1,2,0) # (B,H,W) -> (H,W,B)

    # 保存
    np.savez_compressed(os.path.join(dir_path, ('%07d' % idx) + '_voxel_0.npz'), voxel_0)
    np.savez_compressed(os.path.join(dir_path, ('%07d' % idx) + '_voxel_1.npz'), voxel_1)

def main(args):
    dt = args.dt * 1000
    bins = args.bins

    if args.split == 'test':
        for name in tqdm(names_opticalFlow_Test):
            dir_path = os.path.join(args.save_path, name)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            preLoader_event = dsecPreLoader_Event(root_path=args.root_path, name=name, split='test', location='left', ToMemeory=False)
            timestamps = CSVtimestamp(args.root_path, name, task='flow')

            # 所有待转化文件
            for idx in tqdm(range(len(timestamps))):
                start = timestamps[idx][0]
                end = timestamps[idx][1]
                convert_npz(preLoader_event, start, end, dt, bins, dir_path, idx)

    elif args.split == 'train':
        for name in tqdm(names_opticalFlow_Train):
            dir_path = os.path.join(args.save_path, name)
            if not os.path.exists(dir_path): os.makedirs(dir_path)

            preLoader_event = dsecPreLoader_Event(root_path=args.root_path, name=name, split='train', location='left', ToMemeory=False)
            preLoader_flow = dsecPreLoader_Flow(root_path=args.root_path, name=name, split='train', location='left', direction='forward', ToMemeory=False)
            
            # 所有待转化文件
            for idx in tqdm(range(preLoader_flow.get_len())):
                start = preLoader_flow.get_t(idx)[0]
                end = preLoader_flow.get_t(idx)[1]
                convert_npz(preLoader_event, start, end, dt, bins, dir_path, idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/ssd/zhangpengjie/DSEC', type=str)
    parser.add_argument('--save_path', default='/ssd/zhangpengjie/DSEC/cache/DSECSequence_flow_eventImage/test/left/voxel_dt100_bins15_us/', type=str)
    parser.add_argument('--split', default='test', type=str)

    parser.add_argument('--dt', default=100, type=int)
    parser.add_argument('--bins', default=15, type=int)
    args = parser.parse_args()

    main(args)
