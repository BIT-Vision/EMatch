import os
import h5py
import numpy as np

import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH


class dsecPreLoader_Event():
    def __init__(self, root_path, name, split='train', location='left', ToMemeory=False):
        self.ToMemory = ToMemeory
        # Locate root path
        assert split == 'train' or split == 'test'
        Rootpath = 'train_events' if split == 'train' else 'test_events'
        Rootpath_events = os.path.join(root_path, Rootpath, name, 'events', location, 'events.h5')
        Rootpath_rectify = os.path.join(root_path, Rootpath, name, 'events', location, 'rectify_map.h5')

        # Load Events
        if ToMemeory:
            self.event_raw = h5py.File(Rootpath_events, 'r')
            self.x = self.event_raw['events']['x'][:]
            self.y = self.event_raw['events']['y'][:]
            self.t = self.event_raw['events']['t'][:]
            self.p = self.event_raw['events']['p'][:].astype(float) * 2 - 1
            self.ms_to_idx = self.event_raw['ms_to_idx'][:]
        else:
            # Load Event data
            self.event_raw = h5py.File(Rootpath_events, 'r')
        
        # Load Toffset and rectifyMap
        # The time offset in microseconds that must be added to the timestamps of the events. 
        #     By doing so, the event timestamps are in the same clock as the image timestamps.
        self.event_T = self.event_raw['t_offset'][()]
        self.rectify_event_maps = h5py.File(Rootpath_rectify, 'r')['rectify_map'][()]

        # Len
        self.event_len = len(self.x) if ToMemeory else len(self.event_raw['events']['x'])
        
    def get_events(self, start:int, end:int, rectify:bool=True) -> np.ndarray:
        """
        :return e: A [N,4] list of (x,y,t,p)
          - (x,y) is in range of (640,480)
          - t is dt relative to t_offset
          - p = [1,-1]
        """
        # Load events
        if self.ToMemory:
            x = self.x[int(start):int(end + 1)]
            y = self.y[int(start):int(end + 1)]
            t = self.t[int(start):int(end + 1)]
            p = self.p[int(start):int(end + 1)]
        else:
            x = self.event_raw['events']['x'][int(start):int(end + 1)]
            y = self.event_raw['events']['y'][int(start):int(end + 1)]
            t = self.event_raw['events']['t'][int(start):int(end + 1)]
            p = self.event_raw['events']['p'][int(start):int(end + 1)].astype(float) * 2 - 1
        
        # Recitfy
        if rectify:
            rectify_map = self.rectify_event_maps
            rectify_map = rectify_map[y, x]
            x_rect = rectify_map[:, 0]
            y_rect = rectify_map[:, 1]

            mask = (x_rect >= 0) & (x_rect < 640) & (y_rect >= 0) & (y_rect < 480)
            x_rect = x_rect[mask]
            y_rect = y_rect[mask]
            p = p[mask]
            t = t[mask]

            return np.stack((x_rect, y_rect, t, p), axis=1)
            # return x_rect, y_rect, t, p
        else:
            return np.stack((x, y, t, p), axis=1)
            # return x, y, t, p

    def get_event_t_offset(self) -> float:
        """
        :return t_offset: The time offset in microseconds that must be added to the timestamps of the events. 
                          By doing so, the event timestamps are in the same clock as the image timestamps.
        """
        return self.event_T
    
    def search_events_fromT(self, T_start:int, T_end:int, F_us:bool=True) -> np.ndarray:
        """
        :param T_start,T_end: The timestampes from clock of image
        :param F_us: if True, the timestampes of events will be precise to us. 
                     if False,                                          to ms.
        :returns e: A [N,4] list of (x,y,t,p)
        """
        # Load ms_to_idx
        if self.ToMemory:
            ms_to_idx = self.ms_to_idx
        else:
            ms_to_idx = self.event_raw['ms_to_idx']

        # Look up
        T_start = T_start - self.event_T if T_start - self.event_T >= 0 else 0
        T_end = T_end - self.event_T if T_end - self.event_T <= (ms_to_idx.shape[0]-2) * 1000 else  (ms_to_idx.shape[0]-2) * 1000
        if F_us == True:
            # 1. Search start
            start_left = ms_to_idx[int(T_start / 1000)]
            start_right = ms_to_idx[int(T_start / 1000 + 1)] - 1
            while start_left <= start_right:
                strat_mid = int((start_left + start_right) / 2)
                t = self.event_raw['events']['t'][strat_mid]
                if t >= T_start:
                    start_right = strat_mid - 1
                elif t < T_start:
                    start_left = strat_mid + 1
            start = start_left

            # 2. Search end
            end_left = ms_to_idx[int(T_end / 1000)]
            end_right = ms_to_idx[int(T_end / 1000 + 1)] - 1
            while end_left <= end_right:
                end_mid = int((end_left + end_right) / 2)
                t = self.event_raw['events']['t'][end_mid]
                if t > T_end:
                    end_right = end_mid - 1
                elif t <= T_end:
                    end_left = end_mid + 1
            end = int(end_right)
        else:
            start = ms_to_idx[int(T_start / 1000)]
            end = ms_to_idx[int(T_end / 1000 + 1)]

        return self.get_events(start, end)
    
    def get_len(self):
        return self.event_len
    