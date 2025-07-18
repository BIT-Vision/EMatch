import torch.utils.data

from .DSECSequence_ematch_flow import DSECSequence_ematch_flow
from .DSECSequence_ematch_disparity import DSECSequence_ematch_disparity

# SplitNames
names_opticalFlow_Train = ['thun_00_a', 'zurich_city_01_a', 'zurich_city_02_a', 'zurich_city_02_c',
                           'zurich_city_02_d', 'zurich_city_02_e', 'zurich_city_03_a', 'zurich_city_05_a',
                           'zurich_city_05_b', 'zurich_city_06_a', 'zurich_city_07_a', 'zurich_city_08_a',
                           'zurich_city_09_a', 'zurich_city_10_b', 'zurich_city_10_a', 'zurich_city_11_a', 
                           'zurich_city_11_b', 'zurich_city_11_c']
names_opticalFlow_Test = ['interlaken_00_b','interlaken_01_a','thun_01_a','thun_01_b',
                          'zurich_city_12_a','zurich_city_14_c','zurich_city_15_a']
names_disparity_Train = ['interlaken_00_c','interlaken_00_d','interlaken_00_e','interlaken_00_f',
                         'interlaken_00_g','thun_00_a','zurich_city_00_a','zurich_city_00_b',
                         'zurich_city_01_a','zurich_city_01_b','zurich_city_01_c','zurich_city_01_d',
                         'zurich_city_01_e','zurich_city_01_f','zurich_city_02_a','zurich_city_02_b',
                         'zurich_city_02_c','zurich_city_02_d','zurich_city_02_e','zurich_city_03_a',
                         'zurich_city_04_a','zurich_city_04_b','zurich_city_04_c','zurich_city_04_d',
                         'zurich_city_04_e','zurich_city_04_f','zurich_city_05_a','zurich_city_05_b',
                         'zurich_city_06_a','zurich_city_07_a','zurich_city_08_a','zurich_city_09_a',
                         'zurich_city_09_b','zurich_city_09_c','zurich_city_09_d','zurich_city_09_e',
                         'zurich_city_10_a','zurich_city_10_b','zurich_city_11_a','zurich_city_11_b',
                         'zurich_city_11_c']
names_disparity_Test = ['interlaken_00_a','interlaken_00_b','interlaken_01_a','thun_01_a',
                        'thun_01_b','zurich_city_12_a','zurich_city_13_a','zurich_city_13_b',
                        'zurich_city_14_a','zurich_city_14_b','zurich_city_14_c','zurich_city_15_a']
names_dict = {'flow':{'train':names_opticalFlow_Train, 'test':names_opticalFlow_Test},
              'disparity':{'train':names_disparity_Train, 'test':names_disparity_Test}}


class DSECProvider:
    def __init__(self, cfgs):
        self.class_dict = {'DSECSequence_ematch_flow': DSECSequence_ematch_flow, 
                           'DSECSequence_ematch_disparity': DSECSequence_ematch_disparity}

        # Load configs
        self.task = cfgs['task']
        self.split = cfgs['split']
        self.SequenceClass = self.class_dict[cfgs['SequenceClass']]

        # Load Sequences
        self.DSECSequences = []
        self.SequenceNames = names_dict[self.task][self.split]
        for name in self.SequenceNames:
            self.DSECSequences.append(self.SequenceClass(cfgs, name))
        
    def get_datasets(self, F_list=False):
        if F_list:
            return self.DSECSequences, self.SequenceNames
        else:
            return torch.utils.data.ConcatDataset(self.DSECSequences)
