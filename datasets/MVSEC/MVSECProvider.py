import os
import torch.utils.data

from .MVSECSequence_ematch_flow import MVSECSequence_ematch_flow
from .MVSECSequence_ematch_disparity import MVSECSequence_ematch_disparity

class MVSECProvider:
    def __init__(self, cfgs):
        self.class_dict = {'MVSECSequence_ematch_flow':MVSECSequence_ematch_flow, 
                           'MVSECSequence_ematch_disparity':MVSECSequence_ematch_disparity}

        # Load configs
        self.SequenceClass = self.class_dict[cfgs['SequenceClass']]

        # Load Sequences
        self.MVSECSequences = []
        self.SequenceNames = cfgs['SequenceNames']
        for name in self.SequenceNames:
            self.MVSECSequences.append(self.SequenceClass(cfgs, name))

    def get_datasets(self, F_list=False):
        if F_list:
            return self.MVSECSequences, self.SequenceNames
        else:
            return torch.utils.data.ConcatDataset(self.MVSECSequences)
