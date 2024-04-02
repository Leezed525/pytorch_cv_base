"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/2 19:25
"""
import os
import pandas
import numpy as np
from lib.dataset.base_video_dataset import BaseVideoDataset
from lib.data.image_loader import jpeg4py_loader_w_failsafe
from lib.config.cfg_loader import env_setting


class VisEvent(BaseVideoDataset):
    def __init__(self, root=None, dtype='rgbrgb', split='train', image_loader=jpeg4py_loader_w_failsafe, cfg_name=None):
        root = env_setting(cfg_name).dataset.VisEvent[split]['dir'] if root is None else root
        assert split in ['train'], 'Only support train split in VisEvent, got {}'.format(split)
        super().__init__('VisEvent', root, image_loader)

        self.dtype = dtype
        self.split = split
        self.sequence_list = self._build_sequence_list()
        print(self.sequence_list)

    def _build_sequence_list(self):
        ltr_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(ltr_path, 'data_specs', 'VisEvent_%s_list.txt' % self.split)
        sequence_list = pandas.read_csv(file_path, header=None).squeeze().values.tolist()
        return sequence_list

if __name__ == '__main__':
    ev = VisEvent()
