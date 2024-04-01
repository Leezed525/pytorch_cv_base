"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/26 21:36
"""
import os
import pandas
from lib.dataset.base_video_dataset import BaseVideoDataset
from lib.data.image_loader import jpeg4py_loader_w_failsafe
from lib.config.cfg_loader import CfgLoader


class DepthTrack(BaseVideoDataset):
    def __init__(self, root=None, dtype='rgbcolormap', split='train', image_loader=jpeg4py_loader_w_failsafe, setting: CfgLoader = None):

        super(DepthTrack, self).__init__('DepthTrack', root, image_loader)
        self.dtype = dtype
        self.split = split
        self.sequence_list = self._build_sequence_list()
        self.class_list = self._build_class_list()

    def _build_sequence_list(self):
        # ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        ltr_path = os.path.dirname(os.path.realpath(__file__))
        print(ltr_path)
        file_path = os.path.join(ltr_path, 'data_specs', 'depthtrack_%s.txt' % self.split)
        sequence_list = pandas.read_csv(file_path, header=None).squeeze().values.tolist()
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        class_list = []
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('_')[0]

            if class_name not in class_list:
                class_list.append(class_name)

            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class, class_list


if __name__ == '__main__':
    depthTrack = DepthTrack()
