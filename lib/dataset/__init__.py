"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/26 18:49
"""
from lib.dataset.DepthTrack import DepthTrack
from lib.config.cfg_loader import env_setting


def names_to_datasets(name_list: list, setting, image_loader):
    cfg = env_setting(cfg_name=None)
    datasets = []
    valid_dataset_name = ["LasHeR_all", "LasHeR_train", "LasHeR_val", "DepthTrack_train", "DepthTrack_val", "VisEvent"]
    for name in name_list:
        assert name in valid_dataset_name, "Invalid dataset name:{}".format(name)
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(root=cfg.dataset.DepthTrack.train.dir, dtype='rgbcolormap', split='train'))
        if name == 'DepthTrack_val':
            datasets.append(DepthTrack(root=cfg.dataset.DepthTrack.val.dir, dtype='rgbcolormap', split='val'))
