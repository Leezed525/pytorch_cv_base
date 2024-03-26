"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/26 18:49
"""
def names_to_datasets(name_list: list, setting, image_loader):
    datasets = []
    valid_dataset_name = ["LasHeR_all","LasHeR_train","LasHeR_val", "DepthTrack_train", "DepthTrack_val", "VisEvent"]
    for name in name_list:
        assert name in valid_dataset_name, "Invalid dataset name:{}".format(name)
        # if name == "DepthTrack_train":