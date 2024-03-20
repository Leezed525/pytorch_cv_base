"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/20 20:52
"""


def names_to_datasets(name_list: list, setting, image_loader):
    datasets = []
    valid_dataset_name = ["LasHeR_all","LasHeR_train","LasHeR_val", "depthTracking_train", "depthTracking_val", "VisEvent"]
    for name in name_list:
        assert name in valid_dataset_name, "Invalid dataset name:{}".format(name)
