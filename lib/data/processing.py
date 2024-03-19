"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/19 18:10
"""
import torch
import torchvision.transforms as transforms

from lib.common.tensor import TensorDict


class BaseProcessing:
    """
    基础数据处理类
    主要负责将数据集中的数据进行处理，包括数据增强，数据转换等，最后转换成tensor

    """

    def __init__(self, transform=transforms.ToTensor(), template_transformer=None, search_transformer=None, joint_transformer=None):
        self.transform = {
            'template': transform if template_transformer is None else template_transformer,
            'search': transform if search_transformer is None else search_transformer,
            'joint': transform
        }

    def __call__(self, data: TensorDict):
        raise NotImplementedError
