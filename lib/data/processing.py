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
            'joint': joint_transformer
        }

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class ViPTProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, transform=transforms.ToTensor(),
                 template_transformer=None, search_transformer=None, joint_transformer=None, mode='pair'):
        """

        :param search_area_factor:  The size of the search region  relative to the target size.
        :param output_sz: An integer, denoting the size to which the search region is resized. The search region is always square.
        :param center_jitter_factor: A dict containing the amount of jittering to be applied to the target center before extracting the search region. See _get_jittered_box for how the jittering is done.
        :param scale_jitter_factor: A dict containing the amount of jittering to be applied to the target size before extracting the search region. See _get_jittered_box for how the jittering is done.
        :param transform: see BaseProcessing
        :param template_transformer: see BaseProcessing
        :param search_transformer: see BaseProcessing
        :param joint_transformer: see BaseProcessing
        :param mode: Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(transform, template_transformer, search_transformer, joint_transformer)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """
        Jitter the input box
        :param box: input bounding box
        :param mode: string 'template' or 'search' indicating template or search data
        :return: torch.Tensor - jittered box
        """
        raise NotImplementedError
