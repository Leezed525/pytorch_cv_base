"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/19 19:34
"""
import random
import torch.utils.data
from lib.common.tensor import TensorDict


def no_processing(data):
    """
    默认数据处理函数，就是不处理
    :param data: 原始数据
    :return:  原始数据
    """
    return data


class TrackingSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap, num_search_frames, num_template_frames=1, processing=no_processing,
                 frame_sample_mode='causal', train_cls=False, pos_prob=0.5):
        """

        :param datasets: 训练要用的数据集
        :param p_datasets: 每个数据集被使用的可能性
        :param samples_per_epoch: 每个epoch采样数
        :param max_gap: 采样是随机选择一帧，然后前gap帧作为train帧，后gap帧为test帧
        :param num_search_frames:
        :param num_template_frames:
        :param processing:数据预处理函数
        :param frame_sample_mode:
        :param train_cls: 是否训练分类
        :param pos_prob:分类时抽样正样本的概率
        """

        self.datasets = datasets
        self.train_cls = train_cls
        self.pos_prob = pos_prob

        # 如果没有指定数据集概率，就同一概率
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        p_total = sum(p_datasets)

        # 生成概率
        self.p_datasets = [p / p_total for p in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):

        valid = False

        while not valid:
            # 选择一个数据集
            dataset = random.choices(self.datasets, weights=self.p_datasets)[0]
