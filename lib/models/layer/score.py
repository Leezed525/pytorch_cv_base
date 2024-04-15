"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/14 22:17
"""
import torch.nn as nn


class ScoreLayer(nn.Module):
    def __init__(self,patch_nums,threshold1 = 0.33,threshold2 = 0.66):
        super().__init__()
