"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/16 20:26
"""
import torch
import torch.nn as nn
from lib.config.cfg_loader import CfgLoader


class ScorePureRMTMLP(nn.Module):
    def __init__(self, backbone, box_head, cfg: CfgLoader = None):
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.cfg = cfg

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        out = self.backbone(template, search)
        predict_box = self.box_head(out)
        return predict_box
