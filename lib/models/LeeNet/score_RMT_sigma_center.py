"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/16 20:26
"""
import torch
import torch.nn as nn
from lib.config.cfg_loader import CfgLoader

from lib.models.head.mlp import MLP


class ScorePureRMTCENTER(nn.Module):
    def __init__(self, backbone, box_head, cfg: CfgLoader = None):
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.cfg = cfg

        self.mlp = MLP(input_dim=16, hidden_dim=225, output_dim=400, num_layers=5, BN=False)

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        out = self.backbone(template, search)  # out shape: (batch, 13, embed_dim[-1])
        # print(out.shape) # (B,4,4,dim)
        B, H, W, D = out.shape
        out = out.reshape(B, D, H * W)
        out = self.mlp(out)
        B, C, HW = out.shape
        H = W = int(HW ** 0.5)
        out = out.view(B, C, H, W)
        score_map_ctr, bbox, size_map, offset_map = self.box_head(out)
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(-1, 1, 4)
        out = {'pred_boxes': outputs_coord_new,
               'score_map': score_map_ctr,
               'size_map': size_map,
               'offset_map': offset_map}
        return out
