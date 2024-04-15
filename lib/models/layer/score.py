"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/14 22:17
"""
import torch.nn as nn
from lib.models.layer.RMT import RetBlock, RelPos2d
from lib.models.layer.Conv import Conv


class ScoreLayerUseConv(nn.Module):
    def __init__(self, threshold1=0.33, threshold2=0.66, embed_dim=768, num_heads=12, ffn_dim=96, initial_value=1, heads_range=3):
        super().__init__()
        # self.rel_pos = RelPos2d(embed_dim=embed_dim, num_heads=num_heads, initial_value=initial_value, heads_range=heads_range)
        # self.ret_block = RetBlock(retention='whole', embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
        self.embed_dim = embed_dim
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.conv1 = Conv(embed_dim, embed_dim // 2, 5, 1)
        self.conv2 = Conv(embed_dim // 2, 1, 3, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (B, H, W, C)
        # relpos = self.rel_pos((H, W))
        # x = self.ret_block(x, retention_rel_pos=relpos)
        y = x.permute(0, 3, 1, 2).contiguous()  # (B,H,W,C) -> (B,C,H,W)
        mask = self.softmax(self.conv1(self.conv2(y)))
        positive_mask = mask > self.threshold2
        uncertain_mask = self.threshold1 < mask <= self.threshold2
        negative_mask = mask <= self.threshold1
        return positive_mask, uncertain_mask, negative_mask
