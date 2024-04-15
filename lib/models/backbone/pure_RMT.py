"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/15 17:28
"""
import torch.nn as nn
from lib.models.layer.patch_embed import PatchEmbed
from lib.models.layer.score import ScoreLayerUseConv


class PureRMT(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, cfg=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=3, embed_dim=embed_dim, flatten=False)
        self.score = ScoreLayerUseConv(embed_dim=embed_dim)

    def forward(self, z, x):
        # get rgb information (B,C,H,W)
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]

        # get modal information (B,C,H,W)
        x_modal = x[:, 3:, :, :]
        z_modal = z[:, 3:, :, :]

        # patch embedding      ->(B,C:Embed_dim,P_N,P_N) P_N = patch_nums(H / patch_size, W / patch_size)
        x_rgb, _ = self.patch_embed(x_rgb)
        z_rgb, _ = self.patch_embed(z_rgb)

        x_modal, _ = self.patch_embed(x_modal)
        z_modal, _ = self.patch_embed(z_modal)

        # use score function
        t_positive_mask, t_uncertain_mask, t_negative_mask = self.score(z_rgb)  # (B,1,P_N,P_N)
        s_positive_mask, s_uncertain_mask, s_negative_mask = self.score(x_rgb)

        # the factor 0.9 0.1 0.5 may can be learned by the model itself
        t = t_positive_mask * (0.9 * z_rgb + 0.1 * z_modal) + t_uncertain_mask * (0.5 * z_rgb + 0.5 * z_modal) + t_negative_mask * (
                0.1 * z_rgb + 0.9 * z_modal)

        x = s_positive_mask * (0.9 * x_rgb + 0.1 * x_modal) + s_uncertain_mask * (0.5 * x_rgb + 0.5 * x_modal) + s_negative_mask * (
                0.1 * x_rgb + 0.9 * x_modal)

        pass
