"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/15 17:28
"""
import torch
import torch.nn as nn
from lib.models.layer.patch_embed import PatchEmbed
from lib.models.layer.score import ScoreLayerUseConv
from lib.config.cfg_loader import CfgLoader
from lib.models.layer.RMT import BasicLayer, PatchMerging
from lib.utils.backbone_utils import combine_tokens


class PureRMT(nn.Module):
    def __init__(self, patch_size=16, norm_layer=nn.LayerNorm, layer_init_values=1e-6, cfg: CfgLoader = None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = cfg.model.pureRMT.embed_dim
        depths = cfg.model.pureRMT.depth
        self.num_layers = len(depths)
        num_heads = cfg.model.pureRMT.num_heads
        init_values = cfg.model.pureRMT.init_values
        head_ranges = cfg.model.pureRMT.head_ranges
        mlp_ratios = cfg.model.pureRMT.mlp_ratios
        drop_path_rate = cfg.model.pureRMT.drop_path_rate
        chunkwise_recurrents = cfg.model.pureRMT.chunkwise_recurrents
        self.combine_token_mode = cfg.model.pureRMT.combine_token_mode

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=3, embed_dim=self.embed_dim[0], flatten=False)
        self.score = ScoreLayerUseConv(embed_dim=self.embed_dim[0])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.RMT_layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                embed_dim=self.embed_dim[i_layer],
                out_dim=self.embed_dim[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=head_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * self.embed_dim[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
                layerscale=False,
                layer_init_values=layer_init_values
            )
            self.RMT_layers.append(layer)

        self.combine_token = combine_tokens

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
        z = t_positive_mask * (0.9 * z_rgb + 0.1 * z_modal) + t_uncertain_mask * (0.5 * z_rgb + 0.5 * z_modal) + t_negative_mask * (
                0.1 * z_rgb + 0.9 * z_modal)

        x = s_positive_mask * (0.9 * x_rgb + 0.1 * x_modal) + s_uncertain_mask * (0.5 * x_rgb + 0.5 * x_modal) + s_negative_mask * (
                0.1 * x_rgb + 0.9 * x_modal)

        z = z.permute(0, 2, 3, 1).contiguous()  # -> (B,P_N,P_N,C)
        x = x.permute(0, 2, 3, 1).contiguous()  # -> (B,P_N,P_N,C)

        # in pure RMT backbone
        for layer in self.RMT_layers:
            z = layer(z)
            x = layer(x)

        x = x.reshape(x.shape[0], -1, x.shape[-1])
        z = z.reshape(z.shape[0], -1, z.shape[-1])

        x = self.combine_token(z, x, mode=self.combine_token_mode)

        return x


"""
z shape torch.Size([16, 8, 8, 96])
x shape torch.Size([16, 20, 20, 96])
layer 1
z shape torch.Size([16, 4, 4, 192])
x shape torch.Size([16, 10, 10, 192])
layer 2
z shape torch.Size([16, 2, 2, 384])
x shape torch.Size([16, 5, 5, 384])
layer 3
z shape torch.Size([16, 1, 1, 768])
x shape torch.Size([16, 3, 3, 768])
layer 4
z shape torch.Size([16, 1, 1, 768])
x shape torch.Size([16, 3, 3, 768])
z shape torch.Size([16, 1, 768])
x shape torch.Size([16, 9, 768])
x shape torch.Size([16, 10, 768])
"""
