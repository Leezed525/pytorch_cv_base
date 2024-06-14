"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/25 15:42
"""
import torch
import torch.nn as nn
from lib.models.layer.patch_embed import PatchEmbed
from lib.models.layer.score import PLScoreLayerUseConv
from lib.config.cfg_loader import CfgLoader
from lib.models.layer.RMT import BasicLayer, PatchMerging
from lib.utils.backbone_utils import combine_tokens
from lib.models.layer.vmamba import CrossMambaFusionBlock, ConcatMambaFusionBlock


class PLScoreRMT(nn.Module):
    def __init__(self, patch_size=16, norm_layer=nn.LayerNorm, layer_init_values=1e-6, down_sample: PatchMerging = None, cfg: CfgLoader = None):
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
        self.score = PLScoreLayerUseConv(embed_dim=self.embed_dim[0])

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
                downsample=down_sample if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
                layerscale=False,
                layer_init_values=layer_init_values
            )
            self.RMT_layers.append(layer)

        self.combine_token = combine_tokens

        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=self.embed_dim[i + 1] if i < 3 else self.embed_dim[-1],
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=self.embed_dim[i + 1] if i < 3 else self.embed_dim[-1],
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

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
        mz = self.score(z_rgb, z_modal)  # (B,1,P_N,P_N)
        mx = self.score(x_rgb, x_modal)

        mz = mz.permute(0, 2, 3, 1).contiguous()  # -> (B,P_N,P_N,C)
        mx = mx.permute(0, 2, 3, 1).contiguous()  # -> (B,P_N,P_N,C)

        x_modal = x_modal.permute(0, 2, 3, 1).contiguous()
        z_modal = z_modal.permute(0, 2, 3, 1).contiguous()

        x_rgb = x_rgb.permute(0, 2, 3, 1).contiguous()
        z_rgb = z_rgb.permute(0, 2, 3, 1).contiguous()

        # in pure RMT backbone
        for i, layer in enumerate(self.RMT_layers):
            mz = layer(mz)
            mx = layer(mx)

            x_rgb = layer(x_rgb)
            z_rgb = layer(z_rgb)

            x_modal = layer(x_modal)
            z_modal = layer(z_modal)

            # mz = mz + self.score(z_rgb.permute(0, 3, 1, 2), z_modal.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # mx = mx + self.score(x_rgb.permute(0, 3, 1, 2), x_modal.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


            x_f_rgb, x_f_modal = self.cross_mamba[i](x_rgb, x_modal)
            x_fuse = self.channel_attn_mamba[i](x_f_rgb, x_f_modal)
            mx += x_fuse

            z_f_rgb, z_f_modal = self.cross_mamba[i](z_rgb, z_modal)
            z_fuse = self.channel_attn_mamba[i](z_f_rgb, z_f_modal)
            mz += z_fuse

        mx = mx.reshape(mx.shape[0], -1, mx.shape[-1])  # -> (B,9,C)
        mz = mz.reshape(mz.shape[0], -1, mz.shape[-1])  # -> (B,1,C)

        mx = self.combine_token(mz, mx, mode=self.combine_token_mode)

        return mx
