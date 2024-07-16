import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layer.patch_embed import PatchEmbed
from lib.utils.backbone_utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layer.attn_blocks import CEBlock
from lib.models.layer.vmamba import CrossMambaFusionBlock, ConcatMambaFusionBlock
from lib.models.layer.score import PLScoreLayerUseConv
from lib.models.layer.adapter import Fusion_adapter

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None, new_patch_size=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, flatten=False)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W

        self.search_norm = norm_layer([new_P_H, new_P_W])

        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W

        self.template_norm = norm_layer([new_P_H, new_P_W])

        """add here, no need use backbone.finetune_track """  #
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        # score function
        self.score = PLScoreLayerUseConv(embed_dim=self.embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)

        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=self.embed_dim,
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=self.embed_dim,
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        self.score_in_layer = nn.ModuleList(
            PLScoreLayerUseConv(embed_dim=self.embed_dim) for i in range(4)
        )

        # self.adapter = Fusion_adapter()

        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]

        # modal_img
        x_modal = x[:, 3:, :, :]
        z_modal = z[:, 3:, :, :]

        x, z = x_rgb, z_rgb

        x, _ = self.patch_embed(x)  # (B, 768,16,16)
        z, _ = self.patch_embed(z)  # (B, 768,8,8)

        x_modal, _ = self.patch_embed(x_modal)
        z_modal, _ = self.patch_embed(z_modal)

        # x = self.search_norm(x)
        # z = self.template_norm(z)
        #
        # x_modal = self.search_norm(x_modal)
        # z_modal = self.template_norm(z_modal)

        mx = self.score(x, x_modal)  # (B, 768,16,16)
        mz = self.score(z, z_modal)  # (B, 768,8,8)

        mx = self.search_norm(mx)
        mz = self.template_norm(mz)

        zw, zh = mz.shape[2], mz.shape[3]
        xw, xh = mx.shape[2], mx.shape[3]

        # (B,C,H,W) -> (B,H*W,C)
        x = x.flatten(2).transpose(1, 2)
        z = z.flatten(2).transpose(1, 2)

        x_modal = x_modal.flatten(2).transpose(1, 2)
        z_modal = z_modal.flatten(2).transpose(1, 2)

        mx = mx.flatten(2).transpose(1, 2)
        mz = mz.flatten(2).transpose(1, 2)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # print("x.shape", x.shape)  # x.shape torch.Size([16, 256, 768])
        # print("z.shape", z.shape)  # z.shape torch.Size([16, 64, 768])
        # print("x_modal.shape", x_modal.shape)  # x_modal.shape torch.Size([16, 256, 768])
        # print("z_modal.shape", z_modal.shape)  # z_modal.shape torch.Size([16, 64, 768])

        z += self.pos_embed_z
        x += self.pos_embed_x

        z_modal += self.pos_embed_z
        x_modal += self.pos_embed_x

        mz += self.pos_embed_z
        mx += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
            x_modal += self.search_segment_pos_embed
            z_modal += self.template_segment_pos_embed
            mx += self.search_segment_pos_embed
            mz += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        x_modal = combine_tokens(z_modal, x_modal, mode=self.cat_mode)
        mx = combine_tokens(mz, mx, mode=self.cat_mode)

        # print("after combine :x", x.shape)  # x.shape torch.Size([16, 320, 768])

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            x_modal = torch.cat([cls_tokens, x_modal], dim=1)

        x = self.pos_drop(x)
        x_modal = self.pos_drop(x_modal)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        global_index_t_modal = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t_modal = global_index_t_modal.repeat(B, 1)

        global_index_s_modal = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s_modal = global_index_s_modal.repeat(B, 1)

        global_index_mt = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_mt = global_index_mt.repeat(B, 1)

        global_index_ms = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_ms = global_index_ms.repeat(B, 1)

        removed_indexes_s = []
        removed_indexes_s_modal = []

        for i, blk in enumerate(self.blocks):  # -> (B,320,768)
            x, global_index_t, global_index_s, removed_index_s, attn = blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            x_modal, global_index_t_modal, global_index_s_modal, removed_index_s_modal, attn_modal = blk(x_modal, global_index_t_modal,
                                                                                                         global_index_s_modal, mask_x,
                                                                                                         ce_template_mask, ce_keep_rate)
            # 使用adapter融合数据
            # x, x_modal = self.adapter(x, x_modal)

            if i % 4 == 3:
                # sigma fusion
                x, z = self.token2wh(x, xw, xh, zw, zh, B)  # x -> (B, 16,16, 768) z - > (B, 8,8, 768)
                x_modal, z_modal = self.token2wh(x_modal, xw, xh, zw, zh, B)
                mx, mz = self.token2wh(mx, xw, xh, zw, zh, B)

                # score in layer start
                score_x = self.score_in_layer[i // 4](x.permute(0, 3, 1, 2), x_modal.permute(0, 3, 1, 2))
                score_z = self.score_in_layer[i // 4](z.permute(0, 3, 1, 2), z_modal.permute(0, 3, 1, 2))

                mx += score_x.permute(0, 2, 3, 1)
                mz += score_z.permute(0, 2, 3, 1)

                # score in layer end

                x_f, x_f_modal = self.cross_mamba[i // 4](x, x_modal)
                x_fuse = self.channel_attn_mamba[i // 4](x_f, x_f_modal)
                mx += x_fuse

                z_f, z_f_modal = self.cross_mamba[i // 4](z, z_modal)
                z_fuse = self.channel_attn_mamba[i // 4](z_f, z_f_modal)
                mz += z_fuse

                x = self.wh2token(x, z, xw, xh, zw, zh, B)
                x_modal = self.wh2token(x_modal, z_modal, xw, xh, zw, zh, B)
                mx = self.wh2token(mx, mz, xw, xh, zw, zh, B)

                # 加个norm防止过拟合
                x_modal = self.norm(x_modal)
                # mx = self.norm(mx)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
                removed_indexes_s_modal.append(removed_index_s_modal)

        x = self.norm(x)
        x_modal = self.norm(x_modal)
        mx = self.norm(mx)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        lens_x_modal_new = global_index_s_modal.shape[1]
        lens_z_modal_new = global_index_s_modal.shape[1]
        lens_mx_new = global_index_ms.shape[1]
        lens_mz_new = global_index_ms.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]
        z_modal = x_modal[:, :lens_z_modal_new]
        x_modal = x_modal[:, lens_z_modal_new:]
        mz = mx[:, :lens_mz_new]
        mx = mx[:, lens_mz_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        if removed_indexes_s_modal and removed_indexes_s_modal[0] is not None:
            removed_indexes_cat_modal = torch.cat(removed_indexes_s_modal, dim=1)

            pruned_lens_x_modal = lens_x - lens_x_modal_new
            pad_x_modal = torch.zeros([B, pruned_lens_x_modal, x_modal.shape[2]], device=x.device)
            x_modal = torch.cat([x_modal, pad_x_modal], dim=1)
            index_all = torch.cat([global_index_s_modal, removed_indexes_cat_modal], dim=1)
            # recover original token order
            C = x_modal.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x_modal = torch.zeros_like(x_modal).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_modal)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)  # -> (B, 256, 768)
        x_modal = recover_tokens(x_modal, lens_z_modal_new, lens_x, mode=self.cat_mode)
        mx = recover_tokens(mx, lens_mz_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)  # -> (b,320,768)
        x_modal = torch.cat([z_modal, x_modal], dim=1)
        mx = torch.cat([mz, mx], dim=1)

        x = x + x_modal + mx

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, )

        return x, aux_dict

    def token2wh(self, token, xw, xh, zw, zh, B):
        return token[:, zw * zh:, :].reshape(B, xw, xh, -1), token[:, :zw * zh, :].reshape(B, zw, zh, -1)

    def wh2token(self, x, z, xw, xh, zw, zh, B):
        x = x.reshape(B, xw * xh, -1)
        z = z.reshape(B, zw * zh, -1)
        return combine_tokens(z, x, mode=self.cat_mode)


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print("missing_keys")
            print(missing_keys)
            print("unexpected_keys")
            print(unexpected_keys)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
