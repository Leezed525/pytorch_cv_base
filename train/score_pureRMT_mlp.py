"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/18 19:47
"""
import os
import sys

sys.path.append(os.getcwd())
from lib.actor.LeeNet import LeeNetActor
from lib.models.backbone.pure_RMT import PureRMT
from lib.models.LeeNet.score_pureRMT_mlp import ScorePureRMTMLP
from lib.models.head.mlp import MLP
from lib.trainer.LeeNet_trainer import LeeNetTrainer
from lib.utils.base_funtion import build_dataloaders, get_optimizer_scheduler
from lib.config.cfg_loader import env_setting
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss


def build_model(cfg):
    backbone = PureRMT(cfg=cfg)
    head = MLP(input_dim=10 * cfg.model.pureRMT.embed_dim[-1], hidden_dim=cfg.model.pureRMT.embed_dim[-1], output_dim=4, num_layers=2)
    model = ScorePureRMTMLP(backbone, head, cfg)
    return model


def run():
    cfg = env_setting(cfg_name="score_pureRMT_mlp.yaml")

    loader_train, loader_val = build_dataloaders(cfg)

    net = build_model(cfg)

    focal_loss = FocalLoss()
    objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
    loss_weight = {'giou': cfg.train.GIOU_weight, 'l1': cfg.train.L1_weight, 'focal': 1., 'cls': 1.0}
    actor = LeeNetActor(net=net, objective=objective, loss_weight=loss_weight, cfg=cfg)

    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # location loss 没计算出来

    trainer = LeeNetTrainer(actor=actor, loaders=[loader_train, loader_val], optimizer=optimizer, lr_scheduler=lr_scheduler, cfg=cfg)
    trainer.train(cfg.train.epoch, load_latest=True)


if __name__ == '__main__':
    #
    # print(sys.path)
    # print(os.getcwd())
    run()
