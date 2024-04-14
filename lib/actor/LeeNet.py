"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/14 21:04
"""

from lib.actor.base_actor import BaseActor


class LeeNetActor(BaseActor):
    def __init__(self, net, objective, loss_weight, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.cfg = cfg
        self.batch_size = self.cfg.train.batch_size

    def __call__(self, data):
        """

        :param data: data需要包含‘template_images’（Num_template,batch,C,H,W）, 'search_images' (Num_search,batch,C,H,W),
        :return: loss, status
        """
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self,data):
        pass

    def compute_losses(self, out_dict, data):
        pass