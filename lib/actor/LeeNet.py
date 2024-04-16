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

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.cfg.data.template.number):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)

        out_dict = self.net(template_list, search_img)

        return out_dict

    def compute_losses(self, out_dict, data):
        return 1,2
