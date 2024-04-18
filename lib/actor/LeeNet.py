"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/14 21:04
"""

from lib.actor.base_actor import BaseActor
import torch
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


class LeeNetActor(BaseActor):
    def __init__(self, net, objective, loss_weight, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.cfg = cfg
        self.batch_size = self.cfg.train.batch_size

        self.device = cfg.train.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.net.to(self.device)

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

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template_list, search_img)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4
        # gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        # gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict

        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(0)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        # gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except Exception as e:
            print(e)
            print("giou loss compute error")
            giou_loss, iou = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        location_loss = torch.tensor(0.0, device=l1_loss.device)
        # if 'score_map' in pred_dict:
        #     location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        # else:
        #     location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
