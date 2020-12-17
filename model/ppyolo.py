#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
import torch


class PPYOLO(torch.nn.Module):
    def __init__(self, backbone, head):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, im_size, eval=True, gt_box=None, gt_label=None, gt_score=None, targets=None):
        body_feats = self.backbone(x)
        if eval:
            out = self.head.get_prediction(body_feats, im_size)
        else:
            out = self.head.get_loss(body_feats, gt_box, gt_label, gt_score, targets)
        return out

    def add_param_group(self, param_groups, base_lr, base_wd):
        self.backbone.add_param_group(param_groups, base_lr, base_wd)
        self.head.add_param_group(param_groups, base_lr, base_wd)




