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
import copy
from collections import OrderedDict



class PPYOLO(torch.nn.Module):
    def __init__(self, backbone, head, ema_decay=0.9998):
        super(PPYOLO, self).__init__()
        self.backbone = backbone
        self.head = head
        self.ema_decay = ema_decay
        self.ema_state_dict = OrderedDict()       # 使用cpu内存，不占用显存
        self.current_state_dict = OrderedDict()   # 使用cpu内存，不占用显存

    def forward(self, x, im_size, eval=True, gt_box=None, gt_label=None, gt_score=None, targets=None):
        body_feats = self.backbone(x)
        if eval:
            out = self.head.get_prediction(body_feats, im_size)
        else:
            out = self.head.get_loss(body_feats, gt_box, gt_label, gt_score, targets)
        return out

    def init_ema_state_dict(self):
        self.ema_state_dict = copy.deepcopy(self.state_dict())
        for k, v in self.ema_state_dict.items():
            v2 = v * 0   # 参数用0初始化（包括可训练参数、bn层的均值、bn层的方差）
            self.ema_state_dict[k] = v2

    def update_ema_state_dict(self, thres_steps):
        decay2 = (1.0 + thres_steps) / (10.0 + thres_steps)
        ema_decay = min(self.ema_decay, decay2)  # 真实的衰减率
        temp_dict = copy.deepcopy(self.state_dict())
        for k, v in temp_dict.items():    # bn层的均值、方差也受该全局ema管理（尽管它们有自己的滑动平均）
            v = v.cpu()   # 放进cpu内存
            v2 = self.ema_state_dict[k]   # ema中旧的值
            v2 = ema_decay * v2 + (1.0 - ema_decay) * v   # ema中新的值
            v2 = v2 / (1.0 - ema_decay ** (thres_steps + 1))   # 偏置校正
            self.ema_state_dict[k] = v2   # ema写入新的值

    def apply_ema_state_dict(self):
        self.current_state_dict = copy.deepcopy(self.state_dict())   # 备份
        temp_dict = copy.deepcopy(self.ema_state_dict)
        for k, v in self.current_state_dict.items():
            v = v.cpu()   # 放进cpu内存
            self.current_state_dict[k] = v
        for k, v in temp_dict.items():
            v = v.cuda()   # 放进显存
            temp_dict[k] = v
        self.load_state_dict(temp_dict)

    def restore_current_state_dict(self):
        temp_dict = copy.deepcopy(self.current_state_dict)
        for k, v in temp_dict.items():
            v = v.cuda()   # 放进显存
            temp_dict[k] = v
        self.load_state_dict(temp_dict)




