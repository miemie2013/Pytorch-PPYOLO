#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
from model.losses import *
from model.iou_losses import *
from model.head import *
from model.resnet_vd import *


def select_backbone(name):
    if name == 'Resnet50Vd':
        return Resnet50Vd
    if name == 'Resnet18Vd':
        return Resnet18Vd

def select_head(name):
    if name == 'YOLOv3Head':
        return YOLOv3Head

def select_loss(name):
    if name == 'YOLOv3Loss':
        return YOLOv3Loss
    if name == 'IouLoss':
        return IouLoss
    if name == 'IouAwareLoss':
        return IouAwareLoss

def select_optimizer(name):
    if name == 'Momentum':
        return torch.optim.SGD
    if name == 'Adam':
        return torch.optim.Adam
    if name == 'SGD':
        return torch.optim.SGD





