#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
import copy

from config import *
from tools.cocotools import get_classes, catid2clsid, clsid2catid
import json
import os
import argparse

from tools.cocotools import eval
from model.decode_np import Decode
from model.ppyolo import PPYOLO
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PPYOLO Eval Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=0,
                    choices=[0, 1],
                    help='0 -- ppyolo_2x.py;  1 -- ppyolo_1x.py;  ')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu

if __name__ == '__main__':
    cfg = None
    if config_file == 0:
        cfg = PPYOLO_2x_Config()
    elif config_file == 1:
        cfg = PPYOLO_2x_Config()


    # 读取的模型
    model_path = cfg.eval_cfg['model_path']

    # 是否给图片画框。
    draw_image = cfg.eval_cfg['draw_image']
    draw_thresh = cfg.eval_cfg['draw_thresh']

    # 验证时的批大小
    eval_batch_size = cfg.eval_cfg['eval_batch_size']

    # 验证集图片的相对路径
    eval_pre_path = cfg.val_pre_path
    anno_file = cfg.val_path
    from pycocotools.coco import COCO
    val_dataset = COCO(anno_file)
    val_img_ids = val_dataset.getImgIds()
    images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        images.append(img_anno)

    all_classes = get_classes(cfg.classes_path)
    num_classes = len(all_classes)


    # 创建模型
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)
    Head = select_head(cfg.head_type)
    head = Head(yolo_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
    ppyolo = PPYOLO(backbone, head)
    if use_gpu:
        ppyolo = ppyolo.cuda()
    ppyolo.load_state_dict(torch.load(model_path))
    ppyolo.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k

    _decode = Decode(ppyolo, all_classes, use_gpu, cfg, for_test=False)
    box_ap = eval(_decode, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image, draw_thresh)

