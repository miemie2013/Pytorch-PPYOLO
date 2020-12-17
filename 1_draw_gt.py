#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description : 把gt画出来，看有无漏标、错标。
#
# ================================================================
import cv2
import time
import json
import numpy as np
import threading
import os
import shutil
import colorsys
import random
from pycocotools.coco import COCO

from tools.data_process import data_clean

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class Dataset_Config(object):
    def __init__(self):
        # 自定义数据集
        self.train_path = 'annotation_json/voc2012_train.json'
        self.val_path = 'annotation_json/voc2012_val.json'
        self.classes_path = 'data/voc_classes.txt'
        self.train_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        self.val_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径
        self.num_classes = 20                                      # 数据集类别数

        # COCO数据集
        # self.train_path = '../COCO/annotations/instances_train2017.json'
        # self.val_path = '../COCO/annotations/instances_val2017.json'
        # self.classes_path = 'data/coco_classes.txt'
        # self.train_pre_path = '../COCO/train2017/'  # 训练集图片相对路径
        # self.val_pre_path = '../COCO/val2017/'      # 验证集图片相对路径
        # self.num_classes = 80                       # 数据集类别数

    def draw(self, image, boxes, scores, classes, all_classes):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[cl]
            # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            bbox_thick = 1
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            bbox_mess = '%s: %.2f' % (all_classes[cl], score)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    cfg = Dataset_Config()

    # 种类id
    _catid2clsid = {}
    _clsid2catid = {}
    _clsid2cname = {}
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:
        dataset_text = ''
        for line in f2:
            line = line.strip()
            dataset_text += line
        eval_dataset = json.loads(dataset_text)
        categories = eval_dataset['categories']
        for clsid, cate_dic in enumerate(categories):
            catid = cate_dic['id']
            cname = cate_dic['name']
            _catid2clsid[catid] = clsid
            _clsid2catid[clsid] = catid
            _clsid2cname[clsid] = cname
    class_names = []
    num_classes = len(_clsid2cname.keys())
    for clsid in range(num_classes):
        class_names.append(_clsid2cname[clsid])


    result_dir = 'draw_gt'
    if os.path.exists('%s/train/' % result_dir): shutil.rmtree('%s/train/' % result_dir)
    if os.path.exists('%s/val/' % result_dir): shutil.rmtree('%s/val/' % result_dir)
    if not os.path.exists('%s/' % result_dir): os.mkdir('%s/' % result_dir)
    os.mkdir('%s/train/' % result_dir)
    os.mkdir('%s/val/' % result_dir)


    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    for i in range(num_train):
        sample = train_records[i]
        # DecodeImage()
        filename = sample['im_file'].split('/')[-1]
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)

        boxes2 = sample['gt_bbox']    # [M, 4]
        scores2 = sample['gt_score'][:, 0]   # [M, ]
        classes2 = sample['gt_class'][:, 0]  # [M, ]
        if len(boxes2) > 0:
            cfg.draw(im, boxes2, scores2, classes2, _clsid2cname)

        cv2.imwrite('%s/train/%s'%(result_dir, filename), im)
        logger.info('Train: %d/%d' % (i, num_train))

    # 验证集
    val_dataset = COCO(cfg.val_path)
    val_img_ids = val_dataset.getImgIds()
    val_records = data_clean(val_dataset, val_img_ids, _catid2clsid, cfg.val_pre_path)
    num_val = len(val_records)
    for i in range(num_val):
        sample = val_records[i]
        # DecodeImage()
        filename = sample['im_file'].split('/')[-1]
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)

        boxes2 = sample['gt_bbox']    # [M, 4]
        scores2 = sample['gt_score'][:, 0]   # [M, ]
        classes2 = sample['gt_class'][:, 0]  # [M, ]
        if len(boxes2) > 0:
            cfg.draw(im, boxes2, scores2, classes2, _clsid2cname)

        cv2.imwrite('%s/val/%s'%(result_dir, filename), im)
        logger.info('Val: %d/%d' % (i, num_val))


    logger.info('Done.')

