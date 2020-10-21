#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import time
import threading
import argparse

from config import *
from model.decode_np import Decode
from model.ppyolo import *
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PPYOLO Infer Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=0,
                    choices=[0, 1],
                    help='0 -- ppyolo_2x.py;  1 -- ppyolo_1x.py;  ')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu


def read_test_data(path_dir,
                   _decode,
                   test_dic):
    for k, filename in enumerate(path_dir):
        key_list = list(test_dic.keys())
        key_len = len(key_list)
        while key_len >= 3:
            time.sleep(0.01)
            key_list = list(test_dic.keys())
            key_len = len(key_list)

        image = cv2.imread('images/test/' + filename)
        pimage, im_size = _decode.process_image(np.copy(image))
        dic = {}
        dic['image'] = image
        dic['pimage'] = pimage
        dic['im_size'] = im_size
        test_dic['%.8d' % k] = dic

def save_img(filename, image):
    cv2.imwrite('images/res/' + filename, image)

if __name__ == '__main__':
    cfg = None
    if config_file == 0:
        cfg = PPYOLO_2x_Config()
    elif config_file == 1:
        cfg = PPYOLO_2x_Config()


    # 读取的模型
    model_path = cfg.test_cfg['model_path']

    # 是否给图片画框。
    draw_image = cfg.test_cfg['draw_image']
    draw_thresh = cfg.test_cfg['draw_thresh']

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

    _decode = Decode(ppyolo, all_classes, use_gpu, cfg, for_test=True)

    if not os.path.exists('images/res/'): os.mkdir('images/res/')
    path_dir = os.listdir('images/test')

    # 读数据的线程
    test_dic = {}
    thr = threading.Thread(target=read_test_data,
                           args=(path_dir,
                                 _decode,
                                 test_dic))
    thr.start()

    key_list = list(test_dic.keys())
    key_len = len(key_list)
    while key_len == 0:
        time.sleep(0.01)
        key_list = list(test_dic.keys())
        key_len = len(key_list)
    dic = test_dic['%.8d' % 0]
    image = dic['image']
    pimage = dic['pimage']
    im_size = dic['im_size']


    # warm up
    if use_gpu:
        for k in range(10):
            image, boxes, scores, classes = _decode.detect_image(image, pimage, im_size, draw_image=False)


    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()
    num_imgs = len(path_dir)
    start = time.time()
    for k, filename in enumerate(path_dir):
        key_list = list(test_dic.keys())
        key_len = len(key_list)
        while key_len == 0:
            time.sleep(0.01)
            key_list = list(test_dic.keys())
            key_len = len(key_list)
        dic = test_dic.pop('%.8d' % k)
        image = dic['image']
        pimage = dic['pimage']
        im_size = dic['im_size']

        image, boxes, scores, classes = _decode.detect_image(image, pimage, im_size, draw_image, draw_thresh)

        # 估计剩余时间
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (num_imgs - k) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        logger.info('Infer iter {}, num_imgs={}, eta={}.'.format(k, num_imgs, eta))
        if draw_image:
            t2 = threading.Thread(target=save_img, args=(filename, image))
            t2.start()
            logger.info("Detection bbox results save in images/res/{}".format(filename))
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.'%((cost / num_imgs), (num_imgs / cost)))


