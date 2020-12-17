#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import argparse
import textwrap
from config import *


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Script', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu. True or False')
        # parser.add_argument('-c', '--config', type=int, default=0,
        #                     choices=[0, 1, 2, 3, 4, 5],
        #                     help=textwrap.dedent('''\
        #                     select one of these config files:
        #                     0 -- ppyolo_2x.py
        #                     1 -- yolov4_2x.py
        #                     2 -- ppyolo_r18vd.py
        #                     3 -- ppyolo_mobilenet_v3_large.py
        #                     4 -- ppyolo_mobilenet_v3_small.py
        #                     5 -- ppyolo_mdf_2x.py'''))
        parser.add_argument('-c', '--config', type=int, default=0,
                            choices=[0, 1, 2],
                            help=textwrap.dedent('''\
                            select one of these config files:
                            0 -- ppyolo_2x.py
                            1 -- ppyolo_2x.py
                            2 -- ppyolo_r18vd.py'''))
        self.args = parser.parse_args()
        self.config_file = self.args.config
        self.use_gpu = self.args.use_gpu

    def get_use_gpu(self):
        return self.use_gpu

    def get_cfg(self):
        config_file = self.config_file
        cfg = None
        if config_file == 0:
            cfg = PPYOLO_2x_Config()
        elif config_file == 1:
            cfg = PPYOLO_2x_Config()
        elif config_file == 2:
            cfg = PPYOLO_r18vd_Config()
        # elif config_file == 3:
        #     cfg = PPYOLO_mobilenet_v3_large_Config()
        # elif config_file == 4:
        #     cfg = PPYOLO_mobilenet_v3_large_Config()
        # elif config_file == 5:
        #     cfg = PPYOLO_mdf_2x_Config()
        return cfg


