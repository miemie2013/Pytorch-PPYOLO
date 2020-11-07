#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : PaddleDetection
#   Created date:
#   Description :
#
# ================================================================
import torch
import numpy as np
import time
import threading


class ExponentialMovingAverage():
    def __init__(self, model, decay, thres_steps=True):
        self._model = model
        self._decay = decay
        self._thres_steps = thres_steps
        self._shadow = {}
        self._backup = {}

    def register(self):
        self._update_step = 0
        for name, param in self._model.named_parameters():
            if param.requires_grad is True:   # 只记录可训练参数。bn层的均值、方差不会被记录，它们有自己的滑动平均。
                self._shadow[name] = param.cpu().detach().numpy().copy()

    def update(self):
        start = time.time()
        for name, param in self._model.named_parameters():
            if param.requires_grad is True:
                assert name in self._shadow
                new_val = np.array(param.cpu().detach().numpy().copy())
                old_val = np.array(self._shadow[name])
                decay = min(self._decay, (1 + self._update_step) / (10 + self._update_step)) if self._thres_steps else self._decay
                new_average = decay * old_val + (1 - decay) * new_val
                self._shadow[name] = new_average
        self._update_step += 1
        cost = time.time() - start
        # print('cost time: {0:.6f}s'.format(cost))
        return decay

    def apply(self):
        for name, param in self._model.named_parameters():
            if param.requires_grad is True:
                assert name in self._shadow
                self._backup[name] = np.array(param.cpu().detach().numpy().copy())
                param.data = torch.Tensor(np.array(self._shadow[name])).cuda()

    def restore(self):
        for name, param in self._model.named_parameters():
            if param.requires_grad is True:
                assert name in self._backup
                param.data = torch.Tensor(self._backup[name]).cuda()
        self._backup = {}
