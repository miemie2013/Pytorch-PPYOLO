#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo。测试ema实现是否成功。
#
# ================================================================
import datetime
import json
from collections import deque
import paddle.fluid as fluid
import paddle.fluid.layers as P
import sys
import time
import shutil
import math
import copy
import random
import threading
import numpy as np
import os
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.optimizer import ExponentialMovingAverage


import torch
from model.custom_layers import Conv2dUnit

from collections import OrderedDict



class MyNet(torch.nn.Module):
    def __init__(self, ema_decay):
        super(MyNet, self).__init__()
        self.ema_decay = ema_decay

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(8, momentum=0.1)
        self.act1 = torch.nn.LeakyReLU(0.1)

        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(8, momentum=0.1)
        self.act2 = torch.nn.LeakyReLU(0.1)


        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.bn1.weight.requires_grad = False
        self.bn1.bias.requires_grad = False

        self.ema_state_dict = OrderedDict()
        self.current_state_dict = OrderedDict()

    def init_ema_state_dict(self):
        temp_dict = self.state_dict()
        for k, v in temp_dict.items():
            v2 = v * 0   # 参数用0初始化（包括可训练参数、bn层的均值、bn层的方差）
            self.ema_state_dict[k] = v2

    def update_ema_state_dict(self, thres_steps):
        decay2 = (1.0 + thres_steps) / (10.0 + thres_steps)
        ema_decay = min(self.ema_decay, decay2)  # 真实的衰减率
        temp_dict = self.state_dict()
        for k, v in temp_dict.items():    # bn层的均值、方差也受该全局ema管理（尽管它们有自己的滑动平均）
            v2 = self.ema_state_dict[k]   # ema中旧的值
            v2 = ema_decay * v2 + (1.0 - ema_decay) * v   # ema中新的值
            v2 = v2 / (1.0 - ema_decay ** (thres_steps + 1))   # 偏置校正
            v2.requires_grad = False      # v2不需要更新。
            self.ema_state_dict[k] = v2   # ema写入新的值

    def apply_ema_state_dict(self):
        # self.current_state_dict = copy.deepcopy(self.state_dict())   # 备份
        # temp_dict = copy.deepcopy(self.ema_state_dict)
        # self.load_state_dict(temp_dict)
        torch.save(self.state_dict(), 'current.pt')
        torch.save(self.ema_state_dict, 'ema.pt')
        self.load_state_dict(self.ema_state_dict)

    def restore_current_state_dict(self):
        # temp_dict = copy.deepcopy(self.current_state_dict)
        # self.load_state_dict(temp_dict)
        self.ema_state_dict = torch.load('ema.pt')
        current = torch.load('current.pt')
        self.load_state_dict(current)

    def __call__(self, input_tensor):
        x0 = self.conv1(input_tensor)
        x1 = self.bn1(x0)
        x = self.act1(x1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x0, x1, x




if __name__ == '__main__':
    use_gpu = False

    lr = 0.1

    ema_decay = 0.9998

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs = P.data(name='input_1', shape=[-1, 3, 28, 28], append_batch_size=False, dtype='float32')
            conv01_out_tensor = fluid.layers.conv2d(input=inputs, num_filters=8, filter_size=1, stride=1, padding=0,
                                                    param_attr=ParamAttr(name="conv01_weights"),
                                                    bias_attr=ParamAttr(name="conv01_bias", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)))
            bn_name = "bn01"
            bn01_out_tensor = fluid.layers.batch_norm(
                input=conv01_out_tensor,
                is_test=False,
                param_attr=ParamAttr(initializer=fluid.initializer.Constant(1.0), name=bn_name + '_scale'),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name=bn_name + '_offset'),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
            act01_out_tensor = fluid.layers.leaky_relu(bn01_out_tensor, alpha=0.1)
            act01_out_tensor.stop_gradient = True

            conv02_out_tensor = fluid.layers.conv2d(input=act01_out_tensor, num_filters=8, filter_size=3, stride=1, padding=1,
                                                    param_attr=ParamAttr(name="conv02_weights"),
                                                    bias_attr=ParamAttr(name="conv02_bias", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)))
            bn_name = "bn02"
            bn02_out_tensor = fluid.layers.batch_norm(
                input=conv02_out_tensor,
                is_test=False,
                param_attr=ParamAttr(initializer=fluid.initializer.Constant(1.0), name=bn_name + '_scale'),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name=bn_name + '_offset'),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
            act02_out_tensor = fluid.layers.leaky_relu(bn02_out_tensor, alpha=0.1)


            # 建立损失函数
            y_true = P.data(name='y_true', shape=[-1, 8, 28, 28], append_batch_size=False, dtype='float32')
            # 先把差值逐项平方，可以用P.pow()这个op，也可以用python里的运算符**。
            mseloss = P.pow(y_true - act02_out_tensor, 2)
            mseloss = P.reduce_mean(mseloss)       # 再求平均，即mse损失函数

            # 优化器
            optimizer = fluid.optimizer.SGD(learning_rate=lr)
            optimizer.minimize(mseloss)


            # ema
            global_steps = _decay_step_counter()
            ema = ExponentialMovingAverage(ema_decay, thres_steps=global_steps)
            ema.update()


    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # 重新建立一次网络，用相同的张量名，不用写损失层
            inputs = P.data(name='input_1', shape=[-1, 3, 28, 28], append_batch_size=False, dtype='float32')
            conv01_out_tensor = fluid.layers.conv2d(input=inputs, num_filters=8, filter_size=1, stride=1, padding=0,
                                                    param_attr=ParamAttr(name="conv01_weights"),
                                                    bias_attr=ParamAttr(name="conv01_bias", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)))
            bn_name = "bn01"
            bn01_out_tensor = fluid.layers.batch_norm(
                input=conv01_out_tensor,
                is_test=False,
                param_attr=ParamAttr(initializer=fluid.initializer.Constant(1.0), name=bn_name + '_scale'),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name=bn_name + '_offset'),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
            act01_out_tensor = fluid.layers.leaky_relu(bn01_out_tensor, alpha=0.1)
            act01_out_tensor.stop_gradient = True

            conv02_out_tensor = fluid.layers.conv2d(input=act01_out_tensor, num_filters=8, filter_size=3, stride=1, padding=1,
                                                    param_attr=ParamAttr(name="conv02_weights"),
                                                    bias_attr=ParamAttr(name="conv02_bias", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)))
            bn_name = "bn02"
            bn02_out_tensor = fluid.layers.batch_norm(
                input=conv02_out_tensor,
                is_test=False,
                param_attr=ParamAttr(initializer=fluid.initializer.Constant(1.0), name=bn_name + '_scale'),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name=bn_name + '_offset'),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
            act02_out_tensor = fluid.layers.leaky_relu(bn02_out_tensor, alpha=0.1)
            eval_fetch_list = [conv01_out_tensor, bn01_out_tensor, act02_out_tensor]
    eval_prog = eval_prog.clone(for_test=True)
    # 参数初始化
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)


    # pytorch搭建的神经网络的权重。初始值是paddle相同层的初始值。为了模拟paddle训练过程。
    # 1.卷积层
    paddle_conv01_weights = np.array(fluid.global_scope().find_var('conv01_weights').get_tensor())
    paddle_conv01_bias = np.array(fluid.global_scope().find_var('conv01_bias').get_tensor())
    # 2.bn层
    paddle_bn01_scale = np.array(fluid.global_scope().find_var('bn01_scale').get_tensor())
    paddle_bn01_offset = np.array(fluid.global_scope().find_var('bn01_offset').get_tensor())
    paddle_bn01_mean = np.array(fluid.global_scope().find_var('bn01_mean').get_tensor())
    paddle_bn01_variance = np.array(fluid.global_scope().find_var('bn01_variance').get_tensor())
    # 3.激活层
    # 4.卷积层
    paddle_conv02_weights = np.array(fluid.global_scope().find_var('conv02_weights').get_tensor())
    paddle_conv02_bias = np.array(fluid.global_scope().find_var('conv02_bias').get_tensor())
    # 5.bn层
    paddle_bn02_scale = np.array(fluid.global_scope().find_var('bn02_scale').get_tensor())
    paddle_bn02_offset = np.array(fluid.global_scope().find_var('bn02_offset').get_tensor())
    paddle_bn02_mean = np.array(fluid.global_scope().find_var('bn02_mean').get_tensor())
    paddle_bn02_variance = np.array(fluid.global_scope().find_var('bn02_variance').get_tensor())
    # 6.激活层
    # 7.损失函数层，没有权重。

    #  pytorch搭建的神经网络
    myNet = MyNet(ema_decay)
    # myNet = myNet.cuda()
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    # loss_fn = loss_fn.cuda()
    optimizer2 = torch.optim.SGD(filter(lambda p: p.requires_grad, myNet.parameters()), lr=lr)   # requires_grad==True 的参数才可以被更新
    # 初始化自己网络的权重
    myNet.conv1.weight.data = torch.Tensor(np.copy(paddle_conv01_weights))
    myNet.conv1.bias.data = torch.Tensor(np.copy(paddle_conv01_bias))
    myNet.bn1.weight.data = torch.Tensor(np.copy(paddle_bn01_scale))
    myNet.bn1.bias.data = torch.Tensor(np.copy(paddle_bn01_offset))
    myNet.bn1.running_mean.data = torch.Tensor(np.copy(paddle_bn01_mean))
    myNet.bn1.running_var.data = torch.Tensor(np.copy(paddle_bn01_variance))
    myNet.conv2.weight.data = torch.Tensor(np.copy(paddle_conv02_weights))
    myNet.conv2.bias.data = torch.Tensor(np.copy(paddle_conv02_bias))
    myNet.bn2.weight.data = torch.Tensor(np.copy(paddle_bn02_scale))
    myNet.bn2.bias.data = torch.Tensor(np.copy(paddle_bn02_offset))
    myNet.bn2.running_mean.data = torch.Tensor(np.copy(paddle_bn02_mean))
    myNet.bn2.running_var.data = torch.Tensor(np.copy(paddle_bn02_variance))

    myNet.init_ema_state_dict()
    print()


    myNet.train()  # 切换到训练模式


    # 只训练8步
    for step in range(8):
        print('------------------ step %d ------------------' % step)
        # ==================== train ====================
        batch_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        y_true_arr = np.random.normal(loc=0, scale=1, size=(2, 8, 28, 28)).astype(np.float32)

        paddle_mseloss_out, paddle_conv01_out, paddle_bn02_out = exe.run(train_prog, feed={"input_1": batch_data, "y_true": y_true_arr, },
                                                                       fetch_list=[mseloss, conv01_out_tensor, bn02_out_tensor])

        print('train_forward:')


        # python代码模拟训练过程，与paddle的输出校验。我们希望和飞桨有相同的输出。
        batch_data = torch.Tensor(batch_data)
        y_true_arr = torch.Tensor(y_true_arr)
        _, _, my_act02_out = myNet(batch_data)
        my_mseloss_out = loss_fn(my_act02_out,  y_true_arr)

        # 更新权重
        optimizer2.zero_grad()  # 清空上一步的残余更新参数值
        my_mseloss_out.backward()  # 误差反向传播, 计算参数更新值
        optimizer2.step()  # 将参数更新值施加到 net 的 parameters 上
        myNet.update_ema_state_dict(step)   # 更新ema_state_dict


        _my_mseloss_out = my_mseloss_out.cpu().data.numpy()

        diff_mseloss_out = np.sum((paddle_mseloss_out - _my_mseloss_out)**2)
        print('diff_mseloss_out=%.6f' % diff_mseloss_out)   # 若是0，则表示成功模拟出PaddlePaddle bn层的输出结果


        # 应用滑动平均参数进行test
        exe.run(ema.apply_program)
        myNet.apply_ema_state_dict()
        print('\nema_apply:')
        # 和飞桨更新后的权重校验。
        paddle_conv01_weights = np.array(fluid.global_scope().find_var('conv01_weights').get_tensor())
        paddle_conv01_bias = np.array(fluid.global_scope().find_var('conv01_bias').get_tensor())
        paddle_conv02_weights = np.array(fluid.global_scope().find_var('conv02_weights').get_tensor())
        paddle_conv02_bias = np.array(fluid.global_scope().find_var('conv02_bias').get_tensor())
        paddle_bn01_scale = np.array(fluid.global_scope().find_var('bn01_scale').get_tensor())
        paddle_bn01_offset = np.array(fluid.global_scope().find_var('bn01_offset').get_tensor())
        paddle_bn01_mean = np.array(fluid.global_scope().find_var('bn01_mean').get_tensor())
        paddle_bn01_variance = np.array(fluid.global_scope().find_var('bn01_variance').get_tensor())
        paddle_bn02_scale = np.array(fluid.global_scope().find_var('bn02_scale').get_tensor())
        paddle_bn02_offset = np.array(fluid.global_scope().find_var('bn02_offset').get_tensor())
        paddle_bn02_mean = np.array(fluid.global_scope().find_var('bn02_mean').get_tensor())
        paddle_bn02_variance = np.array(fluid.global_scope().find_var('bn02_variance').get_tensor())


        diff_conv01_weights = np.sum((paddle_conv01_weights - myNet.conv1.weight.data.numpy())**2)
        print('diff_conv01_weights=%.6f' % diff_conv01_weights)   # 若是0，则表示成功模拟出权重更新
        diff_conv01_bias = np.sum((paddle_conv01_bias - myNet.conv1.bias.data.numpy())**2)
        print('diff_conv01_bias=%.6f' % diff_conv01_bias)   # 若是0，则表示成功模拟出权重更新
        diff_conv02_weights = np.sum((paddle_conv02_weights - myNet.conv2.weight.data.numpy())**2)
        print('diff_conv02_weights=%.6f' % diff_conv02_weights)   # 若是0，则表示成功模拟出权重更新
        diff_conv02_bias = np.sum((paddle_conv02_bias - myNet.conv2.bias.data.numpy())**2)
        print('diff_conv02_bias=%.6f' % diff_conv02_bias)   # 若是0，则表示成功模拟出权重更新


        diff_bn02_scale = np.sum((paddle_bn02_scale - myNet.bn2.weight.data.numpy())**2)
        print('diff_bn02_scale=%.6f' % diff_bn02_scale)   # 若是0，则表示成功模拟出权重更新
        diff_bn02_offset = np.sum((paddle_bn02_offset - myNet.bn2.bias.data.numpy())**2)
        print('diff_bn02_offset=%.6f' % diff_bn02_offset)   # 若是0，则表示成功模拟出权重更新
        diff_bn01_scale = np.sum((paddle_bn01_scale - myNet.bn1.weight.data.numpy())**2)
        print('diff_bn01_scale=%.6f' % diff_bn01_scale)   # 若是0，则表示成功模拟出权重更新
        diff_bn01_offset = np.sum((paddle_bn01_offset - myNet.bn1.bias.data.numpy())**2)
        print('diff_bn01_offset=%.6f' % diff_bn01_offset)   # 若是0，则表示成功模拟出权重更新

        # 均值和方差，在train_forward()阶段就已经被更新
        print('bn mean var:')
        diff_bn02_mean = np.sum((paddle_bn02_mean - myNet.bn2.running_mean.data.numpy())**2)
        print('diff_bn02_mean=%.6f' % diff_bn02_mean)   # 若是0，则表示成功模拟出均值更新
        diff_bn02_variance = np.sum((paddle_bn02_variance - myNet.bn2.running_var.data.numpy())**2)
        print('diff_bn02_variance=%.6f' % diff_bn02_variance)   # 若是0，则表示成功模拟出方差更新
        diff_bn01_mean = np.sum((paddle_bn01_mean - myNet.bn1.running_mean.data.numpy())**2)
        print('diff_bn01_mean=%.6f' % diff_bn01_mean)   # 若是0，则表示成功模拟出均值更新
        diff_bn01_variance = np.sum((paddle_bn01_variance - myNet.bn1.running_var.data.numpy())**2)
        print('diff_bn01_variance=%.6f' % diff_bn01_variance)   # 若是0，则表示成功模拟出方差更新

        # ==================== test ====================
        test_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        aa1, aa2, paddle_test_act02_out = exe.run(compiled_eval_prog, feed={"input_1": test_data, }, fetch_list=eval_fetch_list)
        # 自己网络的test
        print('\ntest_forward:')
        myNet.eval()  # 切换到验证模式
        test_data = torch.Tensor(test_data)
        a1, a2, my_test_act02_out_ = myNet(test_data)
        a1 = a1.cpu().data.numpy()
        a2 = a2.cpu().data.numpy()
        my_test_act02_out = my_test_act02_out_.cpu().data.numpy()
        myNet.train()  # 切换到训练模式
        d1 = np.sum((aa1 - a1)**2)
        print('d1=%.6f' % d1)   # 若是0，则表示成功模拟出推理
        d2 = np.sum((aa2 - a2)**2)
        print('d2=%.6f' % d2)   # 若是0，则表示成功模拟出推理
        diff_test_act02_out = np.sum((paddle_test_act02_out - my_test_act02_out)**2)
        print('diff_test_act02_out=%.6f' % diff_test_act02_out)   # 若是0，则表示成功模拟出推理


        # 恢复之前的参数
        exe.run(ema.restore_program)
        myNet.restore_current_state_dict()



