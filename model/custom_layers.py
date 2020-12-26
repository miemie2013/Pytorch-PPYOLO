#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import torch
import torch as T
import torch.nn.functional as F
import numpy as np
try:
    from dcn_v2 import DCN
except ImportError:
    def DCN(*args, **kwdargs):
        raise Exception(
            'DCN could not be imported. If you want to use PPYOLO models, compile DCN. Check the README for instructions.')


def get_norm(norm_type):
    bn = 0
    gn = 0
    af = 0
    if norm_type == 'bn':
        bn = 1
    elif norm_type == 'sync_bn':
        bn = 1
    elif norm_type == 'gn':
        gn = 1
    elif norm_type == 'affine_channel':
        af = 1
    return bn, gn, af


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class AffineChannel(torch.nn.Module):
    def __init__(self, num_features):
        super(AffineChannel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_features, ))
        self.bias = torch.nn.Parameter(torch.randn(num_features, ))

    def forward(self, x):
        N = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        transpose_x = x.permute(0, 2, 3, 1)
        flatten_x = transpose_x.reshape(N*H*W, C)
        out = flatten_x * self.weight + self.bias
        out = out.reshape(N, H, W, C)
        out = out.permute(0, 3, 1, 2)
        return out


class Conv2dUnit(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 bn=0,
                 gn=0,
                 af=0,
                 groups=32,
                 act=None,
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 lr=1.,
                 bias_lr=None,
                 weight_init=None,
                 bias_init=None,
                 use_dcn=False,
                 name=''):
        super(Conv2dUnit, self).__init__()
        self.groups = groups
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = (filter_size - 1) // 2
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn
        self.name = name
        self.lr = lr

        # conv
        if use_dcn:
            # self.conv = DCN(input_dim, filters, kernel_size=3, stride=stride, padding=1, dilation=1, deformable_groups=1)
            # self.conv.bias.data.zero_()
            # self.conv.conv_offset_mask.weight.data.zero_()
            # self.conv.conv_offset_mask.bias.data.zero_()

            # 咩酱自实现的DCNv2，咩酱的得意之作，Pytorch的纯python接口实现，效率极高。
            self.conv = DCNv2(input_dim, filters, filter_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, bias_attr=False)
        else:
            if bias_attr:
                blr = lr
                if bias_lr:
                    blr = bias_lr
                self.blr = blr
            self.conv = torch.nn.Conv2d(input_dim, filters, kernel_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, bias=bias_attr)

        # norm
        self.bn = None
        self.gn = None
        self.af = None
        if bn:
            self.bn = torch.nn.BatchNorm2d(filters)
        if gn:
            self.gn = torch.nn.GroupNorm(num_groups=groups, num_channels=filters)
        if af:
            self.af = AffineChannel(filters)

        # act
        self.act = None
        if act == 'relu':
            self.act = torch.nn.ReLU(inplace=True)
        elif act == 'leaky':
            self.act = torch.nn.LeakyReLU(0.1)
        elif act == 'mish':
            self.act = Mish()
        elif act is None:
            pass
        else:
            raise NotImplementedError("Activation \'{}\' is not implemented.".format(act))


    def freeze(self):
        if isinstance(self.conv, torch.nn.Conv2d):
            self.conv.weight.requires_grad = False
            if self.conv.bias is not None:
                self.conv.bias.requires_grad = False
        elif isinstance(self.conv, DCNv2):   # 自实现的DCNv2
            self.conv.conv_offset.weight.requires_grad = False
            self.conv.conv_offset.bias.requires_grad = False
            self.conv.dcn_weight.requires_grad = False
            if self.conv.dcn_bias is not None:
                self.conv.dcn_bias.requires_grad = False
        else:   # 官方DCNv2
            self.conv.weight.requires_grad = False
            if self.conv.bias is not None:
                self.conv.bias.requires_grad = False
        if self.bn is not None:
            self.bn.weight.requires_grad = False
            self.bn.bias.requires_grad = False
        if self.gn is not None:
            self.gn.weight.requires_grad = False
            self.gn.bias.requires_grad = False
        if self.af is not None:
            self.af.weight.requires_grad = False
            self.af.bias.requires_grad = False

    def add_param_group(self, param_groups, base_lr, base_wd):
        if isinstance(self.conv, torch.nn.Conv2d):
            if self.conv.weight.requires_grad:
                param_group_conv = {'params': [self.conv.weight]}
                param_group_conv['lr'] = base_lr * self.lr
                param_group_conv['base_lr'] = base_lr * self.lr
                param_group_conv['weight_decay'] = base_wd
                param_groups.append(param_group_conv)
                if self.conv.bias is not None:
                    if self.conv.bias.requires_grad:
                        param_group_conv_bias = {'params': [self.conv.bias]}
                        param_group_conv_bias['lr'] = base_lr * self.blr
                        param_group_conv_bias['base_lr'] = base_lr * self.blr
                        param_group_conv_bias['weight_decay'] = 0.0
                        param_groups.append(param_group_conv_bias)
        elif isinstance(self.conv, DCNv2):   # 自实现的DCNv2
            if self.conv.conv_offset.weight.requires_grad:
                param_group_conv_offset_w = {'params': [self.conv.conv_offset.weight]}
                param_group_conv_offset_w['lr'] = base_lr * self.lr
                param_group_conv_offset_w['base_lr'] = base_lr * self.lr
                param_group_conv_offset_w['weight_decay'] = base_wd
                param_groups.append(param_group_conv_offset_w)
            if self.conv.conv_offset.bias.requires_grad:
                param_group_conv_offset_b = {'params': [self.conv.conv_offset.bias]}
                param_group_conv_offset_b['lr'] = base_lr * self.lr
                param_group_conv_offset_b['base_lr'] = base_lr * self.lr
                param_group_conv_offset_b['weight_decay'] = base_wd
                param_groups.append(param_group_conv_offset_b)
            if self.conv.dcn_weight.requires_grad:
                param_group_dcn_weight = {'params': [self.conv.dcn_weight]}
                param_group_dcn_weight['lr'] = base_lr * self.lr
                param_group_dcn_weight['base_lr'] = base_lr * self.lr
                param_group_dcn_weight['weight_decay'] = base_wd
                param_groups.append(param_group_dcn_weight)
        else:   # 官方DCNv2
            pass
        if self.bn is not None:
            if self.bn.weight.requires_grad:
                param_group_norm_weight = {'params': [self.bn.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_groups.append(param_group_norm_weight)
            if self.bn.bias.requires_grad:
                param_group_norm_bias = {'params': [self.bn.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_groups.append(param_group_norm_bias)
        if self.gn is not None:
            if self.gn.weight.requires_grad:
                param_group_norm_weight = {'params': [self.gn.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_groups.append(param_group_norm_weight)
            if self.gn.bias.requires_grad:
                param_group_norm_bias = {'params': [self.gn.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_groups.append(param_group_norm_bias)
        if self.af is not None:
            if self.af.weight.requires_grad:
                param_group_norm_weight = {'params': [self.af.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_groups.append(param_group_norm_weight)
            if self.af.bias.requires_grad:
                param_group_norm_bias = {'params': [self.af.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_groups.append(param_group_norm_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.gn:
            x = self.gn(x)
        if self.af:
            x = self.af(x)
        if self.act:
            x = self.act(x)
        return x


class CoordConv(torch.nn.Module):
    def __init__(self, coord_conv=True):
        super(CoordConv, self).__init__()
        self.coord_conv = coord_conv

    def __call__(self, input):
        if not self.coord_conv:
            return input
        b = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        x_range = T.arange(0, w, dtype=T.float32, device=input.device) / (w - 1) * 2.0 - 1
        y_range = T.arange(0, h, dtype=T.float32, device=input.device) / (h - 1) * 2.0 - 1
        x_range = x_range[np.newaxis, np.newaxis, np.newaxis, :].repeat((b, 1, h, 1))
        y_range = y_range[np.newaxis, np.newaxis, :, np.newaxis].repeat((b, 1, 1, w))
        offset = T.cat([input, x_range, y_range], dim=1)
        return offset


class SPP(torch.nn.Module):
    def __init__(self, seq='asc'):
        super(SPP, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq

    def __call__(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, 1, 2)
        x_3 = F.max_pool2d(x, 9, 1, 4)
        x_4 = F.max_pool2d(x, 13, 1, 6)
        if self.seq == 'desc':
            out = torch.cat([x_4, x_3, x_2, x_1], dim=1)
        else:
            out = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        return out


class DropBlock(torch.nn.Module):
    def __init__(self,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=False):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.is_test = is_test

    def __call__(self, input):
        if self.is_test:
            return input

        def CalculateGamma(input, block_size, keep_prob):
            h = input.shape[2]  # int
            h = np.array([h])
            h = torch.tensor(h, dtype=torch.float32, device=input.device)
            feat_shape_t = h.reshape((1, 1, 1, 1))  # shape: [1, 1, 1, 1]
            feat_area = torch.pow(feat_shape_t, 2)  # shape: [1, 1, 1, 1]

            block_shape_t = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=input.device) + block_size
            block_area = torch.pow(block_shape_t, 2)

            useful_shape_t = feat_shape_t - block_shape_t + 1
            useful_area = torch.pow(useful_shape_t, 2)

            upper_t = feat_area * (1 - keep_prob)
            bottom_t = block_area * useful_area
            output = upper_t / bottom_t
            return output

        gamma = CalculateGamma(input, block_size=self.block_size, keep_prob=self.keep_prob)
        input_shape = input.shape
        p = gamma.repeat(input_shape)

        input_shape_tmp = input.shape
        random_matrix = torch.rand(input_shape_tmp, device=input.device)
        one_zero_m = (random_matrix < p).float()

        mask_flag = torch.nn.functional.max_pool2d(one_zero_m, (self.block_size, self.block_size), stride=1, padding=1)
        mask = 1.0 - mask_flag

        elem_numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        elem_numel_m = float(elem_numel)

        elem_sum = mask.sum()

        output = input * mask * elem_numel_m / elem_sum
        return output



class DCNv2_Slow(torch.nn.Module):
    '''
    自实现的DCNv2，非常慢=_=!
    '''
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 distribution='normal',
                 gain=1):
        super(DCNv2_Slow, self).__init__()
        assert distribution in ['uniform', 'normal']
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.conv_offset = torch.nn.Conv2d(input_dim, filter_size * filter_size * 3, kernel_size=filter_size,
                                           stride=stride,
                                           padding=padding, bias=True)
        # 初始化代码摘抄自SOLOv2  mmcv/cnn/weight_init.py  里的代码
        torch.nn.init.constant_(self.conv_offset.weight, 0.0)
        torch.nn.init.constant_(self.conv_offset.bias, 0.0)

        self.sigmoid = torch.nn.Sigmoid()

        self.dcn_weight = torch.nn.Parameter(torch.randn(filters, input_dim, filter_size, filter_size))
        self.dcn_bias = None
        if bias_attr:
            self.dcn_bias = torch.nn.Parameter(torch.randn(filters, ))
            torch.nn.init.constant_(self.dcn_bias, 0.0)
        if distribution == 'uniform':
            torch.nn.init.xavier_uniform_(self.dcn_weight, gain=gain)
        else:
            torch.nn.init.xavier_normal_(self.dcn_weight, gain=gain)

    def get_pix(self, input, y, x):
        _x = x.int()
        _y = y.int()
        return input[:, _y:_y + 1, _x:_x + 1]

    def bilinear(self, input, h, w):
        # 双线性插值
        C, height, width = input.shape

        y0 = torch.floor(h)
        x0 = torch.floor(w)
        y1 = y0 + 1
        x1 = x0 + 1

        lh = h - y0
        lw = w - x0
        hh = 1 - lh
        hw = 1 - lw

        v1 = torch.zeros((C, 1, 1), device=input.device)
        if (y0 >= 0 and x0 >= 0 and y0 <= height - 1 and x0 <= width - 1):
            v1 = self.get_pix(input, y0, x0)

        v2 = torch.zeros((C, 1, 1), device=input.device)
        if (y0 >= 0 and x1 >= 0 and y0 <= height - 1 and x1 <= width - 1):
            v2 = self.get_pix(input, y0, x1)

        v3 = torch.zeros((C, 1, 1), device=input.device)
        if (y1 >= 0 and x0 >= 0 and y1 <= height - 1 and x0 <= width - 1):
            v3 = self.get_pix(input, y1, x0)

        v4 = torch.zeros((C, 1, 1), device=input.device)
        if (y1 >= 0 and x1 >= 0 and y1 <= height - 1 and x1 <= width - 1):
            v4 = self.get_pix(input, y1, x1)

        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
        return out

    def forward(self, x):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias

        offset_mask = self.conv_offset(x)
        offset = offset_mask[:, :filter_size ** 2 * 2, :, :]
        mask = offset_mask[:, filter_size ** 2 * 2:, :, :]
        mask = self.sigmoid(mask)

        N, C, H, W = x.shape
        out_C, in_C, kH, kW = dcn_weight.shape

        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride
        out = torch.zeros((N, out_C, out_H, out_W), device=x.device)

        for bid in range(N):
            input = x[bid]
            sample_offset = offset[bid]
            sample_mask = mask[bid]
            # 2.卷积核滑动，只会在H和W两个方向上滑动
            for i in range(out_H):  # i是纵坐标
                for j in range(out_W):  # j是横坐标
                    ori_x = j * stride - padding  # 卷积核在输入x中的横坐标，等差数列，公差是stride
                    ori_y = i * stride - padding  # 卷积核在输入x中的纵坐标，等差数列，公差是stride
                    point_offset = sample_offset[:, i, j]
                    point_mask = sample_mask[:, i, j]

                    part_x = []
                    for i2 in range(filter_size):
                        for j2 in range(filter_size):
                            # 注意，是先y后x，然后一直交替
                            _offset_y = point_offset[2 * (i2 * filter_size + j2)]
                            _offset_x = point_offset[2 * (i2 * filter_size + j2) + 1]
                            mask_ = point_mask[i2 * filter_size + j2]
                            h_im = ori_y + i2 + _offset_y
                            w_im = ori_x + j2 + _offset_x
                            value = self.bilinear(input, h_im, w_im)
                            value = value * mask_
                            part_x.append(value)
                    part_x = torch.cat(part_x, 1)
                    part_x = torch.reshape(part_x, (in_C, filter_size, filter_size))  # [in_C, kH, kW]

                    exp_part_x = part_x.unsqueeze(0)  # 增加1维，[1,     in_C, kH, kW]。
                    mul = exp_part_x * dcn_weight  # 卷积核和exp_part_x相乘，[out_C, in_C, kH, kW]。
                    mul = mul.sum((1, 2, 3))  # 后3维求和，[out_C, ]。
                    if dcn_bias is not None:
                        mul += dcn_bias

                    # 将得到的新像素写进out的对应位置
                    out[bid, :, i, j] = mul

        return out



class DCNv2(torch.nn.Module):
    '''
    咩酱自实现的DCNv2，咩酱的得意之作，Pytorch的纯python接口实现，效率极高。
    '''
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 distribution='normal',
                 gain=1):
        super(DCNv2, self).__init__()
        assert distribution in ['uniform', 'normal']
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.conv_offset = torch.nn.Conv2d(input_dim, filter_size * filter_size * 3, kernel_size=filter_size,
                                           stride=stride,
                                           padding=padding, bias=True)
        # 初始化代码摘抄自SOLOv2  mmcv/cnn/weight_init.py  里的代码
        torch.nn.init.constant_(self.conv_offset.weight, 0.0)
        torch.nn.init.constant_(self.conv_offset.bias, 0.0)

        self.sigmoid = torch.nn.Sigmoid()

        self.dcn_weight = torch.nn.Parameter(torch.randn(filters, input_dim, filter_size, filter_size))
        self.dcn_bias = None
        if bias_attr:
            self.dcn_bias = torch.nn.Parameter(torch.randn(filters, ))
            torch.nn.init.constant_(self.dcn_bias, 0.0)
        if distribution == 'uniform':
            torch.nn.init.xavier_uniform_(self.dcn_weight, gain=gain)
        else:
            torch.nn.init.xavier_normal_(self.dcn_weight, gain=gain)

    def gather_nd(self, input, index):
        # 不被合并的后面的维
        keep_dims = []
        # 被合并的维
        first_dims = []
        dim_idx = []
        dims = index.shape[1]
        for i, number in enumerate(input.shape):
            if i < dims:
                dim_ = index[:, i]
                dim_idx.append(dim_)
                first_dims.append(number)
            else:
                keep_dims.append(number)

        # 为了不影响输入index的最后一维，避免函数副作用
        target_dix = torch.zeros((index.shape[0],), dtype=torch.long, device=input.device) + dim_idx[-1]
        new_shape = (-1,) + tuple(keep_dims)
        input2 = torch.reshape(input, new_shape)
        mul2 = 1
        for i in range(dims - 1, 0, -1):
            mul2 *= first_dims[i]
            target_dix += mul2 * dim_idx[i - 1]
        o = input2[target_dix]
        return o

    def forward(self, x):
        filter_size = self.filter_size
        stride = self.stride
        padding = self.padding
        dcn_weight = self.dcn_weight
        dcn_bias = self.dcn_bias

        offset_mask = self.conv_offset(x)
        offset = offset_mask[:, :filter_size ** 2 * 2, :, :]
        mask = offset_mask[:, filter_size ** 2 * 2:, :, :]
        mask = self.sigmoid(mask)


        # ===================================
        N, in_C, H, W = x.shape
        out_C, in_C, kH, kW = dcn_weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride

        # 1.先对图片x填充得到填充后的图片pad_x
        pad_x_H = H + padding * 2 + 1
        pad_x_W = W + padding * 2 + 1
        pad_x = torch.zeros((N, in_C, pad_x_H, pad_x_W), dtype=torch.float32, device=x.device)
        pad_x[:, :, padding:padding + H, padding:padding + W] = x

        # 卷积核中心点在pad_x中的位置
        rows = torch.arange(0, out_W, dtype=torch.float32, device=dcn_weight.device) * stride + padding
        cols = torch.arange(0, out_H, dtype=torch.float32, device=dcn_weight.device) * stride + padding
        rows = rows[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis].repeat((1, out_H, 1, 1, 1))
        cols = cols[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis].repeat((1, 1, out_W, 1, 1))
        start_pos_yx = torch.cat([cols, rows], dim=-1)  # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = start_pos_yx.repeat((N, 1, 1, kH * kW, 1))  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置

        # 卷积核内部的偏移
        half_W = (kW - 1) // 2
        half_H = (kW - 1) // 2
        rows2 = torch.arange(0, kW, dtype=torch.float32, device=dcn_weight.device) - half_W
        cols2 = torch.arange(0, kH, dtype=torch.float32, device=dcn_weight.device) - half_H
        rows2 = rows2[np.newaxis, :, np.newaxis].repeat((kH, 1, 1))
        cols2 = cols2[:, np.newaxis, np.newaxis].repeat((1, kW, 1))
        filter_inner_offset_yx = torch.cat([cols2, rows2], dim=-1)  # [kH, kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = torch.reshape(filter_inner_offset_yx,
                                               (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = filter_inner_offset_yx.repeat(
            (N, out_H, out_W, 1, 1))  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移

        mask = mask.permute(0, 2, 3, 1)  # [N, out_H, out_W, kH*kW*1]
        offset = offset.permute(0, 2, 3, 1)  # [N, out_H, out_W, kH*kW*2]
        offset_yx = torch.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # 最终位置。其实也不是最终位置，为了更快速实现DCNv2，还要给y坐标（代表行号）加上图片的偏移来一次性抽取，避免for循环遍历每一张图片。
        start_pos_y.requires_grad = False
        start_pos_x.requires_grad = False
        filter_inner_offset_y.requires_grad = False
        filter_inner_offset_x.requires_grad = False
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = torch.clamp(pos_y, 0.0, H + padding * 2 - 1.0)
        pos_x = torch.clamp(pos_x, 0.0, W + padding * 2 - 1.0)
        ytxt = torch.cat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

        pad_x = pad_x.permute(0, 2, 3, 1)  # [N, pad_x_H, pad_x_W, C]
        pad_x = torch.reshape(pad_x, (N*pad_x_H, pad_x_W, in_C))  # [N*pad_x_H, pad_x_W, C]

        ytxt = torch.reshape(ytxt, (N * out_H * out_W * kH * kW, 2))  # [N*out_H*out_W*kH*kW, 2]
        _yt = ytxt[:, :1]  # [N*out_H*out_W*kH*kW, 1]
        _xt = ytxt[:, 1:]  # [N*out_H*out_W*kH*kW, 1]

        # 为了避免使用for循环遍历每一张图片，还要给y坐标（代表行号）加上图片的偏移来一次性抽取出更兴趣的像素。
        row_offset = torch.arange(0, N, dtype=torch.float32, device=dcn_weight.device) * pad_x_H  # [N, ]
        row_offset = row_offset[:, np.newaxis, np.newaxis].repeat((1, out_H * out_W * kH * kW, 1))  # [N, out_H*out_W*kH*kW, 1]
        row_offset = torch.reshape(row_offset, (N * out_H * out_W * kH * kW, 1))  # [N*out_H*out_W*kH*kW, 1]
        row_offset.requires_grad = False
        _yt += row_offset

        _y1 = torch.floor(_yt)
        _x1 = torch.floor(_xt)
        _y2 = _y1 + 1.0
        _x2 = _x1 + 1.0
        _y1x1 = torch.cat([_y1, _x1], -1)
        _y1x2 = torch.cat([_y1, _x2], -1)
        _y2x1 = torch.cat([_y2, _x1], -1)
        _y2x2 = torch.cat([_y2, _x2], -1)

        _y1x1_int = _y1x1.long()   # [N*out_H*out_W*kH*kW, 2]
        v1 = self.gather_nd(pad_x, _y1x1_int)   # [N*out_H*out_W*kH*kW, in_C]
        _y1x2_int = _y1x2.long()   # [N*out_H*out_W*kH*kW, 2]
        v2 = self.gather_nd(pad_x, _y1x2_int)   # [N*out_H*out_W*kH*kW, in_C]
        _y2x1_int = _y2x1.long()   # [N*out_H*out_W*kH*kW, 2]
        v3 = self.gather_nd(pad_x, _y2x1_int)   # [N*out_H*out_W*kH*kW, in_C]
        _y2x2_int = _y2x2.long()   # [N*out_H*out_W*kH*kW, 2]
        v4 = self.gather_nd(pad_x, _y2x2_int)   # [N*out_H*out_W*kH*kW, in_C]

        lh = _yt - _y1  # [N*out_H*out_W*kH*kW, 1]
        lw = _xt - _x1
        hh = 1 - lh
        hw = 1 - lw
        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4   # [N*out_H*out_W*kH*kW, in_C]
        mask = torch.reshape(mask, (N * out_H * out_W * kH * kW, 1))
        value = value * mask
        value = torch.reshape(value, (N, out_H, out_W, kH, kW, in_C))
        new_x = value.permute(0, 1, 2, 5, 3, 4)   # [N, out_H, out_W, in_C, kH, kW]

        # 旧的方案，使用逐元素相乘，慢！
        # new_x = torch.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))  # [N, out_H, out_W, in_C * kH * kW]
        # new_x = new_x.permute(0, 3, 1, 2)  # [N, in_C*kH*kW, out_H, out_W]
        # exp_new_x = new_x.unsqueeze(1)  # 增加1维，[N,      1, in_C*kH*kW, out_H, out_W]
        # reshape_w = torch.reshape(dcn_weight, (1, out_C, in_C * kH * kW, 1, 1))  # [1, out_C,  in_C*kH*kW,     1,     1]
        # out = exp_new_x * reshape_w  # 逐元素相乘，[N, out_C,  in_C*kH*kW, out_H, out_W]
        # out = out.sum((2,))  # 第2维求和，[N, out_C, out_H, out_W]

        # 新的方案，用等价的1x1卷积代替逐元素相乘，快！
        new_x = torch.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))  # [N, out_H, out_W, in_C * kH * kW]
        new_x = new_x.permute(0, 3, 1, 2)  # [N, in_C*kH*kW, out_H, out_W]
        rw = torch.reshape(dcn_weight, (out_C, in_C*kH*kW, 1, 1))  # [out_C, in_C, kH, kW] -> [out_C, in_C*kH*kW, 1, 1]  变成1x1卷积核
        out = F.conv2d(new_x, rw, stride=1)  # [N, out_C, out_H, out_W]
        return out



