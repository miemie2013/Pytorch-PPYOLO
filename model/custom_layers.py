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
import torch.nn.functional as F
import numpy as np
try:
    from dcn_v2 import DCN
except ImportError:
    def DCN(*args, **kwdargs):
        raise Exception(
            'DCN could not be imported. If you want to use PPYOLO models, compile DCN. Check the README for instructions.')


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


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


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
                 use_dcn=False):
        super(Conv2dUnit, self).__init__()
        self.groups = groups
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn

        # conv
        if use_dcn:
            self.conv = DCN(input_dim, filters, kernel_size=3, stride=stride, padding=1, dilation=1, deformable_groups=1)
            self.conv.bias.data.zero_()
            self.conv.conv_offset_mask.weight.data.zero_()
            self.conv.conv_offset_mask.bias.data.zero_()

            # 自实现的DCNv2，效果慢一些!
            # self.conv = DCNv2(input_dim, filters, filter_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, bias_attr=False)
        else:
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


    def freeze(self):
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
    咩酱自实现的DCNv2，咩酱的得意之作，Pytorch的纯python接口实现，效率慢一点。
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

        # 最终位置
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = torch.clamp(pos_y, 0.0, H + padding * 2 - 1.0)
        pos_x = torch.clamp(pos_x, 0.0, W + padding * 2 - 1.0)

        y1 = torch.floor(pos_y)
        x1 = torch.floor(pos_x)
        y2 = y1 + 1.0
        x2 = x1 + 1.0

        y1x1 = torch.cat([y1, x1], -1)  # [N, out_H, out_W, kH*kW, 2]
        y1x2 = torch.cat([y1, x2], -1)  # [N, out_H, out_W, kH*kW, 2]
        y2x1 = torch.cat([y2, x1], -1)  # [N, out_H, out_W, kH*kW, 2]
        y2x2 = torch.cat([y2, x2], -1)  # [N, out_H, out_W, kH*kW, 2]
        ytxt = torch.cat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

        pad_x = pad_x.permute(0, 2, 3, 1)  # [N, pad_x_H, pad_x_W, C]

        new_x = torch.zeros((N, out_H, out_W, in_C, kH, kW), dtype=torch.float32, device=x.device)

        mask = torch.reshape(mask, (N, out_H, out_W, kH, kW))  # [N, out_H, out_W, kH, kW]

        for bid in range(N):
            pad_x2 = pad_x[bid]  # [pad_x_H, pad_x_W, in_C]
            mask2 = mask[bid]  # [out_H, out_W, kH, kW]

            _y1x1 = y1x1[bid]  # [out_H, out_W, kH*kW, 2]
            _y1x2 = y1x2[bid]  # [out_H, out_W, kH*kW, 2]
            _y2x1 = y2x1[bid]  # [out_H, out_W, kH*kW, 2]
            _y2x2 = y2x2[bid]  # [out_H, out_W, kH*kW, 2]
            _ytxt = ytxt[bid]  # [out_H, out_W, kH*kW, 2]

            _y1x1 = torch.reshape(_y1x1, (out_H * out_W * kH * kW, 2)).long()  # [out_H*out_W*kH*kW, 2]
            v1 = self.gather_nd(pad_x2, _y1x1)  # [out_H*out_W*kH*kW, in_C]
            _y1x2 = torch.reshape(_y1x2, (out_H * out_W * kH * kW, 2)).long()  # [out_H*out_W*kH*kW, 2]
            v2 = self.gather_nd(pad_x2, _y1x2)  # [out_H*out_W*kH*kW, in_C]
            _y2x1 = torch.reshape(_y2x1, (out_H * out_W * kH * kW, 2)).long()  # [out_H*out_W*kH*kW, 2]
            v3 = self.gather_nd(pad_x2, _y2x1)  # [out_H*out_W*kH*kW, in_C]
            _y2x2 = torch.reshape(_y2x2, (out_H * out_W * kH * kW, 2)).long()  # [out_H*out_W*kH*kW, 2]
            v4 = self.gather_nd(pad_x2, _y2x2)  # [out_H*out_W*kH*kW, in_C]

            _ytxt = torch.reshape(_ytxt, (out_H * out_W * kH * kW, 2))  # [out_H*out_W*kH*kW, 2]

            # 必须类型转换，不然。。。出现数值不正确的问题。
            _y1x1 = _y1x1.float()
            _y1x2 = _y1x2.float()
            _y2x1 = _y2x1.float()
            _y2x2 = _y2x2.float()

            lh = _ytxt[:, :1] - _y1x1[:, :1]  # [out_H*out_W*kH*kW, 1]
            lw = _ytxt[:, 1:] - _y1x1[:, 1:]
            hh = 1 - lh
            hw = 1 - lw
            w1 = hh * hw
            w2 = hh * lw
            w3 = lh * hw
            w4 = lh * lw
            value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # [out_H*out_W*kH*kW, in_C]
            mask2 = torch.reshape(mask2, (out_H * out_W * kH * kW, 1))
            value = value * mask2
            value = torch.reshape(value, (out_H, out_W, kH, kW, in_C))
            value = value.permute(0, 1, 4, 2, 3)
            new_x[bid, :, :, :, :, :] = value

        new_x = torch.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))

        new_x = new_x.permute(0, 3, 1, 2)  # [N, in_C*kH*kW, out_H, out_W]

        exp_new_x = new_x.unsqueeze(1)  # 增加1维，[N,      1, in_C*kH*kW, out_H, out_W]
        reshape_w = torch.reshape(dcn_weight, (1, out_C, in_C * kH * kW, 1, 1))  # [1, out_C,  in_C*kH*kW,     1,     1]
        out = exp_new_x * reshape_w  # 逐元素相乘，[N, out_C,  in_C*kH*kW, out_H, out_W]
        out = out.sum((2,))  # 第2维求和，[N, out_C, out_H, out_W]

        return out



