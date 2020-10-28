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

            # 自实现的DCNv2，非常慢=_=!
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



class DCNv2(torch.nn.Module):
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



