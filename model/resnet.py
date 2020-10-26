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
from model.custom_layers import Conv2dUnit



class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, filters, bn, gn, af, use_dcn=False, stride=2, downsample_in3x3=True, is_first=False):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        if downsample_in3x3 == True:
            stride1, stride2 = 1, stride
        else:
            stride1, stride2 = stride, 1
        self.is_first = is_first

        self.conv1 = Conv2dUnit(in_c,     filters1, 1, stride=stride1, bn=bn, gn=gn, af=af, act='relu')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride2, bn=bn, gn=gn, af=af, act='relu', use_dcn=use_dcn)
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, bn=bn, gn=gn, af=af, act=None)

        if not self.is_first:
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=1, bn=bn, gn=gn, af=af, act=None)
        else:
            self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=stride, bn=bn, gn=gn, af=af, act=None)
        self.act = torch.nn.ReLU(inplace=True)

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.conv3.freeze()
        self.conv4.freeze()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        if not self.is_first:
            input_tensor = self.avg_pool(input_tensor)
        shortcut = self.conv4(input_tensor)
        x = x + shortcut
        x = self.act(x)
        return x


class IdentityBlock(torch.nn.Module):
    def __init__(self, in_c, filters, bn, gn, af, use_dcn=False):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(in_c,     filters1, 1, stride=1, bn=bn, gn=gn, af=af, act='relu')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=1, bn=bn, gn=gn, af=af, act='relu', use_dcn=use_dcn)
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, bn=bn, gn=gn, af=af, act=None)

        self.act = torch.nn.ReLU(inplace=True)

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.conv3.freeze()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + input_tensor
        x = self.act(x)
        return x

class Resnet50Vd(torch.nn.Module):
    def __init__(self, norm_type='bn', feature_maps=[3, 4, 5], dcn_v2_stages=[5], downsample_in3x3=True, freeze_at=0):
        super(Resnet50Vd, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        self.freeze_at = freeze_at
        bn = 0
        gn = 0
        af = 0
        if norm_type == 'bn':
            bn = 1
        elif norm_type == 'gn':
            gn = 1
        elif norm_type == 'affine_channel':
            af = 1
        self.stage1_conv1_1 = Conv2dUnit(3,  32, 3, stride=2, bn=bn, gn=gn, af=af, act='relu')
        self.stage1_conv1_2 = Conv2dUnit(32, 32, 3, stride=1, bn=bn, gn=gn, af=af, act='relu')
        self.stage1_conv1_3 = Conv2dUnit(32, 64, 3, stride=1, bn=bn, gn=gn, af=af, act='relu')
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage2
        self.stage2_0 = ConvBlock(64, [64, 64, 256], bn, gn, af, stride=1, downsample_in3x3=downsample_in3x3, is_first=True)
        self.stage2_1 = IdentityBlock(256, [64, 64, 256], bn, gn, af)
        self.stage2_2 = IdentityBlock(256, [64, 64, 256], bn, gn, af)

        # stage3
        use_dcn = 3 in dcn_v2_stages
        self.stage3_0 = ConvBlock(256, [128, 128, 512], bn, gn, af, use_dcn=use_dcn, downsample_in3x3=downsample_in3x3)
        self.stage3_1 = IdentityBlock(512, [128, 128, 512], bn, gn, af, use_dcn=use_dcn)
        self.stage3_2 = IdentityBlock(512, [128, 128, 512], bn, gn, af, use_dcn=use_dcn)
        self.stage3_3 = IdentityBlock(512, [128, 128, 512], bn, gn, af, use_dcn=use_dcn)

        # stage4
        use_dcn = 4 in dcn_v2_stages
        self.stage4_0 = ConvBlock(512, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn, downsample_in3x3=downsample_in3x3)
        self.stage4_1 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn)
        self.stage4_2 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn)
        self.stage4_3 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn)
        self.stage4_4 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn)
        self.stage4_5 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn)

        # stage5
        use_dcn = 5 in dcn_v2_stages
        self.stage5_0 = ConvBlock(1024, [512, 512, 2048], bn, gn, af, use_dcn=use_dcn, downsample_in3x3=downsample_in3x3)
        self.stage5_1 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, use_dcn=use_dcn)
        self.stage5_2 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, use_dcn=use_dcn)

    def __call__(self, input_tensor):
        x = self.stage1_conv1_1(input_tensor)
        x = self.stage1_conv1_2(x)
        x = self.stage1_conv1_3(x)
        x = self.pool(x)

        # stage2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
        # stage3
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
        # stage4
        x = self.stage4_0(s8)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        s16 = self.stage4_5(x)
        # stage5
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)

        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs

    def get_block(self, name):
        layer = getattr(self, name)
        return layer

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.stage1_conv1_1.freeze()
            self.stage1_conv1_2.freeze()
            self.stage1_conv1_3.freeze()
        if freeze_at >= 2:
            self.stage2_0.freeze()
            self.stage2_1.freeze()
            self.stage2_2.freeze()
        if freeze_at >= 3:
            self.stage3_0.freeze()
            self.stage3_1.freeze()
            self.stage3_2.freeze()
            self.stage3_3.freeze()
        if freeze_at >= 4:
            self.stage4_0.freeze()
            self.stage4_1.freeze()
            self.stage4_2.freeze()
            self.stage4_3.freeze()
            self.stage4_4.freeze()
            self.stage4_5.freeze()
        if freeze_at >= 5:
            self.stage5_0.freeze()
            self.stage5_1.freeze()
            self.stage5_2.freeze()



