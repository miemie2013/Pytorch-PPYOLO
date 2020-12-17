#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
import numpy as np
import math
import torch
import torch as T
import torch.nn.functional as F
import copy

from model.custom_layers import *
from model.matrix_nms import matrix_nms


def yolo_box(conv_output, anchors, stride, num_classes, scale_x_y, im_size, clip_bbox, conf_thresh):
    conv_output = conv_output.permute(0, 2, 3, 1)
    conv_shape       = conv_output.shape
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = conv_output.reshape((batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    rows = T.arange(0, output_size, dtype=T.float32, device=conv_raw_dxdy.device)
    cols = T.arange(0, output_size, dtype=T.float32, device=conv_raw_dxdy.device)
    rows = rows[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis].repeat((1, output_size, 1, 1, 1))
    cols = cols[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis].repeat((1, 1, output_size, 1, 1))
    offset = T.cat([rows, cols], dim=-1)
    offset = offset.repeat((batch_size, 1, 1, anchor_per_scale, 1))
    # Grid Sensitive
    pred_xy = (scale_x_y * T.sigmoid(conv_raw_dxdy) + offset - (scale_x_y - 1.0) * 0.5 ) * stride

    # _anchors = T.Tensor(anchors, device=conv_raw_dxdy.device)   # RuntimeError: legacy constructor for device type: cpu was passed device type: cuda, but device type must be: cpu
    _anchors = T.Tensor(anchors).cuda()
    pred_wh = (T.exp(conv_raw_dwdh) * _anchors)

    pred_xyxy = T.cat([pred_xy - pred_wh / 2, pred_xy + pred_wh / 2], dim=-1)   # 左上角xy + 右下角xy
    pred_conf = T.sigmoid(conv_raw_conf)
    # mask = (pred_conf > conf_thresh).float()
    pred_prob = T.sigmoid(conv_raw_prob)
    pred_scores = pred_conf * pred_prob
    # pred_scores = pred_scores * mask
    # pred_xyxy = pred_xyxy * mask

    # paddle中实际的顺序
    # pred_xyxy = pred_xyxy.permute(0, 3, 1, 2, 4)
    # pred_scores = pred_scores.permute(0, 3, 1, 2, 4)

    pred_xyxy = pred_xyxy.reshape((batch_size, output_size*output_size*anchor_per_scale, 4))
    pred_scores = pred_scores.reshape((batch_size, pred_xyxy.shape[1], num_classes))

    _im_size_h = im_size[:, 0:1]
    _im_size_w = im_size[:, 1:2]
    _im_size = T.cat([_im_size_w, _im_size_h], 1)
    _im_size = _im_size.unsqueeze(1)
    _im_size = _im_size.repeat((1, pred_xyxy.shape[1], 1))
    pred_x0y0 = pred_xyxy[:, :, 0:2] / output_size / stride * _im_size
    pred_x1y1 = pred_xyxy[:, :, 2:4] / output_size / stride * _im_size
    if clip_bbox:
        x0 = pred_x0y0[:, :, 0:1]
        y0 = pred_x0y0[:, :, 1:2]
        x1 = pred_x1y1[:, :, 0:1]
        y1 = pred_x1y1[:, :, 1:2]
        x0 = torch.where(x0 < 0, x0 * 0, x0)
        y0 = torch.where(y0 < 0, y0 * 0, y0)
        x1 = torch.where(x1 > _im_size[:, :, 0:1], _im_size[:, :, 0:1], x1)
        y1 = torch.where(y1 > _im_size[:, :, 1:2], _im_size[:, :, 1:2], y1)
        pred_xyxy = T.cat([x0, y0, x1, y1], -1)
    else:
        pred_xyxy = T.cat([pred_x0y0, pred_x1y1], -1)
    return pred_xyxy, pred_scores


def _split_ioup(output, an_num, num_classes):
    """
    Split new output feature map to output, predicted iou
    along channel dimension
    """
    ioup = output[:, :an_num, :, :]
    ioup = torch.sigmoid(ioup)

    oriout = output[:, an_num:, :, :]

    return (ioup, oriout)


# sigmoid()函数的反函数。先取倒数再减一，取对数再取相反数。
def _de_sigmoid(x, eps=1e-7):
    # x限制在区间[eps, 1 / eps]内
    x = torch.clamp(x, eps, 1 / eps)

    # 先取倒数再减一
    x = 1.0 / x - 1.0

    # e^(-x)限制在区间[eps, 1 / eps]内
    x = torch.clamp(x, eps, 1 / eps)

    # 取对数再取相反数
    x = -torch.log(x)
    return x


def _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor):
    """
    post process output objectness score
    """
    tensors = []
    stride = output.shape[1] // an_num
    for m in range(an_num):
        tensors.append(output[:, stride * m:stride * m + 4, :, :])
        obj = output[:, stride * m + 4:stride * m + 5, :, :]
        obj = torch.sigmoid(obj)

        ip = ioup[:, m:m + 1, :, :]

        new_obj = torch.pow(obj, (1 - iou_aware_factor)) * torch.pow(ip, iou_aware_factor)
        new_obj = _de_sigmoid(new_obj)   # 置信位未进行sigmoid()激活

        tensors.append(new_obj)

        tensors.append(output[:, stride * m + 5:stride * m + 5 + num_classes, :, :])

    output = torch.cat(tensors, dim=1)

    return output



def get_iou_aware_score(output, an_num, num_classes, iou_aware_factor):
    ioup, output = _split_ioup(output, an_num, num_classes)
    output = _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor)
    return output




class DetectionBlock(torch.nn.Module):
    def __init__(self,
                 in_c,
                 channel,
                 coord_conv=True,
                 bn=0,
                 gn=0,
                 af=0,
                 norm_decay=0.,
                 conv_block_num=2,
                 is_first=False,
                 use_spp=True,
                 drop_block=True,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=True,
                 name=''):
        super(DetectionBlock, self).__init__()
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)
        self.norm_decay = norm_decay
        self.use_spp = use_spp
        self.coord_conv = coord_conv
        self.is_first = is_first
        self.is_test = is_test
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob

        self.layers = torch.nn.ModuleList()
        self.tip_layers = torch.nn.ModuleList()
        for j in range(conv_block_num):
            coordConv = CoordConv(coord_conv)
            input_c = in_c + 2 if coord_conv else in_c
            conv_unit1 = Conv2dUnit(input_c, channel, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.{}.0'.format(name, j))
            self.layers.append(coordConv)
            self.layers.append(conv_unit1)
            if self.use_spp and is_first and j == 1:
                spp = SPP()
                conv_unit2 = Conv2dUnit(channel * 4, 512, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.{}.spp.conv'.format(name, j))
                conv_unit3 = Conv2dUnit(512, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.{}.1'.format(name, j))
                self.layers.append(spp)
                self.layers.append(conv_unit2)
                self.layers.append(conv_unit3)
            else:
                conv_unit3 = Conv2dUnit(channel, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.{}.1'.format(name, j))
                self.layers.append(conv_unit3)

            if self.drop_block and j == 0 and not is_first:
                dropBlock = DropBlock(
                    block_size=self.block_size,
                    keep_prob=self.keep_prob,
                    is_test=is_test)
                self.layers.append(dropBlock)
            in_c = channel * 2

        if self.drop_block and is_first:
            dropBlock = DropBlock(
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)
            self.layers.append(dropBlock)
        coordConv = CoordConv(coord_conv)
        if conv_block_num == 0:
            input_c = in_c + 2 if coord_conv else in_c
        else:
            input_c = channel * 2 + 2 if coord_conv else channel * 2
        conv_unit = Conv2dUnit(input_c, channel, 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.2'.format(name))
        self.layers.append(coordConv)
        self.layers.append(conv_unit)

        coordConv = CoordConv(coord_conv)
        input_c = channel + 2 if coord_conv else channel
        conv_unit = Conv2dUnit(input_c, channel * 2, 3, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name='{}.tip'.format(name))
        self.tip_layers.append(coordConv)
        self.tip_layers.append(conv_unit)

    def __call__(self, input):
        conv = input
        for ly in self.layers:
            conv = ly(conv)
        route = conv
        tip = conv
        for ly in self.tip_layers:
            tip = ly(tip)
        return route, tip


class YOLOv3Head(torch.nn.Module):
    def __init__(self,
                 conv_block_num=2,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23],
                          [30, 61], [62, 45], [59, 119],
                          [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 norm_type="bn",
                 norm_decay=0.,
                 coord_conv=True,
                 iou_aware=True,
                 iou_aware_factor=0.4,
                 block_size=3,
                 scale_x_y=1.05,
                 spp=True,
                 drop_block=True,
                 keep_prob=0.9,
                 clip_bbox=True,
                 yolo_loss=None,
                 downsample=[32, 16, 8],
                 in_channels=[2048, 1024, 512],
                 nms_cfg=None,
                 focalloss_on_obj=False,
                 prior_prob=0.01,
                 is_train=False
                 ):
        super(YOLOv3Head, self).__init__()
        self.conv_block_num = conv_block_num
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.coord_conv = coord_conv
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor
        self.scale_x_y = scale_x_y
        self.use_spp = spp
        self.drop_block = drop_block
        self.keep_prob = keep_prob
        self.clip_bbox = clip_bbox
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.block_size = block_size
        self.downsample = downsample
        self.in_channels = in_channels
        self.yolo_loss = yolo_loss
        self.nms_cfg = nms_cfg
        self.focalloss_on_obj = focalloss_on_obj
        self.prior_prob = prior_prob
        self.is_train = is_train

        _anchors = copy.deepcopy(anchors)
        _anchors = np.array(_anchors)
        _anchors = _anchors.astype(np.float32)
        self._anchors = _anchors   # [9, 2]

        self.mask_anchors = []
        for m in anchor_masks:
            temp = []
            for aid in m:
                temp += anchors[aid]
            self.mask_anchors.append(temp)

        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)

        self.detection_blocks = torch.nn.ModuleList()
        self.yolo_output_convs = torch.nn.ModuleList()
        self.upsample_layers = torch.nn.ModuleList()
        out_layer_num = len(downsample)
        for i in range(out_layer_num):
            in_c = self.in_channels[i]
            if i > 0:  # perform concat in first 2 detection_block
                in_c = self.in_channels[i] + 512 // (2**i)
            _detection_block = DetectionBlock(
                in_c=in_c,
                channel=64 * (2**out_layer_num) // (2**i),
                coord_conv=self.coord_conv,
                bn=bn,
                gn=gn,
                af=af,
                norm_decay=self.norm_decay,
                is_first=i == 0,
                conv_block_num=self.conv_block_num,
                use_spp=self.use_spp,
                drop_block=self.drop_block,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=(not self.is_train),
                name="yolo_block.{}".format(i)
            )
            # out channel number = mask_num * (5 + class_num)
            if self.iou_aware:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            bias_init = None
            # if self.focalloss_on_obj:
            #     # 设置偏移的初始值使得obj预测概率初始值为self.prior_prob (根据激活函数是sigmoid()时推导出)
            #     bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            #     bias_init_array = np.zeros((num_filters, ), np.float32)
            #     an_num = len(self.anchor_masks[i])
            #     start = 0
            #     stride = (self.num_classes + 5)
            #     if self.iou_aware:
            #         start = an_num
            #     # 只设置置信位
            #     for o in range(an_num):
            #         bias_init_array[start + o * stride + 4] = bias_init_value
            #     bias_init = fluid.initializer.NumpyArrayInitializer(bias_init_array)
            yolo_output_conv = Conv2dUnit(64 * (2**out_layer_num) // (2**i) * 2, num_filters, 1, stride=1, bias_attr=True, act=None,
                                          bias_init=bias_init, name="yolo_output.{}.conv".format(i))
            self.detection_blocks.append(_detection_block)
            self.yolo_output_convs.append(yolo_output_conv)


            if i < out_layer_num - 1:
                # do not perform upsample in the last detection_block
                conv_unit = Conv2dUnit(64 * (2**out_layer_num) // (2**i), 256 // (2**i), 1, stride=1, bn=bn, gn=gn, af=af, act='leaky', norm_decay=self.norm_decay, name="yolo_transition.{}".format(i))
                # upsample
                upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
                self.upsample_layers.append(conv_unit)
                self.upsample_layers.append(upsample)

    def set_dropblock(self, is_test):
        for detection_block in self.detection_blocks:
            for l in detection_block.layers:
                if isinstance(l, DropBlock):
                    l.is_test = is_test

    def _get_outputs(self, body_feats):
        outputs = []

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = body_feats[-1:-out_layer_num - 1:-1]

        route = None
        for i, block in enumerate(blocks):
            if i > 0:  # perform concat in first 2 detection_block
                block = torch.cat([route, block], dim=1)
            route, tip = self.detection_blocks[i](block)
            block_out = self.yolo_output_convs[i](tip)
            outputs.append(block_out)
            if i < out_layer_num - 1:
                route = self.upsample_layers[i*2](route)
                route = self.upsample_layers[i*2+1](route)
        return outputs

    def get_loss(self, input, gt_box, gt_label, gt_score, targets):
        """
        Get final loss of network of YOLOv3.

        Args:
            input (list): List of Variables, output of backbone stages
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.
            targets ([Variables]): List of Variables, the targets for yolo
                                   loss calculatation.

        Returns:
            loss (Variable): The loss Variable of YOLOv3 network.

        """

        # outputs里为大中小感受野的输出
        outputs = self._get_outputs(input)

        return self.yolo_loss(outputs, gt_box, gt_label, gt_score, targets,
                              self.anchors, self.anchor_masks,
                              self.mask_anchors, self.num_classes)

    def get_prediction(self, body_feats, im_size):
        """
        Get prediction result of YOLOv3 network

        Args:
            input (list): List of Variables, output of backbone stages
            im_size (Variable): Variable of size([h, w]) of each image

        Returns:
            pred (Variable): shape = [bs, keep_top_k, 6]

        """
        # outputs里为大中小感受野的输出
        outputs = self._get_outputs(body_feats)

        boxes = []
        scores = []
        for i, output in enumerate(outputs):
            if self.iou_aware:
                output = get_iou_aware_score(output,
                                             len(self.anchor_masks[i]),
                                             self.num_classes,
                                             self.iou_aware_factor)
            box, score = yolo_box(output, self._anchors[self.anchor_masks[i]], self.downsample[i],
                                  self.num_classes, self.scale_x_y, im_size, self.clip_bbox,
                                  conf_thresh=self.nms_cfg['score_threshold'])
            boxes.append(box)
            scores.append(score)
        yolo_boxes = torch.cat(boxes, dim=1)
        yolo_scores = torch.cat(scores, dim=1)


        # nms
        preds = []
        nms_cfg = copy.deepcopy(self.nms_cfg)
        nms_type = nms_cfg.pop('nms_type')
        batch_size = yolo_boxes.shape[0]
        if nms_type == 'matrix_nms':
            for i in range(batch_size):
                pred = matrix_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                preds.append(pred)
        # elif nms_type == 'multiclass_nms':
        #     for i in range(batch_size):
        #         pred = multiclass_nms(yolo_boxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
        #         preds.append(pred)
        return preds





