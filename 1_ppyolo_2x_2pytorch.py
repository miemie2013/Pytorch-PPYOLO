#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
from config import *
from model.custom_layers import DCNv2
from model.ppyolo import PPYOLO
import paddle.fluid as fluid


use_gpu = True


cfg = PPYOLO_2x_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes
model_path = 'ppyolo.pdparams'




import torch

def load_weights(path):
    state_dict = fluid.io.load_program_state(path)
    return state_dict

state_dict = load_weights(model_path)
print('============================================================')




# 创建模型
Backbone = select_backbone(cfg.backbone_type)
backbone = Backbone(**cfg.backbone)
Head = select_head(cfg.head_type)
head = Head(yolo_loss=None, **cfg.head)
ppyolo = PPYOLO(backbone, head)
if use_gpu:
    ppyolo = ppyolo.cuda()
ppyolo.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

print('\nCopying...')


def copy_conv_bn(conv_unit, w, scale, offset, m, v):
    conv_unit.conv.weight.data = torch.Tensor(w).cuda()
    conv_unit.bn.weight.data = torch.Tensor(scale).cuda()
    conv_unit.bn.bias.data = torch.Tensor(offset).cuda()
    conv_unit.bn.running_mean.data = torch.Tensor(m).cuda()
    conv_unit.bn.running_var.data = torch.Tensor(v).cuda()

def copy_conv(conv_layer, w, b):
    conv_layer.weight.data = torch.Tensor(w).cuda()
    conv_layer.bias.data = torch.Tensor(b).cuda()



# Resnet50Vd
w = state_dict['conv1_1_weights']
scale = state_dict['bnv1_1_scale']
offset = state_dict['bnv1_1_offset']
m = state_dict['bnv1_1_mean']
v = state_dict['bnv1_1_variance']
copy_conv_bn(backbone.stage1_conv1_1, w, scale, offset, m, v)

w = state_dict['conv1_2_weights']
scale = state_dict['bnv1_2_scale']
offset = state_dict['bnv1_2_offset']
m = state_dict['bnv1_2_mean']
v = state_dict['bnv1_2_variance']
copy_conv_bn(backbone.stage1_conv1_2, w, scale, offset, m, v)

w = state_dict['conv1_3_weights']
scale = state_dict['bnv1_3_scale']
offset = state_dict['bnv1_3_offset']
m = state_dict['bnv1_3_mean']
v = state_dict['bnv1_3_variance']
copy_conv_bn(backbone.stage1_conv1_3, w, scale, offset, m, v)


nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)
        conv_name1 = block_name + "_branch2a"
        conv_name2 = block_name + "_branch2b"
        conv_name3 = block_name + "_branch2c"
        shortcut_name = block_name + "_branch1"

        bn_name1 = 'bn' + conv_name1[3:]
        bn_name2 = 'bn' + conv_name2[3:]
        bn_name3 = 'bn' + conv_name3[3:]
        shortcut_bn_name = 'bn' + shortcut_name[3:]


        w = state_dict[conv_name1 + '_weights']
        scale = state_dict[bn_name1 + '_scale']
        offset = state_dict[bn_name1 + '_offset']
        m = state_dict[bn_name1 + '_mean']
        v = state_dict[bn_name1 + '_variance']
        copy_conv_bn(backbone.get_block('stage%d_%d' % (2+nid, kk)).conv1, w, scale, offset, m, v)

        if nid == 3:   # DCNv2
            conv_unit = backbone.get_block('stage%d_%d' % (2+nid, kk)).conv2

            offset_w = state_dict[conv_name2 + '_conv_offset.w_0']
            offset_b = state_dict[conv_name2 + '_conv_offset.b_0']
            if isinstance(conv_unit.conv, DCNv2):   # 如果是自实现的DCNv2
                copy_conv(conv_unit.conv.conv_offset, offset_w, offset_b)
            else:
                copy_conv(conv_unit.conv.conv_offset_mask, offset_w, offset_b)

            w = state_dict[conv_name2 + '_weights']
            scale = state_dict[bn_name2 + '_scale']
            offset = state_dict[bn_name2 + '_offset']
            m = state_dict[bn_name2 + '_mean']
            v = state_dict[bn_name2 + '_variance']

            if isinstance(conv_unit.conv, DCNv2):   # 如果是自实现的DCNv2
                conv_unit.conv.dcn_weight.data = torch.Tensor(w).cuda()
                conv_unit.bn.weight.data = torch.Tensor(scale).cuda()
                conv_unit.bn.bias.data = torch.Tensor(offset).cuda()
                conv_unit.bn.running_mean.data = torch.Tensor(m).cuda()
                conv_unit.bn.running_var.data = torch.Tensor(v).cuda()
            else:
                copy_conv_bn(conv_unit, w, scale, offset, m, v)
        else:
            w = state_dict[conv_name2 + '_weights']
            scale = state_dict[bn_name2 + '_scale']
            offset = state_dict[bn_name2 + '_offset']
            m = state_dict[bn_name2 + '_mean']
            v = state_dict[bn_name2 + '_variance']
            copy_conv_bn(backbone.get_block('stage%d_%d' % (2+nid, kk)).conv2, w, scale, offset, m, v)

        w = state_dict[conv_name3 + '_weights']
        scale = state_dict[bn_name3 + '_scale']
        offset = state_dict[bn_name3 + '_offset']
        m = state_dict[bn_name3 + '_mean']
        v = state_dict[bn_name3 + '_variance']
        copy_conv_bn(backbone.get_block('stage%d_%d' % (2+nid, kk)).conv3, w, scale, offset, m, v)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            w = state_dict[shortcut_name + '_weights']
            scale = state_dict[shortcut_bn_name + '_scale']
            offset = state_dict[shortcut_bn_name + '_offset']
            m = state_dict[shortcut_bn_name + '_mean']
            v = state_dict[shortcut_bn_name + '_variance']
            copy_conv_bn(backbone.get_block('stage%d_%d' % (2+nid, kk)).conv4, w, scale, offset, m, v)


# head

conv_block_num = 2
num_classes = 80
anchors = [[10, 13], [16, 30], [33, 23],
           [30, 61], [62, 45], [59, 119],
           [116, 90], [156, 198], [373, 326]]
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
batch_size = 1
norm_type = "bn"
coord_conv = True
iou_aware = True
iou_aware_factor = 0.4
block_size = 3
scale_x_y = 1.05
use_spp = True
drop_block = True
keep_prob = 0.9
clip_bbox = True
yolo_loss = None
downsample = [32, 16, 8]
in_channels = [2048, 1024, 512]
nms_cfg = None
is_train = False

bn = 0
gn = 0
af = 0
if norm_type == 'bn':
    bn = 1
elif norm_type == 'gn':
    gn = 1
elif norm_type == 'affine_channel':
    af = 1



def copy_DetectionBlock(
        _detection_block,
        in_c,
             channel,
             coord_conv=True,
             bn=0,
             gn=0,
             af=0,
             conv_block_num=2,
             is_first=False,
             use_spp=True,
             drop_block=True,
             block_size=3,
             keep_prob=0.9,
             is_test=True,
        name=''):
    kkk = 0
    for j in range(conv_block_num):
        kkk += 1

        conv_name = '{}.{}.0'.format(name, j)
        w = state_dict[conv_name + '.conv.weights']
        scale = state_dict[conv_name + '.bn.scale']
        offset = state_dict[conv_name + '.bn.offset']
        m = state_dict[conv_name + '.bn.mean']
        v = state_dict[conv_name + '.bn.var']
        copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v)
        kkk += 1


        if use_spp and is_first and j == 1:
            kkk += 1

            conv_name = '{}.{}.spp.conv'.format(name, j)
            w = state_dict[conv_name + '.conv.weights']
            scale = state_dict[conv_name + '.bn.scale']
            offset = state_dict[conv_name + '.bn.offset']
            m = state_dict[conv_name + '.bn.mean']
            v = state_dict[conv_name + '.bn.var']
            copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v)
            kkk += 1

            conv_name = '{}.{}.1'.format(name, j)
            w = state_dict[conv_name + '.conv.weights']
            scale = state_dict[conv_name + '.bn.scale']
            offset = state_dict[conv_name + '.bn.offset']
            m = state_dict[conv_name + '.bn.mean']
            v = state_dict[conv_name + '.bn.var']
            copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v)
            kkk += 1
        else:
            conv_name = '{}.{}.1'.format(name, j)
            w = state_dict[conv_name + '.conv.weights']
            scale = state_dict[conv_name + '.bn.scale']
            offset = state_dict[conv_name + '.bn.offset']
            m = state_dict[conv_name + '.bn.mean']
            v = state_dict[conv_name + '.bn.var']
            copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v)
            kkk += 1

        if drop_block and j == 0 and not is_first:
            kkk += 1

    if drop_block and is_first:
        kkk += 1

    kkk += 1

    conv_name = '{}.2'.format(name)
    w = state_dict[conv_name + '.conv.weights']
    scale = state_dict[conv_name + '.bn.scale']
    offset = state_dict[conv_name + '.bn.offset']
    m = state_dict[conv_name + '.bn.mean']
    v = state_dict[conv_name + '.bn.var']
    copy_conv_bn(_detection_block.layers[kkk], w, scale, offset, m, v)
    kkk += 1

    conv_name = '{}.tip'.format(name)
    w = state_dict[conv_name + '.conv.weights']
    scale = state_dict[conv_name + '.bn.scale']
    offset = state_dict[conv_name + '.bn.offset']
    m = state_dict[conv_name + '.bn.mean']
    v = state_dict[conv_name + '.bn.var']
    copy_conv_bn(_detection_block.tip_layers[1], w, scale, offset, m, v)



out_layer_num = len(downsample)
for i in range(out_layer_num):
    copy_DetectionBlock(
        head.detection_blocks[i],
        in_c=in_channels[i],
        channel=64 * (2**out_layer_num) // (2**i),
        coord_conv=coord_conv,
        bn=bn,
        gn=gn,
        af=af,
        is_first=i == 0,
        conv_block_num=conv_block_num,
        use_spp=use_spp,
        drop_block=drop_block,
        block_size=block_size,
        keep_prob=keep_prob,
        is_test=(not is_train),
        name="yolo_block.{}".format(i)
    )

    w = state_dict["yolo_output.{}.conv.weights".format(i)]
    b = state_dict["yolo_output.{}.conv.bias".format(i)]
    copy_conv(head.yolo_output_convs[i].conv, w, b)

    if i < out_layer_num - 1:
        conv_name = "yolo_transition.{}".format(i)
        w = state_dict[conv_name + '.conv.weights']
        scale = state_dict[conv_name + '.bn.scale']
        offset = state_dict[conv_name + '.bn.offset']
        m = state_dict[conv_name + '.bn.mean']
        v = state_dict[conv_name + '.bn.var']
        copy_conv_bn(head.upsample_layers[i*2], w, scale, offset, m, v)




torch.save(ppyolo.state_dict(), 'ppyolo_2x.pt')
print('\nDone.')







