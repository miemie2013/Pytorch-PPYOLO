# Pytorch-PPYOLO

## 概述
Pytorch实现PPYOLO。

参考自https://github.com/PaddlePaddle/PaddleDetection

请支持正版，嘤嘤嘤~

## 咩酱刷屏时刻

Keras版YOLOv3: https://github.com/miemie2013/Keras-DIOU-YOLOv3

Pytorch版YOLOv3：https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle版YOLOv3：https://github.com/miemie2013/Paddle-DIOU-YOLOv3

PaddlePaddle完美复刻版版yolact: https://github.com/miemie2013/PaddlePaddle_yolact

Keras版YOLOv4: https://github.com/miemie2013/Keras-YOLOv4 (mAP 41%+)

Pytorch版YOLOv4: https://github.com/miemie2013/Pytorch-YOLOv4 (mAP 41%+)

Paddle版YOLOv4：https://github.com/miemie2013/Paddle-YOLOv4 (mAP 41%+)

PaddleDetection版SOLOv2: https://github.com/miemie2013/PaddleDetection-SOLOv2

Pytorch实时版FCOS，跑得比YOLOv4快: https://github.com/miemie2013/Pytorch-FCOS

Paddle实时版FCOS，跑得比YOLOv4快: https://github.com/miemie2013/Paddle-FCOS

Keras版CartoonGAN: https://github.com/miemie2013/keras_CartoonGAN

纯python实现一个深度学习框架: https://github.com/miemie2013/Pure_Python_Deep_Learning

Pytorch版PPYOLO: https://github.com/miemie2013/Pytorch-PPYOLO (mAP 44.8%)

## 更新日记

2020/10/17:首次公开

2020/10/22:实现EMA

## 已实现的部分

EMA(指数滑动平均)：修改config/ppyolo_2x.py中self.use_ema = True打开。修改config/ppyolo_2x.py中self.use_ema = False关闭。

DropBlock：随机丢弃特征图上的像素。

IoU Loss：iou损失。

IoU Aware：预测预测框和gt的iou。并作用在objness上。

Grid Sensitive：预测框中心点的xy可以出到网格之外，应付gt中心点在网格线上这种情况。

Matrix NMS：SOLOv2中提出的算法，在soft-nms等基础上进行并行化加速，若预测框与同类高分框有iou，减小预测框的分数而不是直接丢弃。这里用box iou代替mask iou。

CoordConv：特征图带上像素的坐标信息（通道数+2）。

SPP：3个池化层的输出和原图拼接。


## 未实现的部分

多卡训练（由于咩酱只有一张6G的卡，也不是硕士生没有实验室，这部分可能不会实现）。

## 环境搭建

安装DCNv2
```
cd external/DCNv2
python setup.py build develop
```

## 训练
下载我从PaddleDetection的仓库保存下来的pytorch模型ppyolo_2x.pt
链接：https://pan.baidu.com/s/18ZUQMWF7qPJ7K7xqx1VnpQ 
提取码：6hph 

该模型在COCO2017 val集的mAP如下
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.649
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.492
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.571
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.624
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.420
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.665
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773
```


将它放在项目根目录下。
运行train.py进行训练。通过修改config/ppyolo_2x.py代码来进行更换数据集、更改超参数以及训练参数。

训练时如果发现mAP很稳定了，就停掉，修改学习率为原来的十分之一，接着继续训练，mAP还会再上升。暂时是这样手动操作。

## 训练自定义数据集
自带的voc2012数据集是一个很好的例子。

将自己数据集的txt注解文件放到annotation目录下，txt注解文件的格式如下：
```
xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx.jpg 48,240,195,371,11 8,12,352,498,14
# 图片名 物体1左上角x坐标,物体1左上角y坐标,物体1右下角x坐标,物体1右下角y坐标,物体1类别id 物体2左上角x坐标,物体2左上角y坐标,物体2右下角x坐标,物体2右下角y坐标,物体2类别id ...
```
运行1_txt2json.py会在annotation_json目录下生成两个coco注解风格的json注解文件，这是train.py支持的注解文件格式。
在config/ppyolo_2x.py里修改train_path、val_path、classes_path、train_pre_path、val_pre_path、num_classes这6个变量（自带的voc2012数据集直接解除注释就ok了）,就可以开始训练自己的数据集了。
而且，直接加载ppyolo_2x.pt的权重训练也是可以的，这时候也仅仅不加载3个输出卷积层的6个权重（因为类别数不同导致了输出通道数不同）。
如果需要跑demo.py、eval.py，与数据集有关的变量也需要修改一下，应该很容易看懂。

## 评估
训练时默认每5000步计算一次验证集的mAP。或者运行eval.py评估指定模型的mAP。该mAP是val集的结果。

## 预测
运行demo.py。

## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

AIStudio主页：[asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

欢迎在GitHub或AIStudio上关注我（求粉）~

## 打赏

如果你觉得这个仓库对你很有帮助，可以给我打钱↓
![Example 0](weixin/sk.png)

咩酱爱你哟！另外，有偿接私活，可联系微信wer186259，金主快点来吧！
