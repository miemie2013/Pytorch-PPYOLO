# Pytorch-PPYOLO

## 概述
PP-YOLO是[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)优化和改进的YOLOv3的模型，其精度(COCO数据集mAP)和推理速度均优于YOLOv4模型。


| 算法 | 骨干网络 | 图片输入大小 | mAP(COCO val2017) | mAP(COCO test2017) | FPS  |
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|
| PPYOLO    | ResNet50-vd | (608,608)  | 0.448  | 0.451  | 16.6 |
| PPYOLO    | ResNet50-vd | (320,320)  | 0.389  | -  | 34.1 |
| PPYOLO_r18vd    | ResNet18-vd | (608,608)  | 0.286  | -  | 51.7 |
| PPYOLO_r18vd    | ResNet18-vd | (416,416)  | 0.286  | -  | 76.2 |
| PPYOLO_r18vd    | ResNet18-vd | (320,320)  | 0.262  | -  | 93.3 |


**注意:**

- 测速环境为：  ubuntu18.04, i5-9400F, 8GB RAM, GTX1660Ti(6GB), cuda10.0。使用了自实现的DCNv2。windows上可能没linux上FPS高。
- FPS由demo.py测得。预测50张图片，预测之前会有一个热身(warm up)阶段使速度稳定。
- PPYOLO_r18vd(416,416) mAP(IoU=0.50)(COCO val2017)为0.470，表中的0.286指的是mAP(IoU=0.50:0.95)(COCO val2017)。
- PPYOLO_r18vd(608,608) mAP(IoU=0.50)(COCO val2017)为0.478。不建议使用608x608输入大小。
- PPYOLO_r18vd(320,320) mAP(IoU=0.50)(COCO val2017)为0.437。


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

2020/11/05:咩酱成功实现DCNv2，不用编译c、c++、cuda、自定义op这些玩意了！

## 已实现的部分

EMA(指数滑动平均)：修改config/ppyolo_2x.py中self.use_ema = True打开。修改config/ppyolo_2x.py中self.use_ema = False关闭。打开ema会拖慢训练速度。由于new_val = np.array(param.cpu().detach().numpy().copy())这一句本身是耗时的，而且无法与训练过程并行，咩酱暂时想不到好办法优化这一部分。

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

因为咩酱用Pytorch的纯python接口实现了DCNv2，效率极高，custom_layers.py里默认使用的也是咩酱自己实现的DCNv2，所以不用编译官方的DCNv2。但是如果读者想试试官方的DCNv2，与咩酱实现的DCNv2对比速度，输入以下命令编译和安装：

```
cd external/DCNv2
python setup.py build develop
```


## 快速开始

(1)环境搭建

需要安装cuda10，Pytorch1.x。

(2)下载预训练模型

下载PaddleDetection的ppyolo.pdparams。如果你使用Linux，请使用以下命令：
```
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
```

如果你使用Windows，请复制以下网址到浏览器或迅雷下载：
```
https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
```
下载好后将它放在项目根目录下。然后运行1_ppyolo_2x_2pytorch.py得到一个ppyolo_2x.pt，它也位于根目录下。


下载PaddleDetection的ppyolo_r18vd.pdparams。如果你使用Linux，请使用以下命令：
```
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams
```

如果你使用Windows，请复制以下网址到浏览器或迅雷下载：
```
https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams
```
下载好后将它放在项目根目录下。然后运行1_ppyolo_r18vd_2pytorch.py得到一个ppyolo_r18vd.pt，它也位于根目录下。

(3)预测图片、获取FPS（预测images/test/里的图片，结果保存在images/res/）

(如果使用ppyolo_2x.py配置文件)
```
python demo.py --config=0
```

(如果使用ppyolo_r18vd.py配置文件)
```
python demo.py --config=2
```


## 训练
(如果使用ppyolo_2x.py配置文件)
```
python train.py --config=0
```

通过修改config/xxxxxxx.py的代码来进行更换数据集、更改超参数以及训练参数。

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
(如果使用ppyolo_2x.py配置文件)
```
python eval.py --config=0
```


## test-dev
(如果使用ppyolo_2x.py配置文件)
```
python test_dev.py --config=0
```


运行完之后，进入results目录，把bbox_detections.json压缩成bbox_detections.zip，提交到
https://competitions.codalab.org/competitions/20794#participate
获得bbox mAP。该mAP是test集的结果，也就是大部分检测算法论文的标准指标。


## 预测
(如果使用ppyolo_2x.py配置文件)
```
python demo.py --config=0
```


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
