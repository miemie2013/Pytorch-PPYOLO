


安装paddle1.8.4
pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple



安装paddle2.0
nvidia-smi
pip install pycocotools
python -m pip install paddlepaddle_gpu==2.0.0b0 -f https://paddlepaddle.org.cn/whl/stable.html
cd ~/w*






-------------------------------- PPYOLO --------------------------------
parser.add_argument('-c', '--config', type=int, default=0,
                    choices=[0, 1, 2],
                    help=textwrap.dedent('''\
                    select one of these config files:
                    0 -- ppyolo_2x.py
                    1 -- ppyolo_2x.py
                    2 -- ppyolo_r18vd.py'''))

训练
cd ~/w*
python train.py --config=0

cd ~/w*
python train.py --config=1

cd ~/w*
python train.py --config=2

cd ~/w*
python train.py --config=3

cd ~/w*
python train.py --config=4

cd ~/w*
python train.py --config=5




预测
cd ~/w*
python demo.py --config=0

cd ~/w*
python demo.py --config=1

cd ~/w*
python demo.py --config=2

cd ~/w*
python demo.py --config=3

cd ~/w*
python demo.py --config=4

cd ~/w*
python demo.py --config=5





验证
cd ~/w*
python eval.py --config=0

cd ~/w*
python eval.py --config=1

cd ~/w*
python eval.py --config=2

cd ~/w*
python eval.py --config=3

cd ~/w*
python eval.py --config=4

cd ~/w*
python eval.py --config=5




跑test_dev
cd ~/w*
python test_dev.py --config=0

cd ~/w*
python test_dev.py --config=1

cd ~/w*
python test_dev.py --config=2

cd ~/w*
python test_dev.py --config=3

cd ~/w*
python test_dev.py --config=4

cd ~/w*
python test_dev.py --config=5




















