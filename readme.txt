


安装paddle1.8.4
pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple



安装paddle2.0
nvidia-smi
pip install pycocotools
python -m pip install paddlepaddle_gpu==2.0.0b0 -f https://paddlepaddle.org.cn/whl/stable.html
cd ~/w*






-------------------------------- PPYOLO --------------------------------
训练
python train.py --config=0


python train.py --config=2




预测
python demo.py --config=0


python demo.py --config=2



验证
python eval.py --config=0


python eval.py --config=2



跑test_dev
python test_dev.py --config=0


python test_dev.py --config=2






















