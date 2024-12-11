# HyperFL

The implementation of [A New Federated Learning Framework Against Gradient Inversion Attacks](https://arxiv.org/abs/2412.07187) [AAAI 2025]. \
[Pengxin Guo](https://pengxin-guo.github.io)\*, [Shuang Zeng](https://scholar.google.com/citations?user=yTP1oqkAAAAJ&hl=en)\*, Wenhao Chen, Xiaodan Zhang, Weihong Ren, Yuyin Zhou, and [Liangqiong Qu](https://liangqiong.github.io).


## Requirements

Some important required packages are lised below:

- Python 3.10
- Pytorch 2.0.1
- torchvision 0.15.2
- timm 0.9.2


## Usage

### 1. Create a conda environment

```bash
cd ./HyperFL
conda create -n hyperfl python=3.10
conda activate hyperfl
pip install -r requirements.txt
```

### 2. Train and test the model
#### HyperFL
```bash
cd ./cnn
python federated_main.py --gpu 0 --train_rule HyperFL --dataset cifar --local_bs 50 --lr 0.02 --num_users 20 --frac 1.0
```

#### HyperFL-LPM

```bash
cd ./vit
python federated_main.py --gpu 0 --train_rule HyperFL-LPM --dataset cifar
```

```bash
cd ./resnet
python federated_main.py --gpu 0 --train_rule HyperFL-LPM --dataset cifar
```

## Acknowledgement
We would like to thank the authors for releasing the public repository: [FedPAC](https://github.com/JianXu95/FedPAC).