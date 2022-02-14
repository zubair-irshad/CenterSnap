# Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation
<img src="demo/Pytorch_logo.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the pytorch implementation of our paper:
<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="demo/tri-logo.png" width="20%"/>
</a>

**Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation**<br>
[__***Muhammad Zubair Irshad***__](https://zubairirshad.com), [Thomas Kollar](http://www.tkollar.com/site/), [Michael Laskey](https://www.linkedin.com/in/michael-laskey-4b087ba2/), [Kevin Stone](https://www.linkedin.com/in/kevin-stone-51171270/), [Zsolt Kira](https://faculty.cc.gatech.edu/~zk15/) <br>
International Conference on Robotics and Automation (ICRA), 2022<br>

[[Project Page](https://zubair-irshad.github.io/projects/robo-vln.html)] [[arXiv](https://arxiv.org/abs/2104.10674)] [[GitHub](https://github.com/GT-RIPL/robo-vln)] 

Code coming soon!

<p align="center">
<img src="demo/POSE_CS.gif" width="100%">
</p>

<p align="center">
<img src="demo/Method_CS.gif" width="100%">
</p>


## Environment

Create a python 3.8 virtual environment and install requirements:

```bash
cd $SIMNET_REPO
conda create -y --prefix ./env python=3.8
./env/bin/python -m pip install --upgrade pip
./env/bin/python -m pip install -r requirements.txt
```

#### Datasets

Download and untar train+val datasets
[simnet2021a.tar](https://tri-robotics-public.s3.amazonaws.com/github/simnet/datasets/simnet2021a.tar)
(18GB, md5 checksum:`b8e1d3cb7200b44b1de223e87141f14b`). This file contains all the training and
validation you need to replicate our small objects results.

```bash
cd $SIMNET_REPO
wget https://tri-robotics-public.s3.amazonaws.com/github/simnet/datasets/simnet2021a.tar -P datasets
tar xf datasets/simnet2021a.tar -C datasets
```


### Train and Validate

Overfit test:
```bash
./runner.sh net_train.py @config/net_config_overfit.txt
```
 
Full training run (requires 13GB GPU memory)
```bash
./runner.sh inference/inference_real.py @configs/net_config.txt --data_dir data_dir_here --checkpoint checkpoint_path_here
```
