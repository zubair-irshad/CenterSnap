# CenterSnap: Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/centersnap-single-shot-multi-object-3d-shape/6d-pose-estimation-using-rgbd-on-camera25)](https://paperswithcode.com/sota/6d-pose-estimation-using-rgbd-on-camera25?p=centersnap-single-shot-multi-object-3d-shape)<img src="demo/Pytorch_logo.png" width="10%">

This repository is the pytorch implementation of our paper:
<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="demo/tri-logo.png" width="20%"/>
</a>

**CenterSnap: Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation**<br>
[__***Muhammad Zubair Irshad***__](https://zubairirshad.com), [Thomas Kollar](http://www.tkollar.com/site/), [Michael Laskey](https://www.linkedin.com/in/michael-laskey-4b087ba2/), [Kevin Stone](https://www.linkedin.com/in/kevin-stone-51171270/), [Zsolt Kira](https://faculty.cc.gatech.edu/~zk15/) <br>
International Conference on Robotics and Automation (ICRA), 2022<br>

[[Project Page](https://zubair-irshad.github.io/projects/CenterSnap.html)] [[arXiv](https://arxiv.org/abs/2203.01929)] [[PDF](https://arxiv.org/pdf/2203.01929.pdf)] [[Video](https://www.youtube.com/watch?v=Bg5vi6DSMdM)] [[GitHub](https://github.com/zubair-irshad/CenterSnap)] 

[![Explore CenterSnap in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zubair-irshad/CenterSnap/blob/pre_public/notebook/explore_CenterSnap.ipynb)<br>

<p align="center">
<img src="demo/POSE_CS.gif" width="100%">
</p>

<p align="center">
<img src="demo/reconstruction.gif" width="100%">
</p>

<p align="center">
<img src="demo/Method_CS.gif" width="100%">
</p>

## Citation

If you find this repository useful, please consider citing:

```
@inproceedings{irshad2022centersnap,
  title={CenterSnap: Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation},
  author={Muhammad Zubair Irshad and Thomas Kollar and Michael Laskey and Kevin Stone and Zsolt Kira},
  journal={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022},
  url={https://arxiv.org/abs/2203.01929},
}
```

### Contents
<div class="toc">
<ul>
<li><a href="#-environment">💻 Environment</a></li>
<li><a href="#-dataset">📊 Dataset</a></li>
<li><a href="#-training-and-validate">✨ Training and Inference</a></li>
<li><a href="#-faqs">📝 FAQ</a></li>
</ul>
</div>

## 💻 Environment

Create a python 3.8 virtual environment and install requirements:

```bash
cd $CenterSnap_Repo
conda create -y --prefix ./env python=3.8
conda activate ./env/
./env/bin/python -m pip install --upgrade pip
./env/bin/python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
The code was built and tested on **cuda 10.2**

## 📊 Dataset


1. Download pre-processed dataset

2. To prepare your own dataset

Download [camera_train](http://download.cs.stanford.edu/orion/nocs/camera_train.zip), [camera_val](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip),
[real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[ground-truth annotations](http://download.cs.stanford.edu/orion/nocs/gts.zip),
[camera_composed_depth](http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip)
and [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)
provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019).<br/>
Unzip and organize these files in $ROOT/data as follows:
```
data
├── CAMERA
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── camera_full_depths
│   ├── train
│   └── val
├── gts
│   ├── val
│   └── real_test
└── obj_models
    ├── train
    ├── val
    ├── real_train
    └── real_test
```
Run python scripts to prepare the datasets.

## ✨ Training and Inference

1. Train on NOCS Synthetic (requires 13GB GPU memory):
```bash
./runner.sh net_train.py @configs/net_config.txt
```

2. Finetune on NOCS Real Train (Note that good results can be obtained after finetuning on the Real train set for only a few epochs i.e. 1-5):
```bash
./runner.sh net_train.py @configs/net_config_real_resume.txt --checkpoint \path\to\best\checkpoint
```
 
3. Inference on a NOCS Real Test Subset

Download a small NOCS Real subset from [[here](https://www.dropbox.com/s/yfenvre5fhx3oda/nocs_test_subset.tar.gz?dl=1)]

```bash
./runner.sh inference/inference_real.py @configs/net_config.txt --data_dir path_to_nocs_test_subset --checkpoint checkpoint_path_here
```

You should see the **visualizations** saved in ```results/CenterSnap```. Change the --ouput_path in *config.txt to save them to a different folder

4. Optional (Shape Auto-Encoder Pre-training)

We provide pretrained model for shape auto-encoder to be used for data collection and inference. Although our codebase doesn't require separately training the shape auto-encoder, if you would like to do so, we provide additional scripts under **external/shape_pretraining**


## 📝 FAQ

**1.** I am getting ```no cuda GPUs available``` while running colab. 

- Ans: Make sure to follow this instruction to activate GPUs in colab:

```
Make sure that you have enabled the GPU under Runtime-> Change runtime type!
```

**2.** I am getting ```raise RuntimeError('received %d items of ancdata' %
RuntimeError: received 0 items of ancdata``` 

- Ans: Increase ulimit to 2048 or 8096 via ```uimit -n 2048```

```
Make sure that you have enabled the GPU under Runtime-> Change runtime type!
```

**3.** I am getting ``` RuntimeError: CUDA error: no kernel image is available for execution on the device``` 

- Ans: Check your pytorch installation with your cuda installation. Try the following:


1. Installing cuda 10.2 and running the same script in requirements.txt

2. Installing the relevant pytorch cuda version i.e. changing this line in the requirements.txt

```
torch==1.7.1
torchvision==0.8.2
```

**4.** I am seeing zero val metrics in ***wandb***
- Ans: Make sure you threshold the metrics. Since pytorch lightning's first validation check metric is high, it seems like all other metrics are zero. Please threshold manually to remove the outlier metric in wandb to see actual metrics.   

## Acknowledgments
* This code is built upon the implementation from [SimNet](https://github.com/ToyotaResearchInstitute/simnet)

