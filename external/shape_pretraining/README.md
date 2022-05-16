## Shape Autoencoder Pre-training<br>
Shape pretraining code is adapted from [object-deformnet](https://github.com/mentian/object-deformnet).

### Install dependencies

```
conda activate ./env/
cd $CenterSnap_Repo
conda install -c bottler nvidiacub
conda install -c conda-forge -c fvcore -c iopath fvcore iopath
./env/bin/python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0"
```

### Dataset Prepration
1. Download [object models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip) provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019)

2. Download NOCS [preprocess data](https://www.dropbox.com/s/8im9fzopo71h6yw/nocs_preprocess.tar.gz?dl=1)

Unzip and organize these files in $CenterSnap/data as follows:
```
data
├── obj_models
    ├── train
    ├── val
    ├── real_train
    ├── real_test
    ├── mug_meta.pkl
```

2. Prepare data:

```
./runner.sh external/shape_pretraining/shape_data.py --obj_model_dir \path\to\object-model\dir
```
A file would generate in ***obj_models*** folder named ***ShapeNetCore_2048.h5***

3. Train shape auto-encoder:
```
cd external/shape_pretraining
./runner.sh external/shape_pretraining\train_ae.py --h5_file \path\to\h5_file
```
