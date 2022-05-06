import os
import sys
import h5py
import glob
import numpy as np
import _pickle as cPickle
import argparse
from utils.shape_utils import sample_points_from_mesh

def save_model_to_hdf5(obj_model_dir, n_points, fps=False, include_distractors=False, with_normal=False):
    """ Save object models (point cloud) to HDF5 file.
        Dataset used to train the auto-encoder.
        Only use models from ShapeNetCore.
        Background objects are not inlcuded as default. We did not observe that it helps
        to train the auto-encoder.
    """
    catId_to_synsetId = {1: '02876657', 2: '02880940', 3: '02942699', 4: '02946921', 5: '03642806', 6: '03797390'}
    distractors_synsetId = ['00000000', '02954340', '02992529', '03211117']
    with open(os.path.join(obj_model_dir, 'mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)
    # read all the paths to models
    print('Sampling points from mesh model ...')
    if with_normal:
        train_data = np.zeros((3000, n_points, 6), dtype=np.float32)
        val_data = np.zeros((500, n_points, 6), dtype=np.float32)
    else:
        train_data = np.zeros((3000, n_points, 3), dtype=np.float32)
        val_data = np.zeros((500, n_points, 3), dtype=np.float32)
    train_label = []
    val_label = []
    train_count = 0
    val_count = 0
    # CAMERA
    for subset in ['train', 'val']:
        for catId in range(1, 7):
            synset_dir = os.path.join(obj_model_dir, subset, catId_to_synsetId[catId])
            inst_list = sorted(os.listdir(synset_dir))
            for instance in inst_list:
                path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                if instance == 'b9be7cfe653740eb7633a2dd89cec754' or instance =='d3b53f56b4a7b3b3c9f016d57db96408':
                    continue
                model_points = sample_points_from_mesh(path_to_mesh_model, n_points, with_normal, fps=fps, ratio=2)
                model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                if catId == 6:
                    shift = mug_meta[instance][0]
                    scale = mug_meta[instance][1]
                    model_points = scale * (model_points + shift)
                if subset == 'train':
                    train_data[train_count] = model_points
                    train_label.append(catId)
                    train_count += 1
                else:
                    val_data[val_count] = model_points
                    val_label.append(catId)
                    val_count += 1
        # distractors
        if include_distractors:
            for synsetId in distractors_synsetId:
                synset_dir = os.path.join(obj_model_dir, subset, synsetId)
                inst_list = sorted(os.listdir(synset_dir))
                for instance in inst_list:
                    path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                    model_points = sample_points_from_mesh(path_to_mesh_model, n_points, with_normal, fps=fps, ratio=2)
                    # TODO: check whether need to flip z-axis, currently not used
                    model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                    if subset == 'train':
                        train_data[train_count] = model_points
                        train_label.append(0)
                        train_count += 1
                    else:
                        val_data[val_count] = model_points
                        val_label.append(0)
                        val_count += 1
    # Real
    for subset in ['real_train', 'real_test']:
        path_to_mesh_models = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in sorted(path_to_mesh_models):
            instance = os.path.basename(inst_path).split('.')[0]
            if instance.startswith('bottle'):
                catId = 1
            elif instance.startswith('bowl'):
                catId = 2
            elif instance.startswith('camera'):
                catId = 3
            elif instance.startswith('can'):
                catId = 4
            elif instance.startswith('laptop'):
                catId = 5
            elif instance.startswith('mug'):
                catId = 6
            else:
                raise NotImplementedError
            model_points = sample_points_from_mesh(inst_path, n_points, with_normal, fps=fps, ratio=2)
            bbox_file = inst_path.replace('.obj', '.txt')
            bbox_dims = np.loadtxt(bbox_file)
            model_points /= np.linalg.norm(bbox_dims)
            if catId == 6:
                shift = mug_meta[instance][0]
                scale = mug_meta[instance][1]
                model_points = scale * (model_points + shift)
            if subset == 'real_train':
                train_data[train_count] = model_points
                train_label.append(catId)
                train_count += 1
            else:
                val_data[val_count] = model_points
                val_label.append(catId)
                val_count += 1

    num_train_instances = len(train_label)
    num_val_instances = len(val_label)
    assert num_train_instances == train_count
    assert num_val_instances == val_count
    train_data = train_data[:num_train_instances]
    val_data = val_data[:num_val_instances]
    train_label = np.array(train_label, dtype=np.uint8)
    val_label = np.array(val_label, dtype=np.uint8)
    print('{} shapes found in train dataset'.format(num_train_instances))
    print('{} shapes found in val dataset'.format(num_val_instances))

    # write to HDF5 file
    print('Writing data to HDF5 file ...')
    if with_normal:
        filename = 'ShapeNetCore_{}_with_normal.h5'.format(n_points)
    else:
        filename = 'ShapeNetCore_{}.h5'.format(n_points)
    hfile = h5py.File(os.path.join(obj_model_dir, filename), 'w')
    train_dataset = hfile.create_group('train')
    train_dataset.attrs.create('len', num_train_instances)
    train_dataset.create_dataset('data', data=train_data, compression='gzip', dtype='float32')
    train_dataset.create_dataset('label', data=train_label, compression='gzip', dtype='uint8')
    val_dataset = hfile.create_group('val')
    val_dataset.attrs.create('len', num_val_instances)
    val_dataset.create_dataset('data', data=val_data, compression='gzip', dtype='float32')
    val_dataset.create_dataset('label', data=val_label, compression='gzip', dtype='uint8')
    hfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model_dir', type=str, required=True)
    args = parser.parse_args()
    obj_model_dir = args.obj_model_dir
    save_model_to_hdf5(obj_model_dir, n_points=2048, fps=True)