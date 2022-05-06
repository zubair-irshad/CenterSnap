import h5py
import numpy as np
import torch.utils.data as data

class ShapeDataset(data.Dataset):
    def __init__(self, h5_file, mode, n_points=1024, augment=False):
        assert (mode == 'train' or mode == 'val'), 'Mode must be "train" or "val".'
        self.mode = mode
        self.n_points = n_points
        self.augment = augment
        # load data from h5py file
        with h5py.File(h5_file, 'r') as f:
            self.length = f[self.mode].attrs['len']
            self.data = f[self.mode]['data'][:]
            self.label = f[self.mode]['label'][:]
        # augmentation parameters
        self.sigma = 0.01
        self.clip = 0.02
        self.shift_range = 0.02

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        xyz = self.data[index]
        label = self.label[index] - 1    # data saved indexed from 1
        # randomly downsample
        np_data = xyz.shape[0]
        assert np_data >= self.n_points, 'Not enough points in shape.'
        idx = np.random.choice(np_data, self.n_points)
        xyz = xyz[idx, :]
        # data augmentation
        if self.augment:
            jitter = np.clip(self.sigma*np.random.randn(self.n_points, 3), -self.clip, self.clip)
            xyz[:, :3] += jitter
            shift = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            xyz[:, :3] += shift
        return xyz, label