# Copyright 2019 Toyota Research Institute.  All rights reserved.

import os
import random
import pathlib
import cv2
import numpy as np
import torch
import IPython
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from simnet.lib import datapoint
from simnet.lib.net.post_processing.segmentation_outputs import SegmentationOutput
from simnet.lib.net.post_processing.depth_outputs import DepthOutput
from simnet.lib.net.post_processing.abs_pose_outputs import OBBOutput

def extract_left_numpy_img(anaglyph):
  anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
  anaglyph_np = anaglyph_np.transpose((1, 2, 0))
  left_img = anaglyph_np[..., 0:3] * 255.0
  return left_img

def extract_right_numpy_img(anaglyph):
  anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
  anaglyph_np = anaglyph_np.transpose((1, 2, 0))
  left_img = anaglyph_np[..., 3:6] * 255.0
  return left_img

def create_anaglyph(stereo_dp):
  height, width, _ = stereo_dp.left_color.shape
  image = torch.zeros(4, height, width, dtype=torch.float32)
  cv2.normalize(stereo_dp.left_color, stereo_dp.left_color, 0, 255, cv2.NORM_MINMAX)

  rgb = stereo_dp.left_color* 1. / 255.0
  norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  rgb = norm(torch.from_numpy(rgb.astype(np.float32).transpose((2,0,1))))

  if len(stereo_dp.right_color.shape) == 2:
    depth = stereo_dp.right_color
    depth = torch.from_numpy(depth.astype(np.float32))

  image[0:3, :] = rgb
  image[3, :] = depth
  return image

class Dataset(Dataset):
  def __init__(self, dataset_uri, hparams, preprocess_image_func=None, datapoint_dataset=None):
    super().__init__()
    if datapoint_dataset is None:
      datapoint_dataset = datapoint.make_dataset(dataset_uri)
    self.datapoint_handles = datapoint_dataset.list()
    # No need to shuffle, already shufled based on random uids
    self.hparams = hparams
    if preprocess_image_func is None:
      self.preprocces_image_func = create_anaglyph
    else:
      self.preprocces_image_func = preprocess_image_func

  def __len__(self):
    return len(self.datapoint_handles)

  def __getitem__(self, idx):
    dp = self.datapoint_handles[idx].read()
    anaglyph = self.preprocces_image_func(dp.stereo)
    segmentation_target = SegmentationOutput(dp.segmentation, self.hparams)
    segmentation_target.convert_to_torch_from_numpy()
    depth_target = DepthOutput(dp.depth, self.hparams)
    depth_target.convert_to_torch_from_numpy()
    pose_target = None
    for pose_dp in dp.object_poses:
      pose_target = OBBOutput(
          pose_dp.heat_map, pose_dp.latent_emb, pose_dp.abs_pose, self.hparams
      )
      pose_target.convert_to_torch_from_numpy()
    scene_name = dp.scene_name
    return anaglyph, segmentation_target, depth_target, pose_target, dp.detections, scene_name
