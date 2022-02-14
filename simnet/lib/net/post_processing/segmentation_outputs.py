import numpy as np
import cv2
import IPython
import torch
import torch.nn as nn
from torch.nn import functional as F

from simnet.lib import color_stuff

# Panoptic Segmentation Colors


class SegmentationOutput:

  def __init__(self, seg_pred, hparams):
    self.seg_pred = seg_pred
    self.is_numpy = False
    self.hparams = hparams

  # Converters for torch to numpy
  def convert_to_numpy_from_torch(self):
    self.seg_pred = np.ascontiguousarray(self.seg_pred.cpu().numpy())
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    self.seg_pred = torch.from_numpy(np.ascontiguousarray(self.seg_pred)).long()
    self.is_numpy = False

  def get_visualization_img(self, left_image):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    return draw_segmentation_mask(left_image, self.seg_pred[0])

  def get_prediction(self):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    return self.seg_pred[0]

  def compute_loss(self, seg_targets, log, name):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")
    seg_target_stacked = []
    for seg_target in seg_targets:
      seg_target_stacked.append(seg_target.seg_pred)
    seg_target_batch = torch.stack(seg_target_stacked)
    seg_target_batch = seg_target_batch.to(torch.device('cuda:0'))
    seg_loss = F.cross_entropy(self.seg_pred, seg_target_batch, reduction="mean", ignore_index=-100)
    # log['segmentation'] = seg_loss
    log[name] = seg_loss.item()
    return self.hparams.loss_seg_mult * seg_loss


def draw_segmentation_mask_gt(color_img, seg_mask, num_classes=5):
  assert len(seg_mask.shape) == 2
  seg_mask = seg_mask.astype(np.uint8)
  colors = color_stuff.get_panoptic_colors()
  color_img = color_img_to_gray(color_img)
  for ii, color in zip(range(num_classes), colors):
    colored_mask = np.zeros([seg_mask.shape[0], seg_mask.shape[1], 3])
    colored_mask[seg_mask == ii, :] = color
    color_img = cv2.addWeighted(
        color_img.astype(np.uint8), 0.9, colored_mask.astype(np.uint8), 0.4, 0
    )
  return cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

def color_img_to_gray(image):
  gray_scale_img = np.zeros(image.shape)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for i in range(3):
    gray_scale_img[:, :, i] = img
  gray_scale_img[:, :, i] = img
  return gray_scale_img

def draw_segmentation_mask(color_img, seg_mask):
  assert len(seg_mask.shape) == 3
  num_classes = seg_mask.shape[0]
  # Convert to predictions
  seg_mask_predictions = np.argmax(seg_mask, axis=0)
  return draw_segmentation_mask_gt(color_img, seg_mask_predictions, num_classes=num_classes)
