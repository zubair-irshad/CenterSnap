import numpy as np
import cv2
import IPython
import torch
import torch.nn as nn
from simnet.lib.net import losses

_mask_l1_loss = losses.MaskedL1Loss(downscale_factor=1)
_mse_loss = losses.MaskedMSELoss()


class BoxOutput:

  def __init__(self, heatmap, vertex_field, hparams, ignore_mask=None):
    self.heatmap = heatmap
    self.vertex_field = vertex_field
    self.ignore_mask = ignore_mask
    self.is_numpy = False
    self.hparams = hparams

  # Converters for torch to numpy
  def convert_to_numpy_from_torch(self):
    self.heatmap = np.ascontiguousarray(self.heatmap.cpu().numpy())
    self.vertex_field = np.ascontiguousarray(self.vertex_field.cpu().numpy())
    self.vertex_field = self.vertex_field.transpose((0, 2, 3, 1))
    self.vertex_field = self.vertex_field / 100.0
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    self.vertex_field = self.vertex_field.transpose((2, 0, 1))
    self.vertex_field = 100.0 * self.vertex_field
    self.vertex_field = torch.from_numpy(np.ascontiguousarray(self.vertex_field)).float()
    self.ignore_mask = torch.from_numpy(np.ascontiguousarray(self.ignore_mask)).bool()
    self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()
    self.is_numpy = False

  def compute_loss(self, pose_targets, log):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")
    vertex_target = torch.stack([pose_target.vertex_field for pose_target in pose_targets])
    heatmap_target = torch.stack([pose_target.heatmap for pose_target in pose_targets])
    ignore_target = torch.stack([pose_target.ignore_mask for pose_target in pose_targets])

    # Move to GPU
    heatmap_target = heatmap_target.to(torch.device('cuda:0'))
    vertex_target = vertex_target.to(torch.device('cuda:0'))
    ignore_target = ignore_target.to(torch.device('cuda:0'))

    vertex_loss = _mask_l1_loss(vertex_target, self.vertex_field, heatmap_target)
    log['vertex_loss'] = vertex_loss
    heatmap_loss = _mse_loss(self.heatmap, heatmap_target, ignore_target)
    log['heatmap'] = heatmap_loss
    return self.hparams.loss_vertex_mult * vertex_loss + self.hparams.loss_heatmap_mult * heatmap_loss

def extract_vertices_from_peaks(peaks, vertex_fields, c_img, scale_factor=1):
  assert peaks.shape[1] == 2
  assert vertex_fields.shape[2] == 4
  height = vertex_fields.shape[0] * scale_factor
  width = vertex_fields.shape[1] * scale_factor
  vertex_fields[:, :, ::2] = (1.0 - vertex_fields[:, :, ::2]) * (2 * height) - height
  vertex_fields[:, :, 1::2] = (1.0 - vertex_fields[:, :, 1::2]) * (2 * width) - width
  bboxes = []
  for ii in range(peaks.shape[0]):
    bbox = get_bbox_from_vertex(vertex_fields, peaks[ii, :], scale_factor=scale_factor)
    bboxes.append(bbox)
  return bboxes


def get_bbox_from_vertex(vertex_fields, index, scale_factor=64):
  assert index.shape[0] == 2
  index[0] = int(index[0] / scale_factor)
  index[1] = int(index[1] / scale_factor)
  bbox = vertex_fields[index[0], index[1], :]
  bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
  bbox = scale_factor * (index) - bbox
  return bbox


def draw_2d_boxes_with_colors(img, bboxes, colors):
  for bbox, color in zip(bboxes, colors):
    pt1 = (int(bbox[0][1]), int(bbox[0][0]))
    pt2 = (int(bbox[1][1]), int(bbox[1][0]))
    img = cv2.rectangle(img, pt1, pt2, color, 2)
  return img


def draw_2d_boxes(img, bboxes):
  for bbox in bboxes:
    pt1 = (int(bbox[0][1]), int(bbox[0][0]))
    pt2 = (int(bbox[1][1]), int(bbox[1][0]))
    img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
  return img
