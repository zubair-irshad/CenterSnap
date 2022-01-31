import numpy as np
import cv2
import torch
import torch.nn as nn

from skimage.feature import peak_local_max
from simnet.lib import transform
from simnet.lib.net import losses

_mask_l1_loss = losses.MaskedL1Loss()
_mse_loss = losses.MSELoss()


class PoseOutput:

  def __init__(self, heatmap, vertex_field, z_centroid_field, hparams):
    self.heatmap = heatmap
    self.vertex_field = vertex_field
    self.z_centroid_field = z_centroid_field
    self.is_numpy = False
    self.hparams = hparams

  # Converters for torch to numpy
  def convert_to_numpy_from_torch(self):
    self.heatmap = np.ascontiguousarray(self.heatmap.cpu().numpy())
    self.vertex_field = np.ascontiguousarray(self.vertex_field.cpu().numpy())
    self.vertex_field = self.vertex_field.transpose((0, 2, 3, 1))
    self.vertex_field = self.vertex_field / 100.0
    self.z_centroid_field = np.ascontiguousarray(self.z_centroid_field.cpu().numpy())
    self.z_centroid_field = self.z_centroid_field / 100.0 + 1.0
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    self.vertex_field = self.vertex_field.transpose((2, 0, 1))
    self.vertex_field = 100.0 * self.vertex_field
    self.vertex_field = torch.from_numpy(np.ascontiguousarray(self.vertex_field)).float()
    self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()
    # Normalize z_centroid by 1.
    self.z_centroid_field = 100.0 * (self.z_centroid_field - 1.0)
    self.z_centroid_field = torch.from_numpy(np.ascontiguousarray(self.z_centroid_field)).float()
    self.is_numpy = False

  def get_visualization_img(self, left_img):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    return draw_pose_from_outputs(
        self.heatmap[0], self.vertex_field[0], self.z_centroid_field[0], left_img
    )

  def compute_loss(self, pose_targets, log):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")
    vertex_target = torch.stack([pose_target.vertex_field for pose_target in pose_targets])
    z_centroid_field_target = torch.stack([
        pose_target.z_centroid_field for pose_target in pose_targets
    ])
    heatmap_target = torch.stack([pose_target.heatmap for pose_target in pose_targets])

    # Move to GPU
    heatmap_target = heatmap_target.to(torch.device('cuda:0'))
    vertex_target = vertex_target.to(torch.device('cuda:0'))
    z_centroid_field_target = z_centroid_field_target.to(torch.device('cuda:0'))

    vertex_loss = _mask_l1_loss(vertex_target, self.vertex_field, heatmap_target)
    log['vertex_loss'] = vertex_loss
    z_centroid_loss = _mask_l1_loss(z_centroid_field_target, self.z_centroid_field, heatmap_target)
    log['z_centroid'] = z_centroid_loss

    heatmap_loss = _mse_loss(heatmap_target, self.heatmap)
    log['heatmap'] = heatmap_loss
    return self.hparams.loss_vertex_mult * vertex_loss + self.hparams.loss_heatmap_mult * heatmap_loss + self.hparams.loss_z_centroid_mult * z_centroid_loss 

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _topk(scores, K=10):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    ind = topk_scores>0.70
    topk_ys = topk_ys[ind.squeeze(0)]
    topk_xs = topk_xs[ind.squeeze(0)]
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def nms_heatmap(heat, kernel=3):
  pad = (kernel - 1) // 2
  hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
  centers_keep = torch.abs(hmax - heat) < 1e-6
  return heat * centers_keep

def extract_peaks_from_centroid_nms(centroid_heatmap, min_distance=5, min_confidence=0.3):
  centroid_heatmap = torch.tensor(centroid_heatmap).unsqueeze(0).unsqueeze(0)
  centroid_heatmap = nms_heatmap(centroid_heatmap)
  topk_scores, topk_inds,_,topk_ys, topk_xs = _topk(centroid_heatmap)
  peaks = np.array([topk_ys.numpy(), topk_xs.numpy()])
  peaks = np.transpose(peaks)
  peaks = find_nearest(peaks,[0,0])
  return peaks, topk_scores, topk_inds, topk_ys, topk_xs

def extract_peaks_from_centroid(centroid_heatmap, min_distance=10, min_confidence=0.20):
  peaks = peak_local_max(centroid_heatmap, min_distance=min_distance, threshold_abs=min_confidence)
  peaks = peaks[peaks[:,1].argsort()]
  return peaks

def find_nearest(peaks,value):
    newList = np.linalg.norm(peaks-value, axis=1)
    return peaks[np.argsort(newList)]

#confidence is 0.3 for CAM and 0.15 for Real
def extract_peaks_from_centroid_sorted(centroid_heatmap,min_confidence=0.15, min_distance=10):
  peaks = peak_local_max(centroid_heatmap, min_distance=min_distance, threshold_abs=min_confidence)
  peaks = find_nearest(peaks,[0,0])
  return peaks

def extract_latent_emb_from_peaks(heatmap_output, peaks, latent_emb_output, scale_factor=8):
  assert peaks.shape[1] == 2
  latent_embeddings = []
  indices = []
  scores = []
  for ii in range(peaks.shape[0]):
    index = np.zeros([2])
    index[0] = int(peaks[ii, 0] / scale_factor)
    index[1] = int(peaks[ii, 1] / scale_factor)
    index = index.astype(np.int)
    latent_emb = latent_emb_output[index[0], index[1], :]
    latent_embeddings.append(latent_emb)
    indices.append(index*scale_factor)
    scores.append(heatmap_output[peaks[ii, 0], peaks[ii, 1]])
  return latent_embeddings, indices, scores

def extract_abs_pose_from_peaks(peaks, abs_pose_output, scale_factor=8):
  assert peaks.shape[1] == 2
  abs_poses = []
  scales = []
  for ii in range(peaks.shape[0]):
    index = np.zeros([2])
    index[0] = int(peaks[ii, 0] / scale_factor)
    index[1] = int(peaks[ii, 1] / scale_factor)
    index = index.astype(np.int)
    abs_pose_values = abs_pose_output[index[0], index[1], :]
    rotation_matrix = np.array([[abs_pose_values[0], abs_pose_values[1], abs_pose_values[2]],
                                [abs_pose_values[3], abs_pose_values[4], abs_pose_values[5]],
                                [abs_pose_values[6], abs_pose_values[7], abs_pose_values[8]]])
    translation_vector = np.array([abs_pose_values[9], abs_pose_values[10], abs_pose_values[11]])
    
    transformation_mat = np.eye(4)
    transformation_mat[:3,:3] = rotation_matrix
    transformation_mat[:3,3] = translation_vector

    scale = abs_pose_values[12]
    scale_matrix = np.eye(4)
    scale_mat = scale*np.eye(3, dtype=float)
    scale_matrix[0:3, 0:3] = scale_mat
    scales.append(scale_matrix)

    abs_poses.append(transform.Pose(
            camera_T_object=transformation_mat, scale_matrix=scale_matrix
        ))
  return abs_poses

def draw_peaks(centroid_target, peaks):
  centroid_target = np.clip(centroid_target, 0.0, 1.0) * 255.0
  heatmap_img = cv2.applyColorMap(centroid_target.astype(np.uint8), cv2.COLORMAP_JET)
  for ii in range(peaks.shape[0]):
    point = (int(peaks[ii, 1]), int(peaks[ii, 0]))
    heatmap_img = cv2.putText(heatmap_img,str(ii), 
    point, 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255,255,255),
    2)
    cv2.line(heatmap_img, (point), ([0,0]), (0, 255, 0), thickness=3, lineType=8)
  return heatmap_img