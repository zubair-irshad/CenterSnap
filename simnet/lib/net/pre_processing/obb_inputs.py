import numpy as np
from simnet.lib import datapoint
from scipy.stats import multivariate_normal

_HEATMAP_THRESHOLD = 0.3
_DOWNSCALE_VALUE = 8
_PEAK_CONCENTRATION = 0.8

def compute_heatmaps_from_masks(masks):
  heatmaps = [compute_heatmap_from_mask(mask) for mask in masks]
  return heatmaps

def compute_heatmap_from_mask(mask):
  if np.sum(mask) == 0:
    raise ValueError('Mask is empty')
  coords = np.indices(mask.shape)
  coords = coords.reshape([2, -1]).T
  mask_f = mask.flatten()
  indices = coords[np.where(mask_f > 0)]
  mean_value = np.floor(np.average(indices, axis=0))
  cov = np.cov((indices - mean_value).T)
  cov = cov * _PEAK_CONCENTRATION
  multi_var = multivariate_normal(mean=mean_value, cov=cov)
  density = multi_var.pdf(coords)
  heat_map = np.zeros(mask.shape)
  heat_map[coords[:, 0], coords[:, 1]] = density
  return heat_map / np.max(heat_map)

def compute_nocs_network_targets(masks, pointembs, poses, height, width):
  if len(masks) == 0:
    height_d = int(height / _DOWNSCALE_VALUE)
    width_d = int(width / _DOWNSCALE_VALUE)
    return datapoint.ABSPOSE(
        heat_map=np.zeros([height, width]),
        latent_emb=np.zeros([height_d, width_d, 128]),
        abs_pose = np.zeros([height_d, width_d, 13])
    )
  heatmaps = compute_heatmaps_from_masks(masks)
  latent_emb_target = compute_latent_emb(masks, pointembs, heatmaps)
  abs_pose_target = compute_nocs_abspose_field(poses, heatmaps)
  return datapoint.ABSPOSE(
      heat_map=np.max(heatmaps, axis=0),
      latent_emb= latent_emb_target,
      abs_pose = abs_pose_target
  )

def compute_rotation_field(obbs, heat_maps, threshold=0.3):
  cov_target = np.zeros([len(obbs), heat_maps[0].shape[0], heat_maps[0].shape[1], 6])
  heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
  for obb, ii in zip(obbs, range(len(heat_maps))):
    mask = (heatmap_indices == ii)
    cov_matrix = obb.cov_matrix
    cov_mat_values = np.array([
        cov_matrix[0, 0], cov_matrix[1, 1], cov_matrix[2, 2], cov_matrix[0, 1], cov_matrix[0, 2],
        cov_matrix[1, 2]
    ])
    cov_target[ii, mask] = cov_mat_values
  return np.sum(cov_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]

def compute_latent_emb(obbs, point_embs, heat_maps):
  latent_emb_target = np.zeros([len(obbs), heat_maps[0].shape[0], heat_maps[0].shape[1], 128])
  heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
  for emb, ii in zip(point_embs, range(len(heat_maps))):
    mask = (heatmap_indices == ii)
    latent_emb_target[ii, mask] = emb
  sum = np.sum(latent_emb_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]
  return sum

def compute_abspose_field(poses, heat_maps,camera_model, threshold=0.3):
  abs_pose_target = np.zeros([len(poses), heat_maps[0].shape[0], heat_maps[0].shape[1], 13])
  heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
  for pose, ii in zip(poses, range(len(heat_maps))):
    mask = (heatmap_indices == ii)
    actual_abs_pose = camera_model.RT_matrix @ pose.camera_T_object
    rotation_matrix = actual_abs_pose[:3,:3]
    translation_vector = actual_abs_pose[:3,3]
    scale = pose.scale_matrix[0,0]
    abs_pose_values = np.array([
        rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], rotation_matrix[2, 0],
        rotation_matrix[2, 1], rotation_matrix[2, 2], translation_vector[0], translation_vector[1], translation_vector[2], scale
    ])
    abs_pose_target[ii, mask] = abs_pose_values
  return np.sum(abs_pose_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]

def compute_nocs_abspose_field(poses, heat_maps, threshold=0.3):
  abs_pose_target = np.zeros([len(poses), heat_maps[0].shape[0], heat_maps[0].shape[1], 13])
  heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
  for pose, ii in zip(poses, range(len(heat_maps))):
    mask = (heatmap_indices == ii)
    rotation_matrix = pose.camera_T_object[:3,:3]
    translation_vector = pose.camera_T_object[:3,3]
    scale = pose.scale_matrix[0,0]
    abs_pose_values = np.array([
        rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], rotation_matrix[2, 0],
        rotation_matrix[2, 1], rotation_matrix[2, 2], translation_vector[0], translation_vector[1], translation_vector[2], scale
    ])
    abs_pose_target[ii, mask] = abs_pose_values
  return np.sum(abs_pose_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]