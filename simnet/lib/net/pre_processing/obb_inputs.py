import numpy as np
from scipy.stats import multivariate_normal

from simnet.lib.net.post_processing import epnp
from simnet.lib.net.pre_processing import pose_inputs
from simnet.lib import datapoint

import time
_HEATMAP_THRESHOLD = 0.3
_DOWNSCALE_VALUE = 8
_PEAK_CONCENTRATION = 0.8

# def compute_network_targets(obbs, masks, height, width, camera_model):
#   assert len(obbs) == len(masks)
#   if len(obbs) == 0:
#     height_d = int(height / _DOWNSCALE_VALUE)
#     width_d = int(width / _DOWNSCALE_VALUE)
#     return datapoint.OBB(
#         heat_map=np.zeros([height, width]),
#         vertex_target=np.zeros([height_d, width_d, 16]),
#         cov_matrices=np.zeros([height_d, width_d, 6]),
#         z_centroid=np.zeros([height_d, width_d])
#     )
#   heatmaps = pose_inputs.compute_heatmaps_from_masks(masks)
#   vertex_target = pose_inputs.compute_vertex_field(obbs, heatmaps, camera_model)
#   z_centroid = pose_inputs.compute_z_centroid_field(obbs, heatmaps)
#   cov_matrix = compute_rotation_field(obbs, heatmaps)
#   return datapoint.OBB(
#       heat_map=np.max(heatmaps, axis=0),
#       vertex_target=vertex_target,
#       cov_matrices=cov_matrix,
#       z_centroid=z_centroid
#   )

def compute_network_targets(obbs, masks, pointembs, poses, height, width, camera_model):
  assert len(obbs) == len(masks)
  if len(obbs) == 0:
    height_d = int(height / _DOWNSCALE_VALUE)
    width_d = int(width / _DOWNSCALE_VALUE)
    return datapoint.ABSPOSE(
        heat_map=np.zeros([height, width]),
        latent_emb=np.zeros([height_d, width_d, 128]),
        abs_pose = np.zeros([height_d, width_d, 13])
    )
  heatmaps = pose_inputs.compute_heatmaps_from_masks(masks)
  latent_emb_target = compute_latent_emb(obbs, pointembs, heatmaps)
  abs_pose_target = compute_abspose_field(poses, heatmaps, camera_model)
  return datapoint.ABSPOSE(
      heat_map=np.max(heatmaps, axis=0),
      latent_emb= latent_emb_target,
      abs_pose = abs_pose_target
  )

def compute_nocs_network_targets(masks, pointembs, poses, height, width, camera_model):
  if len(masks) == 0:
    height_d = int(height / _DOWNSCALE_VALUE)
    width_d = int(width / _DOWNSCALE_VALUE)
    return datapoint.ABSPOSE(
        heat_map=np.zeros([height, width]),
        latent_emb=np.zeros([height_d, width_d, 128]),
        abs_pose = np.zeros([height_d, width_d, 13])
    )
  s_time = time.time()
  heatmaps = pose_inputs.compute_heatmaps_from_masks(masks)
  latent_emb_target = compute_latent_emb(masks, pointembs, heatmaps)
  abs_pose_target = compute_abspose_field(poses, heatmaps, camera_model)
  return datapoint.ABSPOSE(
      heat_map=np.max(heatmaps, axis=0),
      latent_emb= latent_emb_target,
      abs_pose = abs_pose_target
  )

# def compute_nocs_network_targets(masks, pointembs, poses, height, width):
#   if len(masks) == 0:
#     height_d = int(height / _DOWNSCALE_VALUE)
#     width_d = int(width / _DOWNSCALE_VALUE)
#     return datapoint.ABSPOSE(
#         heat_map=np.zeros([height, width]),
#         latent_emb=np.zeros([height_d, width_d, 128]),
#         abs_pose = np.zeros([height_d, width_d, 13])
#     )
#   s_time = time.time()
#   heatmaps = pose_inputs.compute_heatmaps_from_masks(masks)
#   latent_emb_target = compute_latent_emb(masks, pointembs, heatmaps)
#   abs_pose_target = compute_nocs_abspose_field(poses, heatmaps)
#   return datapoint.ABSPOSE(
#       heat_map=np.max(heatmaps, axis=0),
#       latent_emb= latent_emb_target,
#       abs_pose = abs_pose_target
#   )

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

  lat_time = time.time()
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

# import numpy as np
# from scipy.stats import multivariate_normal

# from simnet.lib.net.post_processing import epnp
# from simnet.lib.net.pre_processing import pose_inputs
# from simnet.lib import datapoint

# _HEATMAP_THRESHOLD = 0.3
# _DOWNSCALE_VALUE = 8
# _PEAK_CONCENTRATION = 0.8


# def compute_network_targets(obbs, masks, height, width, camera_model):
#   assert len(obbs) == len(masks)
#   if len(obbs) == 0:
#     height_d = int(height / _DOWNSCALE_VALUE)
#     width_d = int(width / _DOWNSCALE_VALUE)
#     return datapoint.OBB(
#         heat_map=np.zeros([height, width]),
#         vertex_target=np.zeros([height_d, width_d, 16]),
#         cov_matrices=np.zeros([height_d, width_d, 6]),
#         z_centroid=np.zeros([height_d, width_d])
#     )
#   heatmaps = pose_inputs.compute_heatmaps_from_masks(masks)
#   vertex_target = pose_inputs.compute_vertex_field(obbs, heatmaps, camera_model)
#   z_centroid = pose_inputs.compute_z_centroid_field(obbs, heatmaps)
#   cov_matrix = compute_rotation_field(obbs, heatmaps)
#   return datapoint.OBB(
#       heat_map=np.max(heatmaps, axis=0),
#       vertex_target=vertex_target,
#       cov_matrices=cov_matrix,
#       z_centroid=z_centroid
#   )


# def compute_rotation_field(obbs, heat_maps, threshold=0.3):
#   cov_target = np.zeros([len(obbs), heat_maps[0].shape[0], heat_maps[0].shape[1], 6])
#   heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
#   for obb, heat_map, ii in zip(obbs, heat_maps, range(len(heat_maps))):
#     mask = (heatmap_indices == ii)
#     cov_matrix = obb.cov_matrix
#     cov_mat_values = np.array([
#         cov_matrix[0, 0], cov_matrix[1, 1], cov_matrix[2, 2], cov_matrix[0, 1], cov_matrix[0, 2],
#         cov_matrix[1, 2]
#     ])
#     cov_target[ii, mask] = cov_mat_values
#   return np.sum(cov_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]
