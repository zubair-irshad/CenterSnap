import math
import numpy as np
from simnet.lib import camera
import torch

def align_rotation(R):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    return rotation


def get_scaled_pc(scale_matrix, pc):
  pc_homopoints = camera.convert_points_to_homopoints(pc.T)
  scaled_homopoints = (scale_matrix @ pc_homopoints)
  scaled_homopoints = camera.convert_homopoints_to_points(scaled_homopoints).T
  return scaled_homopoints


def project(K, p_3d):
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
    return projections_2d


def get_gt_pointclouds(pose, pc, camera_model=None, sizes = None
):
  if sizes is not None:
    pc_homopoints = camera.convert_points_to_homopoints(pc.T)
    morphed_pc_homopoints = pose @ pc_homopoints
    morphed_pc_homopoints = camera.convert_homopoints_to_points(morphed_pc_homopoints).T
  else:
    pc_homopoints = camera.convert_points_to_homopoints(pc.T)
    morphed_pc_homopoints = pose.camera_T_object @ (pose.scale_matrix @ pc_homopoints)
    morphed_pc_homopoints = camera.convert_homopoints_to_points(morphed_pc_homopoints).T

  if sizes is not None:
    size = sizes
    box = get_3d_bbox(size)
    unit_box_homopoints = camera.convert_points_to_homopoints(box.T)
    morphed_box_homopoints = pose @ unit_box_homopoints
    morphed_box_points = camera.convert_homopoints_to_points(morphed_box_homopoints).T
  else:
    pc_hp = camera.convert_points_to_homopoints(pc.T)
    scaled_homopoints = (pose.scale_matrix @ pc_hp)
    scaled_homopoints = camera.convert_homopoints_to_points(scaled_homopoints).T
    size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
    box = get_3d_bbox(size)
    unit_box_homopoints = camera.convert_points_to_homopoints(box.T)
    morphed_box_homopoints = pose.camera_T_object @ unit_box_homopoints
    morphed_box_points = camera.convert_homopoints_to_points(morphed_box_homopoints).T
  return morphed_pc_homopoints, morphed_box_points, size

def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                    [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                    [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                    [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                    [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    return bbox_3d


def get_2d_box(pose,pc, camera_model):
    unit_box_homopoints = camera.convert_points_to_homopoints(pc.T)
    morphed_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
    morphed_pixels = camera.convert_homopixels_to_pixels(camera_model.K_matrix @ morphed_homopoints).T
    bbox = [
        np.array([np.min(morphed_pixels[:, 0]),
                  np.min(morphed_pixels[:, 1])]),
        np.array([np.max(morphed_pixels[:, 0]),
                  np.max(morphed_pixels[:, 1])])
    ]
    return bbox

def rgbd_size(pc):
    size = 2 * np.amax(np.abs(pc), axis=0)
    return size


def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def rgbd_size(pc):
    size = 2 * np.amax(np.abs(pc), axis=0)
    return size



def compute_sRT_errors(s1, R1, T1, s2, R2, T2):
    """
    Args:
        sRT1: [4, 4]. homogeneous affine transformation
        sRT2: [4, 4]. homogeneous affine transformation

    Returns:
        R_error: angle difference in degree,
        T_error: Euclidean distance
        IoU: relative scale error

    """
    R12 = R1 @ R2.transpose()
    R_error = np.arccos(np.clip((np.trace(R12)-1)/2, -1.0, 1.0)) * 180 / np.pi
    T_error = np.linalg.norm(T1 - T2)
    IoU = np.abs(s1 - s2) / s2
    return R_error, T_error, IoU

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.ones(batch) )
    cos = torch.max(cos, torch.ones(batch)*-1 )
    
    
    theta = torch.acos(cos)
    #theta = torch.min(theta, 2*np.pi - theta)
    return theta

def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have size [batch_size, 9]

  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  u, s, v = torch.svd(m)
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r