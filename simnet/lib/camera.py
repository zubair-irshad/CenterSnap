import copy
import numpy as np
from simnet.lib import transform

CLIPPING_PLANE_NEAR = 0.4
SCALE_FACTOR = 4

class NOCS_Camera():
  def __init__(self, height=480, width=640, scale_factor=1.):

    self.RT_matrix = transform.Transform.from_aa(axis=transform.X_AXIS,
                                                 angle_deg=180.0).matrix
    self.height = int(height / scale_factor)
    self.width = int(width / scale_factor)
    self.f_x = 577.5 
    self.f_y = 577.5
    self.c_x = 319.5 
    self.c_y = 239.5
    self.stereo_baseline = 0.119559
    self._set_intrinsics(
        np.array([
            [self.f_x, 0., self.c_x, 0.0],
            [0., self.f_y, self.c_y, 0.0],
            [0., 0., 1., 0.0],
            [0., 0., 0., 1.],
        ]))
  def _set_intrinsics(self, intrinsics_matrix):
    assert intrinsics_matrix.shape[0] == 4
    assert intrinsics_matrix.shape[1] == 4
    self.K_matrix = intrinsics_matrix
    self.proj_matrix = self.K_matrix @ self.RT_matrix

class NOCS_Real():
  def __init__(self, height=480, width=640, scale_factor=1.):
    self.RT_matrix = transform.Transform.from_aa(axis=transform.X_AXIS,
                                                 angle_deg=180.0).matrix
    self.height = int(height / scale_factor)
    self.width = int(width / scale_factor)
    self.f_x = 591.0125
    self.f_y = 590.16775
    self.c_x = 322.525
    self.c_y = 244.11084
    self.stereo_baseline = 0.119559
    self._set_intrinsics(
        np.array([
            [self.f_x, 0., self.c_x, 0.0],
            [0., self.f_y, self.c_y, 0.0],
            [0., 0., 1., 0.0],
            [0., 0., 0., 1.],
        ]))
  def _set_intrinsics(self, intrinsics_matrix):
    assert intrinsics_matrix.shape[0] == 4
    assert intrinsics_matrix.shape[1] == 4
    self.K_matrix = intrinsics_matrix
    self.proj_matrix = self.K_matrix @ self.RT_matrix

def disp_to_depth(disp):
  hfov = np.deg2rad(100.)
  width = 2560.
  fx = 0.5 * width / np.tan(0.5 * hfov)
  b = 100e-3  # 100mm == 10cm
  with np.errstate(divide='ignore'):
    depth = b * fx / disp
  valid = (depth >= 0.01) & (depth <= 100.0) & np.isfinite(depth)
  depth[~valid] = 0.
  return depth


def depth_to_disp(depth, K_matrix, baseline):
  fx = K_matrix[0, 0]
  b = baseline
  valid = (depth > 300e-10) & (depth < 10e5)
  disp = b * fx / depth
  disp[~valid] = 0.
  return disp


def convert_homopixels_to_pixels(pixels):
  """Project 4d homogenous pixels (4xN) to 2d pixels (2xN)"""
  assert len(pixels.shape) == 2
  assert pixels.shape[0] == 4
  pixels_3d = pixels[:3, :] / pixels[3:4, :]
  pixels_2d = pixels_3d[:2, :] / pixels_3d[2:3, :]
  assert pixels_2d.shape[1] == pixels.shape[1]
  assert pixels_2d.shape[0] == 2
  return pixels_2d


def convert_pixels_to_homopixels(pixels, depths):
  """Project 2d pixels (2xN) and depths (meters, 1xN) to 4d pixels (4xN)"""
  assert len(pixels.shape) == 2
  assert pixels.shape[0] == 2
  assert len(depths.shape) == 2
  assert depths.shape[1] == pixels.shape[1]
  assert depths.shape[0] == 1
  pixels_4d = np.concatenate([
      depths * pixels,
      depths,
      np.ones_like(depths),
  ], axis=0)
  assert pixels_4d.shape[0] == 4
  assert pixels_4d.shape[1] == pixels.shape[1]
  return pixels_4d


def convert_points_to_homopoints(points):
  """Project 3d points (3xN) to 4d homogenous points (4xN)"""
  assert len(points.shape) == 2
  assert points.shape[0] == 3
  points_4d = np.concatenate([
      points,
      np.ones((1, points.shape[1])),
  ], axis=0)
  assert points_4d.shape[1] == points.shape[1]
  assert points_4d.shape[0] == 4
  return points_4d


def convert_homopoints_to_points(points_4d):
  """Project 4d homogenous points (4xN) to 3d points (3xN)"""
  assert len(points_4d.shape) == 2
  assert points_4d.shape[0] == 4
  points_3d = points_4d[:3, :] / points_4d[3:4, :]
  assert points_3d.shape[1] == points_3d.shape[1]
  assert points_3d.shape[0] == 3
  return points_3d
