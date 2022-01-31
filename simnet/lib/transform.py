import dataclasses

import numpy as np

X_AXIS = np.array([1., 0., 0.])
Y_AXIS = np.array([0., 1., 0.])
Z_AXIS = np.array([0., 0., 1.])


@dataclasses.dataclass
class Pose:
  camera_T_object: np.ndarray
  scale_matrix: np.ndarray = np.eye(4)


class Transform:

  def __init__(self, matrix=None):
    if matrix is None:
      self.matrix = np.eye(4)
    else:
      self.matrix = matrix
    self.is_concrete = True

  def apply_transform(self, transform):
    assert self.is_concrete
    assert isinstance(transform, Transform)
    self.matrix = self.matrix @ transform.matrix

  def inverse(self):
    assert self.is_concrete
    return Transform(matrix=np.linalg.inv(self.matrix))

  def __repr__(self):
    assert self.matrix.shape == (4, 4)
    if self.is_SE3():
      return f'Transform(translate={self.translation})'
    else:
      return f'Transform(IS_NOT_SE3,matrix={self.matrix})'

  def is_SE3(self):
    return matrixIsSE3(self.matrix)

  @property
  def translation(self):
    return self.matrix[:3, 3]

  @translation.setter
  def translation(self, value):
    assert value.shape == (3,)
    self.matrix[:3, 3] = value

  @property
  def rotation(self):
    return self.matrix[:3, :3]

  @rotation.setter
  def rotation(self, value):
    assert value.shape == (3, 3)
    self.matrix[:3, :3] = value

  @classmethod
  def from_aa(cls, axis=X_AXIS, angle_deg=0., translation=None):
    assert axis.shape == (3,)
    matrix = np.eye(4)
    if angle_deg != 0.:
      matrix[:3, :3] = axis_angle_to_rotation_matrix(axis, np.deg2rad(angle_deg))
    if translation is not None:
      translation = np.array(translation)
      assert translation.shape == (3,)
      matrix[:3, 3] = translation
    return cls(matrix=matrix)


def matrixIsSE3(matrix):
  if not np.allclose(matrix[3, :], np.array([0., 0., 0., 1.])):
    return False
  rot = matrix[:3, :3]
  if not np.allclose(rot @ rot.T, np.eye(3)):
    return False
  if not np.isclose(np.linalg.det(rot), 1.):
    return False
  return True


def find_closest_SE3(matrix):
  matrix = np.copy(matrix)
  assert np.allclose(matrix[3, :], np.array([0., 0., 0., 1.]))
  rotation = matrix[:3, :3]
  u, s, vh = np.linalg.svd(rotation)
  matrix[:3, :3] = u @ vh
  assert matrixIsSE3(matrix)
  return matrix


def axis_angle_to_rotation_matrix(axis, theta):
  """Return the rotation matrix associated with counterclockwise rotation about
  the given axis by theta radians.
  Args:
      axis: a list which specifies a unit axis
      theta: an angle in radians, for which to rotate around by
  Returns:
      A 3x3 rotation matrix
  """
  axis = np.asarray(axis)
  axis = axis / np.sqrt(np.dot(axis, axis))
  a = np.cos(theta / 2.0)
  b, c, d = -axis * np.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                   [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                   [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])