#!/opt/mmt/python_venv/bin/python
import random
import copy
import numpy as np
import cv2
import pathlib
import json
import IPython

from simnet.lib import enviroment
from simnet.lib import pose_sampler
from simnet.lib import camera
from simnet.lib import sg
from simnet.lib import label
from simnet.lib import transform
from simnet.lib import datapoint
from simnet.lib import urdf
from simnet.lib import wipeable_primitive

from simnet.lib.tidy_objects import sample_tidy_objects
from simnet.lib.net.pre_processing import obb_inputs
from simnet.lib.net.post_processing import obb_outputs

_DATASET_NAME = 'sample_tidy_objects'
_TEST = True
# Load the semantic information for the tidying classes.
_TIDY_CLASSES = json.load(open(pathlib.Path(__file__).parent / 'tidy_classes.json'))

if _TEST:
  import rich.traceback
  rich.traceback.install()

_DEBUG_FILE_PATH = pathlib.Path(f'data/simnet/{_DATASET_NAME}')
_DEBUG_FILE_PATH.mkdir(parents=True, exist_ok=True)
if _TEST:
  _WRITE_DEBUG = True
  _DATASET = datapoint.make_dataset(f'file://{_DEBUG_FILE_PATH}')
else:
  _WRITE_DEBUG = False
  _DATASET = datapoint.make_dataset(f's3://mmt-learning-data/simnet/output/{_DATASET_NAME}')

# Specifies the camera pose in the scene.
_CAMERA_YAW = [-45, 45]
_CAMERA_PITCH = [-50, -20]

_CAMERA = camera.HSRCamera(scale_factor=1)


# Sampel objects on the table.
def sample_objects(scene_node, default_objects=None):
  surface_nodes = scene_node.get_all_nodes_by_name('wipeable_surface_part')
  if len(surface_nodes) == 0:
    surface_nodes = scene_node.get_all_nodes_by_name('wipeable_surface')
  for surface_node in surface_nodes:
    # Have empty surfaces 10% of the time
    if np.random.uniform() < .1:
      continue
    sample_tidy_objects(scene_node, surface_node, _TIDY_CLASSES)


# Camera angles for validation logs
def camera_sampler(scene_node):

  def sample_camera_radius():
    # sample uniformally over inverse distance
    near, far = 0.7, 1.5
    return 1. / np.random.uniform(1. / far, 1. / near)

  surface_node = scene_node.find_by_name('wipeable_surface')
  lookat_node = surface_node.sample_child_in_bbox(ratio=0.5)
  # Sample a camera az/el angle and radius to get an offset from the point.
  surface_camera_offset = pose_sampler.random_sphere(
      _CAMERA_YAW, _CAMERA_PITCH, radius=sample_camera_radius()
  )

  camera_center_node = lookat_node.add_child(
      sg.Node(name='camera_center').apply_transform(
          transform.Transform.from_aa(translation=surface_camera_offset)
      )
  )

  # Add camera_node and apply rotation such that its pointing at the surface
  camera_node = camera_center_node.add_child(sg.Node(name='camera'))
  camera_node.lookat(lookat_node)
  camera_node.camera = _CAMERA

  # Apply camera roll about optical axis
  max_roll_deg = 30.
  camera_roll_degs = np.random.uniform(-max_roll_deg / 2., max_roll_deg / 2.)
  camera_node.transform.apply_transform(
      transform.Transform.from_aa(axis=transform.Z_AXIS, angle_deg=camera_roll_degs)
  )


env = enviroment.Enviroment()
# For now the wipeable surface scene is has a custom layout that doesn't fit into the API.
# We are currently working on making the scene layout part of the app.
env.set_wipeable_surface_scene()

# Set the graspable objects on the table.
env.add_target_class('small_object', sample_objects)

env.set_camera_sampler(camera_sampler)
env.generate_data()

# Get the labeled data.
left_img, right_img, depth_img = env.get_rendered_data()
seg_mask = env.get_segmentation_mask()
obbs, instance_masks = env.get_oriented_bounding_boxes('small_object')

# Enforce that the camera is facing the generated table.
h, w = seg_mask.shape
table_value = 2  # Wipeable Surface Value.
object_value = 3
assert (seg_mask[h // 2, w // 2] == table_value) or (seg_mask[h // 2, w // 2] == object_value)
obb_datapoint = obb_inputs.compute_network_targets(
    obbs, instance_masks, _CAMERA.height, _CAMERA.width, _CAMERA
)

stereo_datapoint = datapoint.Stereo(left_color=left_img, right_color=right_img)
panoptic_datapoint = datapoint.Panoptic(
    stereo=stereo_datapoint,
    depth=depth_img,
    segmentation=seg_mask,
    object_poses=[obb_datapoint],
    boxes=[],
    detections=[]
)

_DATASET.write(panoptic_datapoint)
if _TEST:
  uid = panoptic_datapoint.uid

  cv2.imwrite(
      str(_DEBUG_FILE_PATH / f'{uid}_seg.png'),
      label.draw_pixelwise_mask(np.copy(left_img), seg_mask)
  )
  cv2.imwrite(
      str(_DEBUG_FILE_PATH / f'{uid}_net_obb.png'),
      obb_outputs.draw_oriented_bounding_box_from_outputs(
          obb_datapoint.heat_map,
          obb_datapoint.vertex_target,
          obb_datapoint.cov_matrices,
          obb_datapoint.z_centroid,
          np.copy(left_img),
          camera_model=_CAMERA
      )
  )
