#!/opt/mmt/python_venv/bin/python
import random
import copy
import numpy as np
import cv2
import pathlib
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
import torch
# import open3d as o3d
import json
from simnet.lib.net.post_processing.epnp import _WORLD_T_POINTS
import matplotlib.pyplot as plt

from simnet.lib.graspable_objects import sample_graspable_objects
from simnet.lib.net.pre_processing import obb_inputs
from simnet.lib.net.post_processing import obb_outputs
from simnet.lib.depth_noise import DepthManager

from simnet.lib.net.models.auto_encoder import PointCloudAE
from simnet.lib.tidy_objects import sample_tidy_objects

_DATASET_NAME = 'zubair_panoptic_rgbd_zed2_val'
_TEST = True
_TIDY_CLASSES = json.load(open(pathlib.Path(__file__).parent / 'ycb_classes.json'))

def get_gt_pointclouds(pose, pc, camera_model=None, use_camera_RT=True
):
  # pose.camera_T_object[:3,3] = pose.camera_T_object[:3,3]*100
  # pose.scale_matrix[:3,:3] = pose.scale_matrix[:3,:3]*100
  # for pose, pc in zip(poses, pointclouds):
  pc_homopoints = camera.convert_points_to_homopoints(pc.T)
  unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)
  if use_camera_RT == True:
    morphed_pc_homopoints = _CAMERA.RT_matrix @ pose.camera_T_object @ (pose.scale_matrix @ pc_homopoints)
    morphed_box_homopoints = _CAMERA.RT_matrix @ pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
  else:
    morphed_pc_homopoints = pose.camera_T_object @ (pose.scale_matrix @ pc_homopoints)
    morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
  
  morphed_pc_homopoints = camera.convert_homopoints_to_points(morphed_pc_homopoints).T
  morphed_box_points = camera.convert_homopoints_to_points(morphed_box_homopoints).T
  # box_points.append(morphed_box_points)
  return morphed_pc_homopoints, morphed_box_points

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

_ROBOT_MASK_NAME = random.choice(['fmk_gripper_wipe_tri'])
_CAMERA = camera.ZED2Camera()
# For shipping to FMK this is useful to have on.
_TURN_ON_ROBOT = False
_ANGLES_PATH = pathlib.Path(__file__).parent / 'fmk_joint_angles.txt'


# Sample FMK
# Specifies the way target objects are placed in the scene and returns the sceen node.
def fmk_sampler(scene_node, target_nodes):
  # Only one target node for robot mask, so we assume its the first one.
  # If more target nodes are added you need to loop through and check by name.
  fmk_node = target_nodes[0]
  #Sample fmk joint angles.
  joint_names, angles = urdf.load_joint_samples(angles_path=str(_ANGLES_PATH))
  urdf.assign_joint_samples(fmk_node, joint_names, random.choice(angles))

  bbox = scene_node.concatenated_mesh.bounds
  mid_ground_point = np.array([(bbox[1][0] - bbox[0][0]) / 2.0 + bbox[0][0], bbox[0][1],
                               bbox[1][2]])

  # Sample z offset
  z_offset = np.random.uniform(0.4, 0.6)
  x_offset = np.random.uniform(-0.2, 0.2)
  y_offset = np.random.uniform(0.0, 0.2)
  base_offset = np.array([x_offset, y_offset, z_offset])

  rot_90_x = transform.Transform().from_aa(angle_deg=-90).matrix
  yaw_offset = np.random.uniform(-10, 10)
  rot_90_y = transform.Transform().from_aa(axis=transform.Y_AXIS, angle_deg=90 + yaw_offset).matrix
  root_T_robot_base = np.eye(4) @ (rot_90_y @ rot_90_x)
  root_T_robot_base[0:3, 3] = mid_ground_point + base_offset
  fmk_node.apply_transform(transform.Transform(matrix=root_T_robot_base))
  scene_node.add_child(fmk_node)


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


# Camera angles for FMK.
def fmk_camera_sampler(scene_node):
  fmk_node = scene_node.find_by_name(_ROBOT_MASK_NAME)
  fmk_camera_node = fmk_node.find_by_name("sensor_head_basler_left_rgb")
  fmk_T_camera = fmk_camera_node.get_transform_matrix_to_ancestor(fmk_node)

  # Compute transforms to place pyrender camera on FMK's postion
  fmk_to_camera = sg.Node()
  fmk_camera_node = fmk_node.find_by_name("sensor_head_basler_left_rgb")
  fmk_T_camera = fmk_camera_node.get_transform_matrix_to_ancestor(fmk_node)
  rot_180_y = transform.Transform().from_aa(axis=transform.Y_AXIS, angle_deg=180).matrix
  rot_180_z = transform.Transform().from_aa(axis=transform.Z_AXIS, angle_deg=180).matrix
  fmk_T_camera = fmk_T_camera @ (rot_180_z @ rot_180_y)
  fmk_to_camera.transform = transform.Transform(matrix=fmk_T_camera)

  fmk_node.apply_transforms_to_meshes()
  sensor_head_node = fmk_node.find_by_name("sensor_head_mount")
  fmk_node.concat_into_node()
  camera_node = sg.Node(name='camera')
  camera_node.camera = camera.Camera()
  fmk_node.add_child(fmk_to_camera)
  fmk_to_camera.add_child(camera_node)


env = enviroment.Enviroment()
# For now the wipeable surface scene is has a custom layout that doesn't fit into the API.
# We are currently working on making the scene layout part of the app.
env.set_wipeable_surface_scene()

# Set the graspable objects on the table.
env.add_target_class('small_object', sample_objects)

if _TURN_ON_ROBOT and np.random.uniform() > 0.75:
  env.add_target_class_from_urdf(_ROBOT_MASK_NAME, target_sampler=fmk_sampler)
  env.set_camera_sampler(fmk_camera_sampler)
else:
  env.set_camera_sampler(camera_sampler)

env.generate_data()
# Get the labeled data.
left_img, right_img, depth_img = env.get_rendered_data()
seg_mask = env.get_segmentation_mask()
obbs, instance_masks = env.get_oriented_bounding_boxes('small_object')

point_clouds = env.get_pointclouds('small_object')
poses = env.get_poses('small_object')

emb_dim = 128
n_cat = 57
n_pts = 1024
model_path = str(pathlib.Path(__file__).parent.parent.parent / 'data' / 'auto_encoder_model' / 'model_50.pth')
estimator = PointCloudAE(emb_dim, n_pts)
estimator.cuda()
estimator.load_state_dict(torch.load(model_path))
estimator.eval()

latent_embeddings = []

for i in range(len(point_clouds)):
  batch_xyz = torch.from_numpy(point_clouds[i]).to(device="cuda", dtype=torch.float)
  batch_xyz = batch_xyz.unsqueeze(0)
  # print(batch_xyz.shape)
  emb, pred_points = estimator(batch_xyz)
  # print("emb before cpu", emb.size())
  # emb = emb.unsqueeze(0)
  emb = emb.squeeze(0).cpu().detach().numpy()
  latent_embeddings.append(emb)

# Enforce that the camera is facing the generated table.
h, w = seg_mask.shape
table_value = 2  # Wipeable Surface Value.
object_value = 3
assert (seg_mask[h // 2, w // 2] == table_value) or (seg_mask[h // 2, w // 2] == object_value)
obb_datapoint = obb_inputs.compute_network_targets(
    obbs, instance_masks,latent_embeddings, poses, _CAMERA.height, _CAMERA.width, _CAMERA
)

import sys
import numpy

DM = DepthManager()
noisy_depth  = DM.prepare_depth_data(depth_img)

print(depth_img.shape)
print(depth_img)
print(depth_img.dtype)
plt.imshow(depth_img)
plt.axis('off')
plt.grid(b=None)
plt.show()
plt.imshow(noisy_depth)
plt.axis('off')
plt.grid(b=None)
plt.show()

print("\n")
print(depth_img.shape)
print(depth_img)
print(depth_img.dtype)

detections_gt = obb_outputs.get_gt_detections(
          obb_datapoint.heat_map,
          obb_datapoint.vertex_target,
          obb_datapoint.cov_matrices,
          obb_datapoint.z_centroid,
          camera_model=_CAMERA
      )


# save noisy depth instead of right color image

print("left img", left_img)
print("left image type", left_img.dtype)

cv2.normalize(left_img, left_img, 0, 255, cv2.NORM_MINMAX)
left_img = left_img * 1. / 255.0

print("left img updated", left_img)
print("left image updated type", left_img.dtype)

stereo_datapoint = datapoint.Stereo(left_color=left_img, right_color=noisy_depth)
panoptic_datapoint = datapoint.Panoptic(
    stereo=stereo_datapoint,
    depth=depth_img,
    segmentation=seg_mask,
    object_poses=[obb_datapoint],
    boxes=[],
    detections=[detections_gt]
)


_DATASET.write(panoptic_datapoint)
if _TEST:
  uid = panoptic_datapoint.uid

  cv2.imwrite(
      str(_DEBUG_FILE_PATH / f'{uid}_seg.png'),
      label.draw_pixelwise_mask(np.copy(left_img), seg_mask)
  )

  depth_image = cv2.applyColorMap(noisy_depth, cv2.COLORMAP_JET)
  cv2.imwrite(
      str(_DEBUG_FILE_PATH / f'{uid}_depth.png'),
      np.copy(depth_image)
  )

  cv2.imwrite(
      str(_DEBUG_FILE_PATH / f'{uid}_net_obb.png'),
      obb_outputs.draw_oriented_bounding_box_from_outputs(
          obb_datapoint.heat_map, obb_datapoint.vertex_target, obb_datapoint.cov_matrices,
          obb_datapoint.z_centroid, np.copy(left_img), camera_model=_CAMERA
      )
  )

  cv2.imwrite(
      str(_DEBUG_FILE_PATH / f'{uid}_img.png'),
      np.copy(left_img)
  )
  # cv2.imwrite(
  #     str(_DEBUG_FILE_PATH / f'{uid}_seg.png'),
  #     label.draw_pixelwise_mask(np.copy(left_img), seg_mask)
  # )
  # print("heatmap", obb_datapoint.heat_map.shape)
  # # print('\n')
  # # print("heatmap", obb_datapoint.heat_map)
  centroid_target = np.clip(obb_datapoint.heat_map, 0.0, 1.0) * 255.0
  # heatmap = np.clip(obb_datapoint.heat_map*255,0,255)

  heatmap = cv2.applyColorMap(centroid_target.astype(np.uint8), cv2.COLORMAP_JET)
  cv2.imwrite(
      str(_DEBUG_FILE_PATH / f'{uid}_heatmap.png'),
      heatmap
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
  # cv2.imwrite(
  #     str(_DEBUG_FILE_PATH / f'{uid}_seg.png'),
  #     label.draw_pixelwise_mask(np.copy(left_img), seg_mask)
  # )

  latent_outputs, abs_pose_outputs, peak_img_output, _ = obb_outputs.compute_pointclouds_and_poses(
    obb_datapoint.heat_map, 
    obb_datapoint.latent_emb,
    obb_datapoint.abs_pose)

  cv2.imwrite(
        str(_DEBUG_FILE_PATH / f'{uid}_peak_img_output.png'),
        np.copy(peak_img_output)
    )

  for i in range(len(point_clouds)):
    emb = latent_outputs[i]
    # print("emb before cpu", emb.size())
    # emb = emb.unsqueeze(0)
    # emb = emb.cpu().detach().numpy()
    # print("emb size",emb.size()[0])

    emb = torch.FloatTensor(emb).unsqueeze(0)
    emb = emb.cuda()
    _, shape_out = estimator(None, emb)
    shape_out = shape_out.cpu().detach().numpy()[0]
    
    filename = str(_DEBUG_FILE_PATH) + '/decoder_output'+ uid + '_'+str(i)+'.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(shape_out)
    o3d.io.write_point_cloud(filename, pcd)

    filename = str(_DEBUG_FILE_PATH) + '/original_pc'+ uid + '_'+str(i)+'.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds[i])
    o3d.io.write_point_cloud(filename, pcd)

    rotated_pc, rotated_box = get_gt_pointclouds(poses[i], point_clouds[i], use_camera_RT = True)
    filename = str(_DEBUG_FILE_PATH) + '/actual_rotated_output'+ uid + '_'+str(i)+'.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rotated_pc)
    o3d.io.write_point_cloud(filename, pcd)
    filename = str(_DEBUG_FILE_PATH) + '/actual_box_output'+ uid + '_'+str(i)+'.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rotated_box)
    o3d.io.write_point_cloud(filename, pcd)


    out_rotated_pc, out_rotated_box = get_gt_pointclouds(abs_pose_outputs[i], shape_out, use_camera_RT = False)
    print("out_rotated_pc", out_rotated_pc.shape)
    print("out rotated box", out_rotated_box.shape)
    filename = str(_DEBUG_FILE_PATH) + '/out_rotated_output'+ uid + '_'+str(i)+'.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_rotated_pc)
    o3d.io.write_point_cloud(filename, pcd)
    filename = str(_DEBUG_FILE_PATH) + '/out_box_output'+ uid + '_'+str(i)+'.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_rotated_box)
    o3d.io.write_point_cloud(filename, pcd)
    
    
    
    # filename = str(_DEBUG_FILE_PATH) + '/inroot_rotated_output'+ uid + '_'+str(i)+'.ply'
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(in_root_point_clouds[i])
    # o3d.io.write_point_cloud(filename, pcd)
    print("@@@@@@@@@@@@@@@@@@@@@@@\n")

    actual_pose = _CAMERA.RT_matrix @ poses[i].camera_T_object
    # print("RtR", (np.matmul(actual_pose[:3,:3],actual_pose[:3,:3].T)))
    # print("determinant", np.linalg.det(actual_pose[:3,:3]))

    print("actual pose",actual_pose)
    print("scale", poses[i].scale_matrix)
    print("=============\n")


    print("abs pose", abs_pose_outputs[i].camera_T_object)
    print("abs scale factor", abs_pose_outputs[i].scale_matrix)
    print("------------------------\n")
