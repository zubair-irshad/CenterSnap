#!/opt/mmt/python_venv/bin/python
import random
import copy
import numpy as np
import cv2
import pathlib
import json
import IPython
import torch
import open3d as o3d
from simnet.lib.net.post_processing.epnp import _WORLD_T_POINTS
from simnet.lib import enviroment
from simnet.lib import pose_sampler
from simnet.lib import camera
from simnet.lib import sg
from simnet.lib import label
from simnet.lib import transform
from simnet.lib import datapoint
from simnet.lib import urdf
from simnet.lib import wipeable_primitive
from simnet.lib.depth_noise import DepthManager
from app.panoptic_tidying import tidy_classes_nocs
# from app.panoptic_tidying import shapenet4_classes
# from simnet.lib.graspable_objects import sample_graspable_objects

from simnet.lib.net.models.auto_encoder import PointCloudAE

# from simnet.lib.tidy_objects_shapenet import sample_tidy_objects
from simnet.lib.tidy_objects import sample_tidy_objects

# from simnet.lib.test_objects import sample_tidy_objects
# from simnet.lib.ycb_objects import sample_ycb_objects as sample_tidy_objects

# random.seed(1234)
# np.random.seed(1234)

from simnet.lib.net.pre_processing import obb_inputs
from simnet.lib.net.post_processing import abs_pose_outputs, pose_outputs

_DATASET_NAME = 'zubair_nocs_real_train_test'
_TEST = True

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

# _CAMERA = camera.HSRCamera(scale_factor=1)

_CAMERA = camera.NOCS_Real()

# Specifiy object categories i.e. 5 or 10 here
OBJECT_CATEGORY = 30



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


# Sampel objects on the table.
def sample_objects(scene_node, default_objects=None):
  surface_nodes = scene_node.get_all_nodes_by_name('wipeable_surface_part')
  if len(surface_nodes) == 0:
    surface_nodes = scene_node.get_all_nodes_by_name('wipeable_surface')
  for surface_node in surface_nodes:
    # Have empty surfaces 10% of the time
    if np.random.uniform() < .1:
      continue
    if OBJECT_CATEGORY == 5:
      sample_graspable_objects(scene_node, surface_node)
    elif OBJECT_CATEGORY == 30:
      sample_tidy_objects(scene_node, surface_node)


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

env.set_target_categories(tidy_classes_nocs.categories)
# Set the graspable objects on the table.
env.add_target_class('small_object', sample_objects)

env.set_camera_sampler(camera_sampler)
env.generate_data()

# Get the labeled data.
left_img, right_img, depth_img = env.get_rendered_data()
seg_mask = env.get_segmentation_mask()

object_category_seg_mask, category_names = env.get_object_category_segmentation_mask()

print("CATEGORY NAMES", category_names)
obbs, instance_masks = env.get_oriented_bounding_boxes('small_object')

# print("\n")
# print("len obbs",len(obbs))
# print("\n\n")
point_clouds = env.get_pointclouds('small_object')
poses = env.get_poses('small_object')


#Get sorted pointclouds
# x_index=[]
# for z in range(len(point_clouds)):
#   center = [ np.average(indices) for indices in np.where(instance_masks[z]>0)]
#   x_index.append(center[0])

# point_clouds_sorted = [x for y, x in sorted(zip(x_index, point_clouds), key=lambda pair: pair[0])]

# load auto-encoder weights
emb_dim = 128
n_cat = 57
n_pts = 2048
model_path = str(pathlib.Path(__file__).parent.parent.parent / 'data' / 'auto_encoder_model' / 'model_50_nocs.pth')
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
# obb_datapoint = obb_inputs.compute_network_targets(
#     obbs, instance_masks,latent_embeddings, poses, _CAMERA.height, _CAMERA.width, _CAMERA
# )
obb_datapoint = obb_inputs.compute_nocs_network_targets(instance_masks, latent_embeddings, poses,_CAMERA.height, _CAMERA.width, _CAMERA)

# print("latent emb",obb_datapoint.latent_emb)
# print("vertex target", obb_datapoint.vertex_target.shape)
# print("covariance target", obb_datapoint.cov_matrices.shape)

# detections_gt = obb_outputs.get_gt_detections(
#           obb_datapoint.heat_map,
#           obb_datapoint.vertex_target,
#           obb_datapoint.cov_matrices,
#           obb_datapoint.z_centroid,
#           camera_model=_CAMERA
#       )

DM = DepthManager()
noisy_depth  = DM.prepare_depth_data(depth_img)

stereo_datapoint = datapoint.Stereo(left_color=left_img, right_color=noisy_depth)
panoptic_datapoint = datapoint.Panoptic(
    stereo=stereo_datapoint,
    depth=depth_img,
    segmentation=object_category_seg_mask,
    object_poses=[obb_datapoint],
    boxes=category_names,
    detections=point_clouds
    # boxes=[],
    # detections=[]
)


_DATASET.write(panoptic_datapoint)
if _TEST:
  uid = panoptic_datapoint.uid

  cv2.imwrite(
      str(_DEBUG_FILE_PATH / f'{uid}_img.png'),
      np.copy(left_img)
  )
  # cv2.imwrite(
  #     str(_DEBUG_FILE_PATH / f'{uid}_obj_seg.png'),
  #     label.draw_pixelwise_mask(np.copy(left_img), object_category_seg_mask)
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
  # cv2.imwrite(
  #     str(_DEBUG_FILE_PATH / f'{uid}_seg.png'),
  #     label.draw_pixelwise_mask(np.copy(left_img), seg_mask)
  # )

  latent_outputs, abs_pose_outs, peak_img_output, peak_img_nms, _ = abs_pose_outputs.compute_pointclouds_and_poses(
    obb_datapoint.heat_map, 
    obb_datapoint.latent_emb,
    obb_datapoint.abs_pose, 
    min_confidence = 0.45)

  cv2.imwrite(
        str(_DEBUG_FILE_PATH / f'{uid}_peak_img_output.png'),
        np.copy(peak_img_output)
    )

  for i in range(len(point_clouds)):
    emb = latent_outputs[i]

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


    out_rotated_pc, out_rotated_box = get_gt_pointclouds(abs_pose_outs[i], shape_out, use_camera_RT = False)
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
    
    print("@@@@@@@@@@@@@@@@@@@@@@@\n")

    actual_pose = _CAMERA.RT_matrix @ poses[i].camera_T_object
    print("actual pose",actual_pose)
    print("scale", poses[i].scale_matrix)
    print("=============\n")