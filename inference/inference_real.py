import argparse
import pathlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time
import pytorch_lightning as pl
import _pickle as cPickle
from simnet.lib.net import common
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
from simnet.lib.net.models.auto_encoder import PointCloudAE
from utils.nocs_utils import load_img_NOCS, create_input_norm
from utils.viz_utils import depth2inv, viz_inv_depth
from utils.transform_utils import get_gt_pointclouds, transform_coordinates_3d, calculate_2d_projections
from utils.transform_utils import project
from utils.viz_utils import save_projected_points, draw_bboxes, line_set_mesh

import time
def get_auto_encoder(model_path):
  emb_dim = 128
  n_pts = 2048
  ae = PointCloudAE(emb_dim, n_pts)
  ae.cuda()
  ae.load_state_dict(torch.load(model_path))
  ae.eval()
  return ae

def inference(
    hparams,
    data_dir, 
    output_path,
    min_confidence=0.1,
    use_gpu=True,
):
  model = PanopticModel(hparams, 0, None, None)
  model.eval()
  if use_gpu:
    model.cuda()
  data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
  _CAMERA = camera.NOCS_Real()
  min_confidence = 0.50

  for i, img_path in enumerate(data_path):
    img_full_path = os.path.join(data_dir, 'Real', img_path)
    color_path = img_full_path + '_color.png' 
    if not os.path.exists(color_path):
      continue
    depth_full_path = img_full_path + '_depth.png'
    img_vis = cv2.imread(color_path)
    left_linear, depth, actual_depth = load_img_NOCS(color_path, depth_full_path)
    input = create_input_norm(left_linear, depth)
    input = input[None, :, :, :]
    if use_gpu:
      input = input.to(torch.device('cuda:0'))
    with torch.no_grad():
      _, _, _ , pose_output = model.forward(input)
      latent_emb_outputs, abs_pose_outputs, img_output, _, _ = pose_output.compute_pointclouds_and_poses(min_confidence,is_target = False)

    auto_encoder_path = os.path.join(data_dir, 'ae_checkpoints', 'model_50_nocs.pth')
    ae = get_auto_encoder(auto_encoder_path)
    cv2.imwrite(
        str(output_path / f'{i}_image.png'),
        np.copy(np.copy(img_vis))
    )
    cv2.imwrite(
        str(output_path / f'{i}_peaks_output.png'),
        np.copy(img_output)
    )
    depth_vis = depth2inv(torch.tensor(depth).unsqueeze(0).unsqueeze(0))
    depth_vis = viz_inv_depth(depth_vis)
    depth_vis = depth_vis*255.0
    cv2.imwrite(
        str(output_path / f'{i}_depth_vis.png'),
        np.copy(depth_vis)
    )
    write_pcd = False
    rotated_pcds = []
    points_2d = []
    box_obb = []
    axes = []

    for j in range(len(latent_emb_outputs)):
        emb = latent_emb_outputs[j]
        emb = latent_emb_outputs[j]
        emb = torch.FloatTensor(emb).unsqueeze(0)
        emb = emb.cuda()
        _, shape_out = ae(None, emb)
        shape_out = shape_out.cpu().detach().numpy()[0]

        rotated_pc, rotated_box, _ = get_gt_pointclouds(abs_pose_outputs[j], shape_out, camera_model = _CAMERA)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rotated_pc)
        filename_rotated = str(output_path) + '/pcd_rotated'+str(i)+str(j)+'.ply'
        if write_pcd:
            o3d.io.write_point_cloud(filename_rotated, pcd)
        else:
            rotated_pcds.append(pcd)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        T = abs_pose_outputs[j].camera_T_object
        mesh_frame = mesh_frame.transform(T)
        rotated_pcds.append(mesh_frame)
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
          rotated_pcds.append(cylinder_segments[k])

            
        points_mesh = camera.convert_points_to_homopoints(rotated_pc.T)
        points_2d_mesh = project(_CAMERA.K_matrix, points_mesh)
        points_2d_mesh = points_2d_mesh.T
        points_2d.append(points_2d_mesh)
        #2D output
        points_obb = camera.convert_points_to_homopoints(np.array(rotated_box).T)
        points_2d_obb = project(_CAMERA.K_matrix, points_obb)
        points_2d_obb = points_2d_obb.T
        box_obb.append(points_2d_obb)
        xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
        sRT = abs_pose_outputs[j].camera_T_object @ abs_pose_outputs[j].scale_matrix
        transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
        projected_axes = calculate_2d_projections(transformed_axes, _CAMERA.K_matrix[:3,:3])
        axes.append(projected_axes)
    
    if not write_pcd:
      o3d.visualization.draw_geometries(rotated_pcds)
      save_projected_points(np.copy(img_vis), points_2d, str(output_path), i)
      
      colors_box = [(63, 237, 234)]
      im = np.array(np.copy(img_vis)).copy()
      for k in range(len(colors_box)):
        for points_2d, axis in zip(box_obb, axes):
          points_2d = np.array(points_2d)
          im = draw_bboxes(im, points_2d, axis, colors_box[k])
      box_plot_name = str(output_path)+'/box3d'+str(i)+'.png'
      cv2.imwrite(
          box_plot_name,
          np.copy(im)
      )
    print("done with image: ", i )

if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  app_group = parser.add_argument_group('app')
  app_group.add_argument('--app_output', default='inference_best', type=str)
  app_group.add_argument('--result_name', default='centersnap_nocs', type=str)
  app_group.add_argument('--data_dir', default='nocs_data', type=str)

  hparams = parser.parse_args()
  print(hparams)
  result_name = hparams.result_name
  path = 'data/'+result_name
  output_path = pathlib.Path(path) / hparams.app_output
  output_path.mkdir(parents=True, exist_ok=True)
  inference(hparams, hparams.data_dir, output_path)
