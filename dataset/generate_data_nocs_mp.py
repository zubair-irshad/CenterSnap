#!/opt/mmt/python_venv/bin/python
import random
import copy
import numpy as np
import cv2
import pathlib
import IPython
import dataclasses
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
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# import open3d as o3d
import json
from simnet.lib.net.post_processing.epnp import _WORLD_T_POINTS
import open3d as o3d
from simnet.lib.graspable_objects import sample_graspable_objects
from simnet.lib.net.pre_processing import obb_inputs
from simnet.lib.net.post_processing import abs_pose_outputs
from simnet.lib.depth_noise import DepthManager

from simnet.lib.net.models.auto_encoder import PointCloudAE
from simnet.lib.tidy_objects import sample_tidy_objects
import time
import math
from PIL import Image
from multiprocessing import Pool
@dataclasses.dataclass
class Pose:
  camera_T_object: np.ndarray
  scale_matrix: np.ndarray

_DATASET_NAME = 'zubair_panoptic_NOCS'
_TEST = True

_TIDY_CLASSES = json.load(open(pathlib.Path(__file__).parent / 'ycb_classes.json'))

def plot_jittered(orig_img, imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])
    plt.tight_layout()
    plt.show()

def get_gt_pointclouds(pose, pc
):
    # pose.camera_T_object[:3,3] = pose.camera_T_object[:3,3]*100
    # pose.scale_matrix[:3,:3] = pose.scale_matrix[:3,:3]*100
    # for pose, pc in zip(poses, pointclouds):
    pc_homopoints = camera.convert_points_to_homopoints(pc.T)
    unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)
    morphed_pc_homopoints = pose.camera_T_object @ (pose.scale_matrix @ pc_homopoints)
    morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
  
    morphed_pc_homopoints = camera.convert_homopoints_to_points(morphed_pc_homopoints).T
    morphed_box_points = camera.convert_homopoints_to_points(morphed_box_homopoints).T
    # box_points.append(morphed_box_points)
    return morphed_pc_homopoints, morphed_box_points

import numpy as np
np.set_printoptions(threshold=np.inf)
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


import os
import sys
import glob
import cv2
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
sys.path.append('/home/zubairirshad/object-deformnet/lib')
from align import align_nocs_to_depth
from utils import load_depth


def project(K, p_3d):
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
    return projections_2d

def visualize_projected_points(color_img, pcd_array, box_obb):
    open_3d_lines = [
        [5, 3],
        [6, 4],
        [0, 2],
        [1, 7],
        [0, 3],
        [1, 6],
        [2, 5],
        [4, 7],
        [0, 1],
        [6, 3],
        [4, 5],
        [2, 7],
    ]
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    plt.xlim((0, color_img.shape[1]))
    plt.ylim((0, color_img.shape[0]))
    # Projections
    color = ['g', 'y', 'b', 'r', 'm', 'c', '#3a7c00', '#3a7cd9', '#8b7cd9', '#211249']
    for i, points_2d_mesh in enumerate(pcd_array):
        for points in points_2d_mesh:
            plt.scatter(points[0], points[1], color=color[i], s=2)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.imshow(color_img)
    # plt_name = name+'plot_points'

    # # plt.show()
    # # plt.savefig('/home/zubairirshad/fleet/simnet/data/simnet/zubair_ZED2_test_pose_4/'+plt_name+'img.png', bbox_inches='tight')

    for points_2d in box_obb:
        for edge in open_3d_lines:
            plt.plot(points_2d[edge, 0], points_2d[edge, 1], color='b', linewidth=1.0)
    plt.axis('off')
    plt.show()


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


def create_img_list(data_dir):
    """ Create train/val/test data list for CAMERA and Real. """
    # CAMERA dataset
    for subset in ['train', 'val']:
        img_list = []
        img_dir = os.path.join(data_dir, 'CAMERA', subset)
        folder_list = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
        for i in range(10*len(folder_list)):
            folder_id = int(i) // 10
            img_id = int(i) % 10
            img_path = os.path.join(subset, '{:05d}'.format(folder_id), '{:04d}'.format(img_id))
            img_list.append(img_path)
        with open(os.path.join(data_dir, 'CAMERA', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    # Real dataset
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, 'Real', subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        with open(os.path.join(data_dir, 'Real', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')


def process_data(img_path, depth):
    """ Load instance masks for the objects in the image. """
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
            else:
                model_id = line_info[3]    # CAMERA objs
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754' or model_id == 'd3b53f56b4a7b3b3c9f016d57db96408':
                continue
            # process foreground objects
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes


def annotate_camera_train(img_path, estimator, obj_models, mug_meta):
    _camera = camera.NOCS_Camera()
    """ Generate gt labels for CAMERA train data. """
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    # meta info for re-label mug category
    # start_time = time.time()
    img_full_path = os.path.join(data_dir, 'CAMERA', img_path)
    depth_composed_path = img_path+'_composed.png'
    depth_full_path = os.path.join(data_dir, 'camera_composed_depth','camera_full_depths', depth_composed_path)
    all_exist = os.path.exists(img_full_path + '_color.png') and \
                os.path.exists(img_full_path + '_coord.png') and \
                os.path.exists(img_full_path + '_depth.png') and \
                os.path.exists(img_full_path + '_mask.png') and \
                os.path.exists(img_full_path + '_meta.txt')
    if not all_exist:
        raise ValueError("Incorrect dimension for scene size.")
    depth = load_depth(depth_full_path)
    masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
    if instance_ids is None:
        raise ValueError("Incorrect dimension for scene size.")
    # Umeyama alignment of GT NOCS map with depth image
    scales, rotations, translations, error_messages, _ = \
        align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path)
    if error_messages:
        raise ValueError("Incorrect dimension for scene size.")
    # re-label for mug category
    for i in range(len(class_ids)):
        if class_ids[i] == 6:
            T0 = mug_meta[model_list[i]][0]
            s0 = mug_meta[model_list[i]][1]
            T = translations[i] - scales[i] * rotations[i] @ T0
            s = scales[i] / s0
            scales[i] = s
            translations[i] = T

    # print("time before centerpoint ", time.time()-start_time)
    
    ### GET CENTERPOINT DATAPOINTS
    #get latent embeddings
    
    model_points = [obj_models[model_list[i]].astype(np.float32) for i in range(len(class_ids))]
    latent_embeddings = get_latent_embeddings(model_points, estimator)

    # print("time until latent embedding ", time.time()-start_time)
    #get poses 
    abs_poses=[]
    class_num=1
    seg_mask = np.zeros([_camera.height, _camera.width])
    masks_list = []
    for i in range(len(class_ids)):
        R = rotations[i]
        T = translations[i]
        s = scales[i]
        sym_ids = [0, 1, 3]
        cat_id = np.array(class_ids)[i] - 1 
        if cat_id in sym_ids:
            R = align_rotation(R)
        
        scale_matrix = np.eye(4)
        scale_mat = s*np.eye(3, dtype=float)
        scale_matrix[0:3, 0:3] = scale_mat
        camera_T_object = np.eye(4)
        camera_T_object[:3,:3] = R
        camera_T_object[:3,3] = T
        seg_mask[masks[:,:,i] > 0] = class_num
        class_num += 1
        
        masks_list.append(masks[:,:,i])
        abs_poses.append(transform.Pose(camera_T_object=camera_T_object, scale_matrix=scale_matrix))
    # print("time before obb datapoints",time.time()-start_time )
    obb_datapoint = obb_inputs.compute_nocs_network_targets(masks_list, latent_embeddings, abs_poses,_camera.height, _camera.width)

    # print("time after obb datapoint", time.time()-start_time)
            
    color_img = cv2.imread(img_full_path + '_color.png')
    colorjitter = transforms.ColorJitter(0.5, 0.3, 0.5, 0.05)
    rgb_img = colorjitter(Image.fromarray(color_img))
    jitter_img = colorjitter(rgb_img)
    color_img = np.asarray(jitter_img)
    depth_array = np.array(depth, dtype=np.float32)/255.0
    DM = DepthManager()
    noisy_depth  = DM.prepare_depth_data(depth_array)
    stereo_datapoint = datapoint.Stereo(left_color=color_img, right_color=noisy_depth)
    panoptic_datapoint = datapoint.Panoptic(
    stereo=stereo_datapoint,
    depth=noisy_depth,
    segmentation=seg_mask,
    object_poses=[obb_datapoint],
    boxes=[],
    detections=[]
    )
    # print("end time",time.time()-start_time)
    _DATASET.write(panoptic_datapoint)
    # print("end time with data writing",time.time()-start_time )
    # print("-------------------------------------\n\n\n")
        ### Finish writing datapoint

        # if _TEST:
        #     uid = panoptic_datapoint.uid

        #     cv2.imwrite(
        #         str(_DEBUG_FILE_PATH / f'{uid}_seg.png'),
        #         label.draw_pixelwise_mask(np.copy(color_img), seg_mask)
        #     )

        #     # depth_image = cv2.applyColorMap(noisy_depth, cv2.COLORMAP_JET)
        #     cv2.imwrite(
        #         str(_DEBUG_FILE_PATH / f'{uid}_depth.png'),
        #         np.copy(noisy_depth)
        #     )
        #     cv2.imwrite(
        #         str(_DEBUG_FILE_PATH / f'{uid}_img.png'),
        #         np.copy(color_img)
        #     )
        #     centroid_target = np.clip(obb_datapoint.heat_map, 0.0, 1.0) * 255.0
        #     heatmap = cv2.applyColorMap(centroid_target.astype(np.uint8), cv2.COLORMAP_JET)
        #     cv2.imwrite(
        #         str(_DEBUG_FILE_PATH / f'{uid}_heatmap.png'),
        #         heatmap
        #     )
            
        #     latent_outputs, abs_pose_outs, peak_img_output, _ = abs_pose_outputs.compute_pointclouds_and_poses(
        #     obb_datapoint.heat_map, 
        #     obb_datapoint.latent_emb,
        #     obb_datapoint.abs_pose)

        #     peak_img_overlayed = cv2.addWeighted(
        #         peak_img_output.astype(np.uint8), 0.8, color_img.astype(np.uint8), 0.3, 0
        #     )

        #     cv2.imwrite(
        #         str(_DEBUG_FILE_PATH / f'{uid}_peak_img_output.png'),
        #         np.copy(peak_img_overlayed)
        #     )

        #     for i in range(len(latent_outputs)):
        #         emb = latent_outputs[i]
        #         emb = torch.FloatTensor(emb).unsqueeze(0)
        #         emb = emb.cuda()
        #         _, shape_out = estimator(None, emb)
        #         shape_out = shape_out.cpu().detach().numpy()[0]
                
        #         filename = str(_DEBUG_FILE_PATH) + '/decoder_output'+ uid + '_'+str(i)+'.ply'
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(shape_out)
        #         o3d.io.write_point_cloud(filename, pcd)

        #         filename = str(_DEBUG_FILE_PATH) + '/original_pc'+ uid + '_'+str(i)+'.ply'
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(model_points[i])
        #         o3d.io.write_point_cloud(filename, pcd)

        #         rotated_pc, rotated_box = get_gt_pointclouds(abs_poses[i], model_points[i])
        #         filename = str(_DEBUG_FILE_PATH) + '/actual_rotated_output'+ uid + '_'+str(i)+'.ply'
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(rotated_pc)
        #         o3d.io.write_point_cloud(filename, pcd)
        #         filename = str(_DEBUG_FILE_PATH) + '/actual_box_output'+ uid + '_'+str(i)+'.ply'
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(rotated_box)
        #         o3d.io.write_point_cloud(filename, pcd)

        #         abs_pose_outs[i].camera_T_object[:3,:3] = align_rotation(abs_pose_outs[i].camera_T_object[:3,:3])

        #         out_rotated_pc, out_rotated_box = get_gt_pointclouds(abs_pose_outs[i], shape_out)
        #         print("out_rotated_pc", out_rotated_pc.shape)
        #         print("out rotated box", out_rotated_box.shape)
        #         filename = str(_DEBUG_FILE_PATH) + '/out_rotated_output'+ uid + '_'+str(i)+'.ply'
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(out_rotated_pc)
        #         o3d.io.write_point_cloud(filename, pcd)
        #         filename = str(_DEBUG_FILE_PATH) + '/out_box_output'+ uid + '_'+str(i)+'.ply'
        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(out_rotated_box)
        #         o3d.io.write_point_cloud(filename, pcd)

        #         filename = str(_DEBUG_FILE_PATH) + '/mesh_frame'+ uid + '_'+str(i)+'.ply'
        #         mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        #         mesh_t = copy.deepcopy(mesh_frame)
        #         T = abs_pose_outs[i].camera_T_object
        #         mesh_t = mesh_t.transform(T)
        #         o3d.io.write_triangle_mesh(filename,mesh_t)


        # TEST MODELS
        # test_points = []
        # test_2d_points = []
        # test_obb = []
        # for i in range(len(class_ids)):
        #     R = rotations[i]
        #     T = translations[i]
        #     s = scales[i]
        #     sym_ids = [0, 1, 3]
        #     cat_id = np.array(class_ids)[i] - 1 
        #     print("cat id",cat_id)
        #     if cat_id in sym_ids:
        #         print("R before", R)
        #         R = align_rotation(R)
        #         # R = align_rotation(R)
        #         print("R AFTER", R)
        #     if model_list[i] == 'b9be7cfe653740eb7633a2dd89cec754' or model_list[i] =='d3b53f56b4a7b3b3c9f016d57db96408':
        #         continue
        #     model_points = obj_models[model_list[i]]
        #     print(s)
        #     scale_matrix = np.eye(4)
        #     scale_mat = s*np.eye(3, dtype=float)
        #     scale_matrix[0:3, 0:3] = scale_mat
        #     camera_T_object = np.eye(4)
        #     camera_T_object[:3,:3] = R
        #     camera_T_object[:3,3] = T

        #     pc_homopoints = convert_points_to_homopoints(model_points.T)
        #     morphed_pc_homopoints = camera_T_object @ (scale_matrix @ pc_homopoints)
        #     K_matrix = np.eye(4)
        #     K_matrix[:3,:3] = intrinsics
        #     points_2d_mesh = project(K_matrix, morphed_pc_homopoints)
        #     points_2d_mesh = points_2d_mesh.T

        #     print("--------------\n")
        #     print("scale", s)
        #     print("R,T", camera_T_object)
        #     print("--------------\n")
        #     morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T
        #     color = np.repeat(np.array([[1, 0, 0]]), morphed_pc_homopoints.shape[0], axis=0)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(morphed_pc_homopoints)
            
        #     obb = pcd.get_oriented_bounding_box().get_box_points()
        #     points_obb = convert_points_to_homopoints(np.array(obb).T)
        #     points_2d_obb = project(K_matrix, points_obb)
        #     points_2d_obb = points_2d_obb.T

        #     test_obb.append(points_2d_obb)
        #     pcd.colors = o3d.utility.Vector3dVector(color)
        #     test_points.append(pcd)
        #     test_2d_points.append(points_2d_mesh)

        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        # test_points.append(mesh_frame)
        # color_img = cv2.imread(img_full_path + '_color.png')
        # visualize_projected_points(color_img,test_2d_points, test_obb)


def annotate_real_train(data_dir, estimator):
    """ Generate gt labels for Real train data through PnP. """
    _camera = camera.NOCS_Real()
    real_train = open(os.path.join(data_dir, 'Real/train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # scale factors for all instances
    scale_factors = {}
    path_to_size = glob.glob(os.path.join(data_dir, 'obj_models/real_train', '*_norm.txt'))
    for inst_path in sorted(path_to_size):
        instance = os.path.basename(inst_path).split('.')[0]
        bbox_dims = np.loadtxt(inst_path)
        scale_factors[instance] = np.linalg.norm(bbox_dims)
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)
    #TEST MODELS
    obj_model_dir = '/home/zubairirshad/object-deformnet/data/obj_models'
    with open(os.path.join(obj_model_dir, 'real_train4096.pkl'), 'rb') as f:
        obj_models = cPickle.load(f)

    valid_img_list = []
    for img_path in tqdm(real_train):
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth_full_path = img_full_path+'_depth.png'
        depth = load_depth(depth_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # compute pose
        num_insts = len(class_ids)
        scales = np.zeros(num_insts)
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))
        for i in range(num_insts):
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]
            idxs = np.where(mask)
            coord = coords[:, :, i, :]
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            coord_pts = coord_pts[:, :, None]
            img_pts = np.array([idxs[1], idxs[0]]).transpose()
            img_pts = img_pts[:, :, None].astype(float)
            distCoeffs = np.zeros((4, 1))    # no distoration
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
            assert retval
            R, _ = cv2.Rodrigues(rvec)
            T = np.squeeze(tvec)
            # re-label for mug category
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = T - s * R @ T0
                s = s / s0
            scales[i] = s
            rotations[i] = R
            translations[i] = T

        ### GET CENTERPOINT DATAPOINTS
        #get latent embeddings
        model_points = [obj_models[model_list[i]].astype(np.float32) for i in range(len(class_ids))]
        latent_embeddings = get_latent_embeddings(model_points, estimator)
        #get poses 
        abs_poses=[]
        class_num=1
        seg_mask = np.zeros([_camera.height, _camera.width])
        masks_list = []
        for i in range(len(class_ids)):
            R = rotations[i]
            T = translations[i]
            s = scales[i]
            sym_ids = [0, 1, 3]
            cat_id = np.array(class_ids)[i] - 1 
            if cat_id in sym_ids:
                R = align_rotation(R)
            scale_matrix = np.eye(4)
            scale_mat = s*np.eye(3, dtype=float)
            scale_matrix[0:3, 0:3] = scale_mat
            camera_T_object = np.eye(4)
            camera_T_object[:3,:3] = R
            camera_T_object[:3,3] = T
            seg_mask[masks[:,:,i] > 0] = class_num
            class_num += 1
            masks_list.append(masks[:,:,i])
            abs_poses.append(transform.Pose(camera_T_object=camera_T_object, scale_matrix=scale_matrix))
        obb_datapoint = obb_inputs.compute_nocs_network_targets(masks_list, latent_embeddings, abs_poses,_camera.height, _camera.width)

        color_img = cv2.imread(img_full_path + '_color.png')
        depth_array = np.array(depth, dtype=np.float32)/255.0
        DM = DepthManager()
        noisy_depth  = DM.prepare_depth_data(depth_array)
        stereo_datapoint = datapoint.Stereo(left_color=color_img, right_color=noisy_depth)
        panoptic_datapoint = datapoint.Panoptic(
        stereo=stereo_datapoint,
        depth=noisy_depth,
        segmentation=seg_mask,
        object_poses=[obb_datapoint],
        boxes=[],
        detections=[]
        )
        _DATASET.write(panoptic_datapoint)
        ### Finish writing datapoint

def annotate_test_data(data_dir, estimator):
    """ Generate gt labels for test data.
        Properly copy handle_visibility provided by NOCS gts.
    """

    _camera = camera.NOCS_Camera()
    #SAVE VALIDATION BOTH WITH DEPTH NOISE AND WITHOUT
    # Statistics:
    # test_set    missing file     bad rendering    no (occluded) fg    occlusion (< 64 pts)
    #   val        3792 imgs        132 imgs         1856 (23) imgs      50 insts
    #   test       0 img            0 img            0 img               2 insts

    camera_val = open(os.path.join(data_dir, 'CAMERA', 'val_list_all.txt')).read().splitlines()
    real_test = open(os.path.join(data_dir, 'Real', 'test_list_all.txt')).read().splitlines()
    camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # compute model size
    model_file_path = ['obj_models/camera_val4096.pkl', 'obj_models/real_test4096.pkl']
    models = {}
    for path in model_file_path:
        with open(os.path.join(data_dir, path), 'rb') as f:
            models.update(cPickle.load(f))
    model_sizes = {}
    for key in models.keys():
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    #TEST MODELS
    obj_model_dir = '/home/zubairirshad/object-deformnet/data/obj_models'
    with open(os.path.join(obj_model_dir, 'camera_val.pkl'), 'rb') as f:
        obj_models = cPickle.load(f)

    subset_meta = [('CAMERA', camera_val, camera_intrinsics, 'val'), ('Real', real_test, real_intrinsics, 'test')]
    for source, img_list, intrinsics, subset in subset_meta:
        valid_img_list = []
        for img_path in tqdm(img_list):
            img_full_path = os.path.join(data_dir, source, img_path)
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_coord.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_mask.png') and \
                        os.path.exists(img_full_path + '_meta.txt')
            if not all_exist:
                continue
            depth_composed_path = img_path+'_composed.png'
            depth_full_path = os.path.join(data_dir, 'camera_composed_depth','camera_full_depths', depth_composed_path)
            depth = load_depth(depth_full_path)
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
            if instance_ids is None:
                continue
            num_insts = len(instance_ids)
            # match each instance with NOCS ground truth to properly assign gt_handle_visibility
            nocs_dir = os.path.join(os.path.dirname(data_dir), 'results/nocs_results')
            if source == 'CAMERA':
                nocs_path = os.path.join(nocs_dir, 'val', 'results_val_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            else:
                nocs_path = os.path.join(nocs_dir, 'real_test', 'results_test_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            with open(nocs_path, 'rb') as f:
                nocs = cPickle.load(f)
            gt_class_ids = nocs['gt_class_ids']
            gt_bboxes = nocs['gt_bboxes']
            gt_sRT = nocs['gt_RTs']
            gt_handle_visibility = nocs['gt_handle_visibility']
            map_to_nocs = []
            for i in range(num_insts):
                gt_match = -1
                for j in range(len(gt_class_ids)):
                    if gt_class_ids[j] != class_ids[i]:
                        continue
                    if np.sum(np.abs(bboxes[i] - gt_bboxes[j])) > 5:
                        continue
                    # match found
                    gt_match = j
                    break
                # check match validity
                assert gt_match > -1, print(img_path, instance_ids[i], 'no match for instance')
                assert gt_match not in map_to_nocs, print(img_path, instance_ids[i], 'duplicate match')
                map_to_nocs.append(gt_match)
            # copy from ground truth, re-label for mug category
            handle_visibility = gt_handle_visibility[map_to_nocs]
            sizes = np.zeros((num_insts, 3))
            poses = np.zeros((num_insts, 4, 4))
            scales = np.zeros(num_insts)
            rotations = np.zeros((num_insts, 3, 3))
            translations = np.zeros((num_insts, 3))
            for i in range(num_insts):
                gt_idx = map_to_nocs[i]
                sizes[i] = model_sizes[model_list[i]]
                sRT = gt_sRT[gt_idx]
                s = np.cbrt(np.linalg.det(sRT[:3, :3]))
                R = sRT[:3, :3] / s
                T = sRT[:3, 3]
                # re-label mug category
                if class_ids[i] == 6:
                    T0 = mug_meta[model_list[i]][0]
                    s0 = mug_meta[model_list[i]][1]
                    T = T - s * R @ T0
                    s = s / s0
                # used for test during training
                scales[i] = s
                rotations[i] = R
                translations[i] = T
                # used for evaluation
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = s * R
                sRT[:3, 3] = T
                poses[i] = sRT

            ### GET CENTERPOINT DATAPOINTS
            #get latent embeddings
            model_points = [obj_models[model_list[i]].astype(np.float32) for i in range(len(class_ids))]
            latent_embeddings = get_latent_embeddings(model_points, estimator)
            #get poses 
            abs_poses=[]
            class_num=1
            seg_mask = np.zeros([_camera.height, _camera.width])
            masks_list = []
            for i in range(len(class_ids)):
                R = rotations[i]
                T = translations[i]
                s = scales[i]
                sym_ids = [0, 1, 3]
                cat_id = np.array(class_ids)[i] - 1 
                if cat_id in sym_ids:
                    R = align_rotation(R)
                scale_matrix = np.eye(4)
                scale_mat = s*np.eye(3, dtype=float)
                scale_matrix[0:3, 0:3] = scale_mat
                camera_T_object = np.eye(4)
                camera_T_object[:3,:3] = R
                camera_T_object[:3,3] = T
                seg_mask[masks[:,:,i] > 0] = class_num
                class_num += 1
                masks_list.append(masks[:,:,i])
                abs_poses.append(transform.Pose(camera_T_object=camera_T_object, scale_matrix=scale_matrix))
            obb_datapoint = obb_inputs.compute_nocs_network_targets(masks_list, latent_embeddings, abs_poses,_camera.height, _camera.width)

            color_img = cv2.imread(img_full_path + '_color.png')
            depth_array = np.array(depth, dtype=np.float32)/255.0
            DM = DepthManager()
            noisy_depth  = DM.prepare_depth_data(depth_array)
            stereo_datapoint = datapoint.Stereo(left_color=color_img, right_color=noisy_depth)
            panoptic_datapoint = datapoint.Panoptic(
            stereo=stereo_datapoint,
            depth=noisy_depth,
            segmentation=seg_mask,
            object_poses=[obb_datapoint],
            boxes=[],
            detections=[]
            )
            _DATASET.write(panoptic_datapoint)
            ### Finish writing datapoint

def get_latent_embeddings(point_clouds, estimator):
    latent_embeddings =[]
    for i in range(len(point_clouds)):
        batch_xyz = torch.from_numpy(point_clouds[i]).to(device="cuda", dtype=torch.float)
        batch_xyz = batch_xyz.unsqueeze(0)
        emb, _ = estimator(batch_xyz)
        emb = emb.squeeze(0).cpu().detach().numpy()
        latent_embeddings.append(emb)
    return latent_embeddings


if __name__ == '__main__':
    data_dir = '/home/zubairirshad/object-deformnet/data'
    torch.multiprocessing.set_start_method('spawn')
    emb_dim = 128
    n_cat = 57
    #   n_pts = 1024
    n_pts = 2048
    model_path = str(pathlib.Path(__file__).parent.parent.parent / 'data' / 'auto_encoder_model' / 'model_50_nocs.pth')
    estimator = PointCloudAE(emb_dim, n_pts)
    estimator.cuda()
    estimator.load_state_dict(torch.load(model_path))
    estimator.eval()
    # create list for all data
    #create_img_list(data_dir)
    # annotate dataset and re-write valid data to list

    camera_train = open(os.path.join(data_dir, 'CAMERA', 'train_list_all.txt')).read().splitlines()
    with open(os.path.join(data_dir, 'obj_models/mug_meta_new.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)
    #TEST MODELS
    obj_model_dir = '/home/zubairirshad/object-deformnet/data/obj_models'
    with open(os.path.join(obj_model_dir, 'camera_trainnew.pkl'), 'rb') as f:
        obj_models = cPickle.load(f)

    worker_count = 5
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)

    progress = tqdm(total=len(camera_train))
    def on_complete(*_):
        progress.update()

    valid_img_list = []
    for img_path in camera_train:
        pool.apply_async(annotate_camera_train, args=(img_path, estimator, obj_models, mug_meta), callback=on_complete)
    pool.close()
    pool.join()
    #   annotate_real_train(data_dir, estimator)
    #   annotate_test_data(data_dir, estimator)