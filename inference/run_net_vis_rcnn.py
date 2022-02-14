import argparse
import pathlib
import pprint
from importlib.machinery import SourceFileLoader
import random
import sys
import time
import copy
from simnet.lib import transform
import cv2
import numpy as np
import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from simnet.lib import camera
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import iterative_closest_point
import os
import time
import pytorch_lightning as pl
import math
import _pickle as cPickle
from simnet.lib.net import common
from simnet.lib import datapoint
from simnet.lib.net.panoptic_trainer import PanopticModel
from simnet.lib.net.dataset import Dataset, extract_left_numpy_img
from simnet.lib.net.post_processing import eval3d
from simnet.lib.net.post_processing import pose_outputs
from simnet.lib import occlusions
from simnet.lib import camera
from simnet.lib.net.models.auto_encoder import PointCloudAE
from simnet.lib.net.post_processing import epnp
import glob
import matplotlib.patches as patches
sys.path.append('/home/zubairirshad/object-deformnet/lib')
from align import align_nocs_to_depth
from utils import load_depth
from torchvision import ops
from simnet.lib import color_stuff
import torchvision.transforms as transforms

module_path = os.path.abspath(os.path.join('/home/zubairirshad/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)
from objectron.dataset import iou
from objectron.dataset import box

import matplotlib.pyplot as plt
import colorsys

def visualize_shape(filename,result_dir, shape_list):
    """ Visualization and save image.

    Args:
        name: window name
        shape: list of geoemtries

    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512, left=50, top=25)
    for shape in shape_list:
        vis.add_geometry(shape)
    ctr = vis.get_view_control()
    # ctr.rotate(-300.0, 150.0)
    # if name == 'camera':
    #     ctr.translate(20.0, -20.0)     # (horizontal right +, vertical down +)
    # if name == 'laptop':
    #     ctr.translate(25.0, -60.0)
    vis.run()
    vis.capture_screen_image(os.path.join(result_dir,filename))
    vis.destroy_window()

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

def line_set(points_array):
#   lines = [
#       [0, 1],
#       [0, 2],
#       [1, 3],
#       [2, 3],
#       [4, 5],
#       [4, 6],
#       [5, 7],
#       [6, 7],
#       [0, 4],
#       [1, 5],
#       [2, 6],
#       [3, 7],
#   ]

# 
  open_3d_lines = [
        [0, 1],
        [7,3],
        [1, 3],
        [2, 0],
        [3, 2],
        [0, 4],
        [1, 5],
        [2, 6],
        # [4, 7],
        [7, 6],
        [6, 4],
        [4, 5],
        [5, 7],
    ]
  # colors = [[1, 0, 0] for i in range(len(lines))]
  colors = random_colors(len(open_3d_lines))
  line_set = o3d.geometry.LineSet(
      points=o3d.utility.Vector3dVector(points_array),
      lines=o3d.utility.Vector2iVector(open_3d_lines),
  )
  # print("points", points_array.shape)
  # print("lines", np.array(open_3d_lines).shape)
  # open_3d_lines = np.array(open_3d_lines)
  # line_set = LineMesh(points_array, open_3d_lines,colors=colors, radius=0.001)
  # line_set = line_set.cylinder_segments
  line_set.colors = o3d.utility.Vector3dVector(colors)
  return line_set

from matplotlib.cm import get_cmap
def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor

def depth2inv(depth):
    """
    Invert a depth map to produce an inverse depth map
    Parameters
    ----------
    depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Depth map
    Returns
    -------
    inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Inverse depth map
    """
    inv_depth = 1. / depth.clamp(min=1e-6)
    inv_depth[depth <= 0.] = 0.
    return inv_depth

def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
                  colormap='plasma', filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.
    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization
    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        inv_depth = inv_depth.squeeze(0).squeeze(0)
        # Squeeze if depth channel exists
        # if len(inv_depth.shape) == 3:
        #     inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    print("inv_depth", inv_depth.shape)
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    print("inv depth", inv_depth.shape)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]

def rgbd_size(pc):
    size = 2 * np.amax(np.abs(pc), axis=0)
    return size

def custom_draw_geometry_with_rotation(pcd):
    def rotate_view(vis):
        opt = vis.get_render_option()
        vis.create_window()
        # vis.create_window(window_name=name, width=3000, height=3000)
        opt.background_color = np.asarray([1, 1, 1])
        ctr = vis.get_view_control()
        ctr.rotate(5.0, 0.0)
        # return False
    
    o3d.visualization.draw_geometries_with_animation_callback(pcd,
                                                              rotate_view)

def draw_bboxes(img, img_pts, axes, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    
    # color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    color_ground = (int(color[0]), int(color[1]), int(color[2]))
    
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 3)
    # draw pillars in minor darker color
    # color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    color_pillar = (int(color[0]), int(color[1]), int(color[2]))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 3)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 3)

    # draw axes
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 4) ## y last

    return img

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

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

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

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if not transformation is None:
      source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp])
    return source_temp

def visualize_mrcnn_boxes(detections, img, classes, object_key_to_name, filename):
    # print(classes)
    colors = random_colors(len(classes))
    fig, ax = plt.subplots(1, figsize=(10,7.5))
    plt.axis('off')
    ax.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # ax.imshow(img)
    # plt.show()
    # ax.show(img)
    if detections is not None:
        # unique_labels = detections[:, -1].cpu().unique()
        # n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, len(classes))
        # browse detections and draw bounding boxes
        for (y1, x1, y2, x2), cls_pred,color in zip(detections,classes, bbox_colors):
            box_h = (y2 - y1)
            box_w = (x2 - x1)
            # color = bbox_colors
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                linewidth=6, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            text = object_key_to_name[cls_pred]
            plt.text(x1, y1, s=text, 
                    color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
        plt.axis('off')
        plt.savefig(filename)
        plt.close()    

def visualize(detections, img, classes, seg_mask, object_key_to_name, filename):
    # print(classes)
    colors = random_colors(len(classes))
    fig, ax = plt.subplots(1, figsize=(10,7.5))
    plt.axis('off')
    ax.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # ax.imshow(img)
    # plt.show()
    # ax.show(img)
    if detections is not None:
        # unique_labels = detections[:, -1].cpu().unique()
        # n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, len(classes))
        colored_mask = np.zeros([seg_mask.shape[0], seg_mask.shape[1], 3])
        for ii, color in zip(classes, colors):
            colored_mask[seg_mask == ii, :] = color
        # browse detections and draw bounding boxes
        for (x1, y1, x2, y2), cls_pred, color in zip(detections, classes, bbox_colors):
            box_h = (y2 - y1)
            box_w = (x2 - x1)
            # color = bbox_colors
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            text = object_key_to_name[cls_pred]
            plt.text(x1, y1, s=text, 
                    color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
        plt.axis('off')
        # save image
        plt.imshow(colored_mask, alpha=0.5)
        # plt.show()
        # plt.show()
        plt.savefig(filename)
        plt.close()

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

#./runner.sh app/panoptic_tidying/run_net_vis_rcnn.py @app/panoptic_tidying/net_config.txt --checkpoint [path to checkpoint]
#LOAD REAL DATA
def resize_and_draw(name, img, scale=2):
  dim = (img.shape[1] * scale, img.shape[0] * scale)
  resized_img = cv2.resize(img, dim)
  cv2.imshow(name, img)

def im_resize(img):
  img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
  return img

def resize_upscale(img, scale=2):
  dim = (img.shape[1] * scale, img.shape[0] * scale)
  # resized_img = cv2.resize(img, dim)
  return resized_img

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


# def load_img():
#   left_dir_path = '/home/zubairirshad/Documents/ZED/images_new_2/left'
#   right_dir_path = '/home/zubairirshad/Documents/ZED/images_new_2/right'
  
#   left_object_path = random.choice(glob.glob(left_dir_path + '/*'))
#   right_object_path = random.choice(glob.glob(right_dir_path + '/*'))

#   left_img = cv2.imread(left_object_path)
#   right_img = cv2.imread(right_object_path)

#   # left_img = cv2.imread('/home/zubairirshad/Documents/ZED/images/left/left11.jpeg')
#   # right_img = cv2.imread('/home/zubairirshad/Documents/ZED/images/right/right11.jpeg')

#   left_img = im_resize(left_img)
#   right_img = im_resize(right_img)
#   return left_img, right_img

def load_img_NOCS(color, depth):
  left_img = cv2.imread(color)
  actual_depth = load_depth(depth)
  right_img = np.array(actual_depth, dtype=np.float32)/255.0
  return left_img, right_img, actual_depth

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

def create_anaglyph_w_depth(left_color,right_color ):
  height, width, _ = left_color.shape
  image = np.zeros([height, width, 4], dtype=np.uint8)
  cv2.normalize(left_color, left_color, 0, 255, cv2.NORM_MINMAX)
  # cv2.normalize(stereo_dp.right_color, stereo_dp.right_color, 0, 255, cv2.NORM_MINMAX)
  image[..., 0:3] = left_color
  image = image * 1. / 255.0
  if len(right_color.shape  ) == 2:
    image[..., 3] = right_color
  # print(image.shape)
  image = image.transpose((2, 0, 1))
  return torch.from_numpy(np.ascontiguousarray(image)).float()

def create_anaglyph_norm(left_color,right_color):
  height, width, _ = left_color.shape
  image = torch.zeros(4, height, width, dtype=torch.float32)
  cv2.normalize(left_color, left_color, 0, 255, cv2.NORM_MINMAX)

  rgb = left_color* 1. / 255.0
  norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  rgb = norm(torch.from_numpy(rgb.astype(np.float32).transpose((2,0,1))))

  if len(right_color.shape) == 2:
    depth = right_color
    depth = torch.from_numpy(depth.astype(np.float32))

  image[0:3, :] = rgb
  image[3, :] = depth
  return image

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

def get_scaled_pc(scale_matrix, pc):
  pc_homopoints = camera.convert_points_to_homopoints(pc.T)
  scaled_homopoints = (scale_matrix @ pc_homopoints)
  scaled_homopoints = camera.convert_homopoints_to_points(scaled_homopoints).T
  return scaled_homopoints

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

  # # unit_box_homopoints = camera.convert_points_to_homopoints(epnp._WORLD_T_POINTS.T)
  # unit_box_homopoints = camera.convert_points_to_homopoints(box.T)
  # # morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
  # morphed_box_homopoints = pose.camera_T_object @ unit_box_homopoints
  # morphed_box_points = camera.convert_homopoints_to_points(morphed_box_homopoints).T
  # # box_points.append(morphed_box_points)
  return morphed_pc_homopoints, morphed_box_points, size

def project(K, p_3d):
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
    return projections_2d

def evaluate_dataset(
    hparams,
    input_path,
    output_path,
    min_confidence=0.1,
    overlap_thresh=0.75,
    num_samples=80,
    num_to_draw=20,
    use_gpu=True,
    is_training=False
):
  real_data = True
  model = PanopticModel(hparams, 0, None, None)
  model.eval()
  eval_3d = eval3d.Eval3d()
  if use_gpu:
    model.cuda()
  # get dataset loader
  dataset = Dataset(input_path, hparams)
  print('Dataset size:', len(dataset))
  total_size = num_samples
  total_chamfer_loss = 0
  total_similarity_loss = 0
#   num_samples = len(dataset)
  total_time = 0
  total_shape_time = 0

  emb_dim = 128
  n_cat = 57
  n_pts = 2048
  model_path = str(pathlib.Path(__file__).parent.parent.parent / 'data' / 'auto_encoder_model' / 'model_50_nocs.pth')
  # model_path = '/home/zubairirshad/object-deformnet/results/ae_1024_latent/model_50.pth'
  estimator = PointCloudAE(emb_dim, n_pts)
  estimator.cuda()
  estimator.load_state_dict(torch.load(model_path))
  estimator.eval()

  data_dir = '/home/zubairirshad/object-deformnet/data'

  cam = True
  norm = True
  if cam:
    data_path = open(os.path.join(data_dir, 'CAMERA', 'val_list.txt')).read().splitlines()
    b = np.random.randint(0, 5000, size=250)
    data_path = np.array(data_path)[b].tolist()
    # idx = [129,]
    # data_path = [data_path[i] for i in idx]
    data_type = 'val'
    type = 'val'
    _CAMERA = camera.NOCS_Camera()
    min_confidence = 0.50  
  
  else:
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_scene_3.txt')).read().splitlines()
    # b = np.random.randint(0, 2749, size=250)
    # data_path = data_path[389:]
    # data_path = np.array(data_path)[b].tolist()
    # idx = [68]
    # # idx = [27, 32, 68, 72, 99, 100, 118, 163, 241, 97, 167]
    # data_path = [data_path[i] for i in idx]
    # data_path = data_path[21:23]
    data_type = 'real_test'
    type = 'test'
    _CAMERA = camera.NOCS_Real()
    min_confidence = 0.50

  for i, img_path in enumerate(data_path):
    print("i", i)
    if cam:
      img_full_path = os.path.join(data_dir, 'CAMERA', img_path)
    else:
      img_full_path = os.path.join(data_dir, 'Real', img_path)
    if real_data:
      color_path = img_full_path + '_color.png'
      depth_path = img_full_path + '_depth.png'

      if not os.path.exists(color_path):
        continue
      depth_path = img_full_path + '_depth.png'

      img_path_parsing = img_path.split('/')
      mrcnn_path = os.path.join('/home/zubairirshad/object-deformnet/results/mrcnn_results', data_type, 'results_{}_{}_{}.pkl'.format(
          type, img_path_parsing[-2], img_path_parsing[-1]))

      if not os.path.exists(mrcnn_path):
        continue 
      with open(mrcnn_path, 'rb') as f:
          mrcnn_result = cPickle.load(f)

      # experiment with adding GT result
      with open(img_full_path + '_label.pkl', 'rb') as f:
          gts = cPickle.load(f)

      if cam:
        depth_composed_path = img_path+'_composed.png'
        depth_full_path = os.path.join(data_dir,'camera_composed_depth','camera_full_depths', depth_composed_path)
        print("deth full", depth_full_path)
        
        # depth_full_path = depth_path
      else:
        depth_full_path = depth_path

      if not os.path.exists(depth_full_path):
        continue

      img_vis = cv2.imread(color_path)

      left_linear, depth, actual_depth = load_img_NOCS(color_path, depth_full_path)
      print("depth.shape", depth.shape)
      depth_vis = depth2inv(torch.tensor(depth).unsqueeze(0).unsqueeze(0))
      print("depth_vis", depth_vis.shape)
      depth_vis = viz_inv_depth(depth_vis)
      # plt.imshow(depth_vis)
      # plt.show()
      masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, actual_depth)

      if norm:
        anaglyph = create_anaglyph_norm(left_linear, depth)
      else:
        anaglyph = create_anaglyph_w_depth(left_linear, depth)
      anaglyph = anaglyph[None, :, :, :]
      # print("anaglyph",anaglyph.shape)
    else: 
      sample = dataset[i]
      # anaglyph, _, _, pose_target, _,_, poses, detections_gt, scene_name = sample
      anaglyph, seg_target, depth_target, pose_target, _,_, detections_gt, scene_name = sample
      anaglyph = anaglyph[None, :, :, :]

      # depth_img_target = depth_target.depth_pred
    #   plt.imshow(depth)
    #   plt.show()
    #   plt.axis('off')
    #   plt.grid('off')
    
    if use_gpu:
      anaglyph = anaglyph.to(torch.device('cuda:0'))
    model_start_time = time.time()
    seg_output, depth_output, small_depth, pose_output, box_output, keypoint_outputs = model.forward(anaglyph)

    model_end_time = time.time() - model_start_time
    total_time+=model_end_time
    with torch.no_grad():
      left_image_np = extract_left_numpy_img(anaglyph[0])
      seg_vis_img = seg_output.get_visualization_img(np.copy(img_vis[...,::-1]))
      depth_img = depth_output.get_visualization_img(np.copy(img_vis[...,::-1]))
      category_seg_output = np.ascontiguousarray(seg_output.seg_pred)
      # latent_emb_targets, abs_pose_targets, img_target, img_tar_nms, target_indices = pose_target.compute_pointclouds_and_poses(is_target = True)
      
      latent_emb_outputs, abs_pose_outputs, img_output,scores, output_indices = pose_output.compute_pointclouds_and_poses(min_confidence,is_target = False)

      #all to perform NMS on bounding boxes
      boxes = torch.zeros(len(abs_pose_outputs), 4)
      scores_torch = torch.zeros(len(abs_pose_outputs))
      for p, (pose, score, emb) in enumerate(zip(abs_pose_outputs, scores, latent_emb_outputs)):
        emb = torch.FloatTensor(emb).unsqueeze(0)
        emb = emb.cuda()
        _, shape_out = estimator(None, emb)
        shape_out = shape_out.cpu().detach().numpy()[0]
        bbox = get_2d_box(pose, shape_out, camera_model = _CAMERA)
        image = img_vis[...,::-1].astype(np.uint8).copy() 
        image = cv2.rectangle(image, (int(bbox[0][0]),int(bbox[0][1])) , (int(bbox[1][0]),int(bbox[1][1])), (0, 0, 0), thickness=3)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        boxes[p, :] = torch.tensor([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][0]]) 
        scores_torch[p] = torch.tensor(score)
      keep = ops.nms(boxes, scores_torch, iou_threshold=0.75)

      keep = torch.sort(keep)[0].numpy()

      latent_emb_outputs = np.array(latent_emb_outputs)[keep].tolist()
      output_indices = np.array(output_indices)[keep].tolist()
      abs_pose_outputs = np.array(abs_pose_outputs)[keep].tolist()
      scores= np.array(scores)[keep].tolist()

    # Draw segmentation masks and 2D obbs:
    category_seg_output = np.argmax(category_seg_output[0], axis=0)
    class_ids_predicted = []
    for k in range(len(output_indices)):
      center = output_indices[k]
      class_ids_predicted.append(category_seg_output[center[0], center[1]])

    obj_ids = np.unique(category_seg_output)
    obj_ids = obj_ids[1:]
    masks_target = category_seg_output == obj_ids[:, None, None]
    num_objs = len(obj_ids)
    boxes = [] 
    for m in range(num_objs):
        pos = np.where(masks_target[m])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
    object_key_to_name= {0: 'background', 1:'bottle', 2:'bowl', 3:'camera', 4:'can', 5:'laptop', 6:'mug'}
    path = str(output_path / f'{i}_seg_with_bbox.png')

    visualize(boxes, np.copy(img_vis), obj_ids, category_seg_output, object_key_to_name, path)
    # Draw segmentation masks and 2D obbs:
    depth = np.array(depth, dtype=np.float32)*255.0


    # Get Ground truth pose for every detected 
    gt_ids = np.array(gts['class_ids'])
    gt_boxes = np.array(gts['bboxes'])
    gt_poses = np.array(gts['poses'])
    gt_sizes = np.array(gts['size'])

    path = str(output_path / f'{i}_gt_bbox.png')
    # print("mrcnn_result['bboxes']", mrcnn_result)

    # visualize_mrcnn_boxes(gt_boxes, np.copy(img_vis), gt_ids, path)
    visualize_mrcnn_boxes(mrcnn_result['rois'], np.copy(img_vis), mrcnn_result['class_ids'],object_key_to_name, path)

    if not len(gt_ids):
      continue
    mask_out = []
    num_insts_gt = len(gt_ids)
    index_centers_gt = []
    for m in range(num_insts_gt):
      box = gt_boxes[m]
      center_x_gt = (box[0] + box[2])/2
      center_y_gt = (box[1] + box[3])/2
      index_centers_gt.append([center_x_gt, center_y_gt])

    new_poses = []
    new_ids_gt = []
    new_sizes = []
    index_centers_gt = np.array(index_centers_gt)
    if np.any(np.isnan(index_centers_gt)):
      index_centers_gt = index_centers_gt[~np.any(np.isnan(index_centers_gt), axis=1)]
    for l in range(len(output_indices)):
      point = output_indices[l]
      if len(output_indices) == 0:
        continue
      distances = np.linalg.norm(index_centers_gt-point, axis=1)
      min_index = np.argmin(distances)
      print("distances, min index", distances, min_index)
      new_poses.append(gt_poses[min_index])
      new_ids_gt.append(gt_ids[min_index])
      new_sizes.append(gt_sizes[min_index])
    gt_poses = np.array(new_poses)
    gt_ids = np.array(new_ids_gt)
    gt_sizes = np.array(new_sizes)

    # consider masks from mrcnn_path
    mcrnn_class_ids = np.array(mrcnn_result['class_ids'])
    if not len(mcrnn_class_ids):
      continue
    mask_out = []
    num_insts = len(mrcnn_result['class_ids'])
    for p in range(num_insts):
      mask = np.logical_and(mrcnn_result['masks'][:, :, p], depth > 0)
      mask_out.append(mask)

    mask_out = np.array(mask_out)
    index_centers = []
    for m in range(mask_out.shape[0]):
      pos = np.where(mask_out[m,:,:])
      center_x = np.average(pos[0])
      center_y = np.average(pos[1])
      index_centers.append([center_x, center_y])

    print("class ids output", class_ids_predicted)
    print("class ids mrcnn", mcrnn_class_ids)
    print("class ids GT", np.array(gts['class_ids']))

    new_masks = []
    new_ids = []
    index_centers = np.array(index_centers)
    if np.any(np.isnan(index_centers)):
      index_centers = index_centers[~np.any(np.isnan(index_centers), axis=1)]
    
    mask_out = np.array(mask_out)
    for l in range(len(output_indices)):
      point = output_indices[l]
      if len(output_indices) == 0:
        continue
      distances = np.linalg.norm(index_centers-point, axis=1)
      min_index = np.argmin(distances)
      # if distances[min_index]<28:
      new_masks.append(mask_out[min_index, :,:])
      new_ids.append(mcrnn_class_ids[min_index])
      # else: 
      #   new_masks.append(None)
      #   new_ids.append(class_ids_predicted[l])
    # masks = np.array(new_masks)
    masks =  new_masks
    class_ids = np.array(new_ids)

    print("GT IDS after filtering", gt_ids)
    print("Mask RCNN ids after filtering", class_ids)

    num_objs_new = len(class_ids)
    xmap = np.array([[y for y in range(640)] for z in range(480)])
    ymap = np.array([[z for y in range(640)] for z in range(480)])
    boxes=[]
    for m in range(num_objs_new):
        if masks[m] is not None:
          # pos = np.where(masks[m, :,:]>0)
          print("masks[m]", masks[m].shape)
          pos = np.where(masks[m]>0)
          xmin = np.min(pos[1])
          xmax = np.max(pos[1])
          ymin = np.min(pos[0])
          ymax = np.max(pos[0])
          boxes.append([xmin, ymin, xmax, ymax])
        else:
          boxes.append(None)

    # # #mask vis
    seg_masks =  np.zeros([_CAMERA.height, _CAMERA.width])
    for h in range(len(masks)):
      if masks[h] is not None:
        seg_masks[np.where(masks[h]>0)] = np.array(class_ids)[h]

    def color_img_to_gray(image):
      gray_scale_img = np.zeros(image.shape)
      img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      for i in range(3):
        gray_scale_img[:, :, i] = img
      gray_scale_img[:, :, i] = img
      return gray_scale_img
    
    seg_mask = seg_masks.astype(np.uint8)
    colors = color_stuff.get_panoptic_colors()
    color_img_mask = color_img_to_gray(np.copy(img_vis[...,::-1]))
    for ii, color in zip(range(len(class_ids)), colors):
      colored_mask = np.zeros([seg_mask.shape[0], seg_mask.shape[1], 3])
      colored_mask[seg_mask == ii, :] = color
      color_img_mask = cv2.addWeighted(
          color_img_mask.astype(np.uint8), 0.9, colored_mask.astype(np.uint8), 0.4, 0
      )
    # cv2.imshow('seg mask', color_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rgbd_points = []
    for h in range(num_objs_new):
        if masks[h] is not None:
          n_pts=700
          # sample points
          x1, y1, x2, y2 = boxes[h]
          # mask = masks[h, :, :]>0
          mask = masks[h]>0
          # plt.imshow(mask)
          # plt.show()
          # plt.close()
          mask = np.logical_and(mask, depth > 0)
          # print("mask",mask.shape)
          choose = mask[y1:y2, x1:x2].flatten().nonzero()[0]
          if len(choose) ==0:
            continue
          if len(choose) > n_pts:
              c_mask = np.zeros(len(choose), dtype=int)
              c_mask[:n_pts] = 1
              np.random.shuffle(c_mask)
              choose = choose[c_mask.nonzero()]
          else:
              choose = np.pad(choose, (0, n_pts-len(choose)), 'wrap')
          cam_cx = _CAMERA.c_x
          cam_fx = _CAMERA.f_x
          cam_cy = _CAMERA.c_y
          cam_fy = _CAMERA.f_y

          depth_masked = depth.flatten()[:, np.newaxis]
          xmap_masked = xmap.flatten()[:, np.newaxis]
          ymap_masked = ymap.flatten()[:, np.newaxis]
          # pt2 = depth_masked/1000.0
          pt2 = depth_masked/1000.0
          pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
          pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
          points = np.concatenate((pt0, pt1, pt2), axis=1)
          pcd = o3d.geometry.PointCloud()
          pcd.points = o3d.utility.Vector3dVector(points)
          # custom_draw_geometry_with_rotation([pcd])

          depth_masked = depth[y1:y2, x1:x2].flatten()[choose][:, np.newaxis]
          xmap_masked = xmap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis]
          ymap_masked = ymap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis]
          # pt2 = depth_masked/1000.0
          pt2 = depth_masked/1000.0
          pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
          pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
          points = np.concatenate((pt0, pt1, pt2), axis=1)
          pcd = o3d.geometry.PointCloud()
          pcd.points = o3d.utility.Vector3dVector(points)
          rgbd_points.append(pcd)
        else:
          rgbd_points.append(None)

    # custom_draw_geometry_with_rotation(rgbd_points)

    cv2.imwrite(
        str(output_path / f'{i}_image.png'),
        np.copy(np.copy(img_vis))
    )
    # cv2.imwrite(
    #     str(output_path / f'{i}_mask_rcnn_image.png'),
    #     np.copy(np.copy(color_img_mask))
    # )
    # depth_vis = depth_vis*255.0
    # cv2.imwrite(
    #     str(output_path / f'{i}depth_vis_img.png'),
    #     np.copy(depth_vis[:, :, ::-1])
    # )

    # cv2.imwrite(
    #     str(output_path / f'{i}seg_vis_img.png'),
    #     np.copy(seg_vis_img)
    # )

    # depth_img = depth/1000.0
    # cv2.imwrite(
    #     str(output_path / f'{i}depth_target.png'),
    #     np.copy(depth_img)
    # )

    # cv2.imwrite(
    #     str(output_path / f'{i}_peaks_output.png'),
    #     np.copy(img_output)
    # )

    # # cv2.imwrite(
    # #     str(output_path / f'{i}_peaks_target.png'),
    # #     np.copy(img_target)
    # # )

    # # cv2.imwrite(
    # #     str(output_path / f'{i}_peaks_output_NMS.png'),
    # #     np.copy(img_out_nms)
    # # )

    # peak_img_overlayed = cv2.addWeighted(
    #     img_output.astype(np.uint8), 0.5, np.copy(img_vis).astype(np.uint8), 0.5, 0
    # )

    # peak_img_overlayed_2 = cv2.addWeighted(
    #     img_output.astype(np.uint8), 0.6, np.copy(img_vis).astype(np.uint8), 0.4, 0
    # )

    # peak_img_overlayed_3 = cv2.addWeighted(
    #     img_output.astype(np.uint8), 0.4, np.copy(img_vis).astype(np.uint8), 0.6, 0
    # )

    # cv2.imwrite(
    #     str(output_path / f'{i}peak_img_overlayed.png'),
    #     np.copy(peak_img_overlayed)
    # )

    # cv2.imwrite(
    #     str(output_path / f'{i}peak_img_overlayed_2.png'),
    #     np.copy(peak_img_overlayed_2)
    # )

    # cv2.imwrite(
    #     str(output_path / f'{i}peak_img_overlayed_3.png'),
    #     np.copy(peak_img_overlayed_3)
    # )

    shape_outputs = []
    shape_targets = []
    shape_actuals = []
#----------------------------------------------------------------
#Measuring chamfer distance
#-------------------------------------------------------
    # if actual_pointclouds and len(actual_pointclouds)<len(latent_emb_targets):
    #     total_size -= 1
    # else:
    # for z in range(len(latent_emb_targets)):
    #   if len(indices_list) == 0:
    #     continue
    #   if indices_list[z]:
    #     continue
      
    #   pose_out = filtered_pose_outputs[z]
    #   pose_tar = abs_pose_targets[z]
    #   emb_target = latent_emb_targets[z]
    #   emb_output = filtered_latent_emb_outputs[z]

    #   print("-------------------------------\n")
    #   s1= pose_out.scale_matrix[0,0]
    #   s2 = pose_tar.scale_matrix[0,0]
      
    #   rot_out = pose_out.camera_T_object[:3,:3]
    #   rot_tar = pose_tar.camera_T_object[:3,:3]

    #   T1 = pose_out.camera_T_object[:,3]
    #   T2 = pose_tar.camera_T_object[:,3]


    #   R_error, T_error, IOU = compute_sRT_errors(s1, rot_out, T1, s2, rot_tar, T2)

    #   print("R_error, T error, IOU", R_error, T_error, IOU)
      # print("error x", np.linalg.norm(pose_out.camera_T_object[0,3] - pose_tar.camera_T_object[0,3]))
      # print("error y", np.linalg.norm(pose_out.camera_T_object[1,3] - pose_tar.camera_T_object[1,3]))
      # print("error z", np.linalg.norm(pose_out.camera_T_object[2,3] - pose_tar.camera_T_object[2,3]))
      # print("error scale", np.linalg.norm(pose_out.scale_matrix[0,0] - pose_tar.scale_matrix[0,0]))


      # print("geodesic distance between rotations", compute_geodesic_distance_from_two_matrices(rot_out, rot_tar))

      # print("-----------------\n")

    #   # if actual_pointclouds:
    #   #   emb_actual = torch.FloatTensor(actual_pointclouds[z]).unsqueeze(0)
    #   # else:
    #   #   emb_actual = []

    #   emb_target = torch.FloatTensor(emb_target).unsqueeze(0)
    #   emb_target = emb_target.cuda()
    #   _, shape_out_target = estimator(None, emb_target)

    #   emb_output = torch.FloatTensor(emb_output).unsqueeze(0)
    #   emb_output = emb_output.cuda()
    #   _, shape_out_output = estimator(None, emb_output)
    #   shape_outputs.append(shape_out_output)
    #   shape_targets.append(shape_out_target)
      # shape_actuals.append(emb_actual)

  # if len(shape_targets) == 0:
  #   total_size -= 1
  # else:
  #   shape_outputs = torch.cat((shape_outputs), dim=0)
  #   shape_actuals = torch.cat((shape_actuals), dim=0).cuda()

  #   shape_targets = torch.cat((shape_targets), dim=0)
  #   # loss, _ = chamfer_distance(shape_outputs, shape_targets)
  #   loss, _ = chamfer_distance(shape_outputs, shape_actuals)
  #   loss_2, _ = chamfer_distance(shape_targets, shape_actuals)
  #   total_chamfer_loss += loss.item()
  #   total_similarity_loss +=loss_2.item()

    # print("loss per image", loss.item())
    # print("similairty loss per image", loss_2.item())

      # print("Done eval episode:", i)
      # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
      # print("\n")
    

#----------------------------------------------------------------
#Perform post processing to get pose (where only pc are predicted) (hacky way to do it)
#-------------------------------------------------------

    # nms_latent_emb = []
    # print("len latent", len(latent_emb_outputs))
    # for j in range(len(latent_emb_outputs)):
    #   if j ==2 or j==3 or j==5:
    #     continue
    #   else:
    #     nms_latent_emb.append(latent_emb_outputs[j])
    # # poses[6], poses[7] = poses[7], poses[6]
    
    # poses_new = []
    # for j in range(len(abs_pose_outputs)):
    #   if j ==2 or j==3 or j==5:
    #     continue
    #   else:
    #     poses_new.append(abs_pose_outputs[j])
    # latent_emb_outputs = nms_latent_emb 
    # abs_pose_outputs = poses_new
    # print("len latent emb", len(latent_emb_outputs))
    # print("len poses", len(abs_pose_outputs))

#----------------------------------------------------------------
#Save Actual pointclouds with GT pose
#-------------------------------------------------------
    # box_obb_target= []
    # for j in range(len(latent_emb_targets)):
    #   emb = latent_emb_targets[j]
    #   pose = abs_pose_targets[j]
    #   emb = torch.FloatTensor(emb).unsqueeze(0)
    #   emb = emb.cuda()
    #   _, shape_out = estimator(None, emb)
    #   shape_out = shape_out.cpu().detach().numpy()[0]
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(shape_out)
    #   filename = str(output_path) + '/original_target'+str(i)+str(j)+'.ply'
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(shape_out)
    #   o3d.io.write_point_cloud(filename, pcd)

    #   rotated_pc, rotated_box = get_gt_pointclouds(pose, shape_out, camera_model = _CAMERA)
    #   filename = str(output_path) + '/rotated_target'+ str(i)+str(j)+'.ply'
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(rotated_pc)
    #   o3d.io.write_point_cloud(filename, pcd)

    #   obb = pcd.get_oriented_bounding_box().get_box_points()
    #   points_obb = camera.convert_points_to_homopoints(np.array(obb).T)
    #   points_2d_obb = project(_CAMERA.K_matrix, points_obb)
    #   points_2d_obb = points_2d_obb.T
    #   box_obb_target.append(points_2d_obb)

    #   filename = str(output_path) + '/box_target'+str(i)+str(j)+'.ply'
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(rotated_box)
    #   o3d.io.write_point_cloud(filename, pcd)

    # result = {}
    # with open(img_full_path + '_label.pkl', 'rb') as f:
    #     gts = cPickle.load(f)
    # gt_poses = gts['poses']
    # gt_sizes = gts['size']
    # gt_class_ids = gts['class_ids']

    # print("gt class_ids", gt_class_ids)
    # print("gt poses", gt_poses)

    # gt_pcd_array = []
    # gt_box_obb = []
    
    # for j in range(len(gt_poses)):
    #   if j>= len(latent_emb_outputs):
    #     continue
    #   emb = latent_emb_outputs[j] 
    #   pose = gt_poses[j]
    #   size = gt_sizes[j]
    #   emb = torch.FloatTensor(emb).unsqueeze(0)
    #   emb = emb.cuda()
    #   _, shape_out = estimator(None, emb)
    #   shape_out = shape_out.cpu().detach().numpy()[0]
    #   gt_rotated_pc, gt_rotated_box, sizes = get_gt_pointclouds(pose, shape_out, camera_model = _CAMERA, sizes = size)
      
    #   filename = str(output_path) + '/rotated_output'+ str(i)+str(j)+'.ply'
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(gt_rotated_pc)
    #   o3d.io.write_point_cloud(filename, pcd)
      
    #   #2D output
    #   obb = gt_rotated_box
    #   # obb = pcd.get_oriented_bounding_box().get_box_points()
    #   points_obb = camera.convert_points_to_homopoints(np.array(obb).T)
    #   points_2d_obb = project(_CAMERA.K_matrix, points_obb)
    #   points_2d_obb = points_2d_obb.T
    #   gt_box_obb.append(points_2d_obb)
    #   points_mesh = camera.convert_points_to_homopoints(np.array(pcd.points).T)
    #   points_2d_mesh = project(_CAMERA.K_matrix, points_mesh)
    #   points_2d_mesh = points_2d_mesh.T
    #   gt_pcd_array.append(points_2d_mesh)

    #   filename = str(output_path) + '/box_output'+str(i)+str(j)+'.ply'
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(gt_rotated_box)
    #   o3d.io.write_point_cloud(filename, pcd)

    #   # filename = str(output_path) + '/mesh_frame'+str(i)+str(j)+'.ply'
    #   # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #   # mesh_t = copy.deepcopy(mesh_frame)
    #   # T = pose.camera_T_object
    #   # mesh_t = mesh_t.transform(T)
    #   # o3d.io.write_triangle_mesh(filename,mesh_t)

    shape_start_time = time.time()
    box_obb=[]
    pcd_array= []
    pcd_3d = []
    box_abb = []
    rgbd_array=[]
    gt_array=[]
    gt_box_obb = []
    axes=[]
    # eval_ids = [1,4,5,6]
    eval_ids = [7]

    f_sRT = np.zeros((len(latent_emb_outputs), 4, 4), dtype=float)
    f_size = np.zeros((len(latent_emb_outputs), 3), dtype=float)
    colors_mesh = [(213,185,184), (150,194,187), (125,125,186), (191,183,147), (186,186,214), (234,218,218)]
    colors_mesh = np.array(colors_mesh).astype(np.float32)
    print("len latent emb", len(latent_emb_outputs))
    print("len rgbd points", len(rgbd_points))
    for j in range(len(latent_emb_outputs)):
      emb = latent_emb_outputs[j]
      pose = abs_pose_outputs[j]
      emb = torch.FloatTensor(emb).unsqueeze(0)
      emb = emb.cuda()
      _, shape_out = estimator(None, emb)
      shape_out = shape_out.cpu().detach().numpy()[0]

      filename = str(output_path) + '/original_output'+ str(i)+str(j)+'.ply'
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(shape_out)
      o3d.io.write_point_cloud(filename, pcd)
      
      # if class_ids[j] == 2 or class_ids[j] == 5:
      #   threshold = 0.3
      # else:
      #   # threshold = 0.005
      #   threshold = 0.02
      #upsample the pointclouds to 20000 points
      # pcd = o3d.geometry.PointCloud()
      # pcd.points = o3d.utility.Vector3dVector(shape_out)
      # single_pcd_original = pcd
      # single_pcd_original.normals = o3d.utility.Vector3dVector(np.zeros(
      # (1, 3))) 
      # single_pcd_original.estimate_normals()
      # # o3d.geometry.estimate_normals(
      # #     single_pcd_original,
      # #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
      # #                                                       max_nn=30))
      # # single_pcd_original.orient_normals_consistent_tangent_plane(1000)
      # alpha = 0.08
      # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(single_pcd_original, alpha)
      # mesh.compute_vertex_normals()
      # pcd = mesh.sample_points_poisson_disk(20000)
      # shape_out = np.array(pcd.points)
      # o3d.visualization.draw_geometries([pcd])
      
      #0.5 for CAM, 0.1 for REAL
      # threshold = 0.5
      # threshold = 0.1
      if class_ids[j]== 5:
        threshold = 0.08
      else:
        threshold = 0.01
      trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
      rotated_pc, rotated_box, size = get_gt_pointclouds(pose, shape_out, camera_model = _CAMERA)
      print("pointcloud size", size)
      # rgbd_size = get_rgbd_size
      filename = str(output_path) + '/rotated_output'+ str(i)+str(j)+'.ply'
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(rotated_pc)
      o3d.io.write_point_cloud(filename, pcd)

      rotated_pc_gt, rotated_box_gt, size_gt = get_gt_pointclouds(gt_poses[j], shape_out, camera_model = _CAMERA, sizes = gt_sizes[j])
      if rgbd_points[j] is not None:
        rgbd_p = np.array(rgbd_points[j].points)

      print("translation error pose before", np.linalg.norm(gt_poses[j][:3,3] - pose.camera_T_object[:3,3]) * 100)
      print("translation error mean before", np.linalg.norm(np.mean(rotated_pc_gt, axis=0) - np.mean(rotated_pc, axis=0)) * 100)
      # print("translation error mean b/w rgbd before", np.linalg.norm(np.mean(rgbd_p, axis=0) - np.mean(rotated_pc, axis=0)) * 100)
      print("-----------------------")
      
      # #not enough masks available for target object, can't do ICP
      if class_ids[j] in eval_ids:
        if rgbd_points[j] is not None:
        # if j<len(rgbd_points):
          icp_time = time.time()
          # # Perform ICP on source rgbd_points
          rgbd_point = rgbd_points[j]
          if not cam:
            print("Statistical oulier removal")
            cl, ind = rgbd_point.remove_radius_outlier(nb_points=150, radius=0.1)
            rgbd_point = rgbd_point.select_by_index(ind)

          source_temp= draw_registration_result(pcd, rgbd_point, trans_init)
          # reg_p2l = o3d.pipelines.registration.registration_icp(
          #           pcd, rgbd_point, threshold, trans_init,
          #           o3d.pipelines.registration.TransformationEstimationPointToPoint(),
          #           o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

          reg_p2l = o3d.pipelines.registration.registration_icp(
                    rgbd_point, pcd, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
          # print(reg_p2l)
          # source = torch.Tensor(np.array(pcd.points)).unsqueeze(0).cuda()
          # target = torch.Tensor(np.array(rgbd_point.points)).unsqueeze(0).cuda()
          # ICPSolution = iterative_closest_point(source, target, max_iterations=1000)
          # print("converged, rmse", ICPSolution.converged, ICPSolution.rmse)
          # p_R= ICPSolution.RTs.R
          # p_T = ICPSolution.RTs.T
          # RT = np.eye(4)
          # RT[:3,:3] = p_R.squeeze(0).cpu().numpy()
          # RT[:3,:3] = p_T.squeeze(0).cpu().numpy()
          
          # source_temp = draw_registration_result(pcd, rgbd_point, reg_p2l.transformation)
          source_temp = draw_registration_result(pcd, rgbd_point, np.linalg.inv(reg_p2l.transformation))
          pose.camera_T_object = np.linalg.inv(reg_p2l.transformation) @ pose.camera_T_object
          # pcd = o3d.geometry.PointCloud()
          # pcd.points = o3d.utility.Vector3dVector(ICPSolution.Xt.squeeze(0).cpu().numpy())
          # source_temp = draw_registration_result(pcd, rgbd_point, trans_init)
          # pose.camera_T_object = RT @ pose.camera_T_object
          rotated_pc, rotated_box, _ = get_gt_pointclouds(pose, shape_out, camera_model = _CAMERA)
          filename = str(output_path) + '/rotated_output'+ str(i)+str(j)+'.ply'
          pcd = o3d.geometry.PointCloud()
          pcd.points = o3d.utility.Vector3dVector(rotated_pc)
          o3d.io.write_point_cloud(filename, pcd)
          pcd = source_temp
          print("ICP time per iteration", time.time()-icp_time)
      # pcd.paint_uniform_color(colors_mesh[j]/255.0)

      f_size[j] = size
      f_sRT[j] = pose.camera_T_object @ pose.scale_matrix
      # source_temp = draw_registration_result(pcd, rgbd_points[j], trans_init)
      # if rgbd_points[j] is not None:
      print("translation error pose after", np.linalg.norm(gt_poses[j][:3,3] - pose.camera_T_object[:3,3]) * 100)
      print("translation error mean after", np.linalg.norm(np.mean(rotated_pc_gt, axis=0) - np.mean(rotated_pc, axis=0)) * 100)
        # print("translation error mean b/w rgbd after", np.linalg.norm(np.mean(rgbd_p, axis=0) - np.mean(rotated_pc, axis=0)) * 100)
        # print("================================\n\n")

      # 2D axangles
      xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
      transformed_axes = transform_coordinates_3d(xyz_axis, f_sRT[j])
      projected_axes = calculate_2d_projections(transformed_axes, _CAMERA.K_matrix[:3,:3])
      axes.append(projected_axes)

      # GT 2D output
      obb = rotated_box_gt
      # obb = pcd.get_oriented_bounding_box().get_box_points()
      points_obb = camera.convert_points_to_homopoints(np.array(obb).T)
      points_2d_obb = project(_CAMERA.K_matrix, points_obb)
      points_2d_obb = points_2d_obb.T
      gt_box_obb.append(points_2d_obb)

      #2D output
      obb = rotated_box
      # obb = pcd.get_oriented_bounding_box().get_box_points()
      points_obb = camera.convert_points_to_homopoints(np.array(obb).T)
      points_2d_obb = project(_CAMERA.K_matrix, points_obb)
      points_2d_obb = points_2d_obb.T
      box_obb.append(points_2d_obb)
      
      points_mesh = camera.convert_points_to_homopoints(np.array(pcd.points).T)
      points_2d_mesh = project(_CAMERA.K_matrix, points_mesh)
      points_2d_mesh = points_2d_mesh.T
      pcd_array.append(points_2d_mesh)

      # points_rgbd_mesh = camera.convert_points_to_homopoints(np.array(rgbd_points[j].points).T)
      # points_rgbd_mesh = project(_CAMERA.K_matrix, points_rgbd_mesh)
      # points_rgbd_mesh = points_rgbd_mesh.T
      # rgbd_array.append(points_rgbd_mesh)

      points_gt_mesh = camera.convert_points_to_homopoints(np.array(rotated_pc_gt.T))
      points_gt_mesh = project(_CAMERA.K_matrix, points_gt_mesh)
      points_gt_mesh = points_gt_mesh.T
      gt_array.append(points_gt_mesh)

      # print("translation error 2D silhouttes mean", np.linalg.norm(np.mean(points_2d_mesh, axis=0) - np.mean(points_gt_mesh, axis=0)))
      # print("translation error 2D silhouttes mean with RGBD", np.linalg.norm(np.mean(points_2d_mesh, axis=0) - np.mean(points_rgbd_mesh, axis=0)))

      filename = str(output_path) + '/rotated_output'+ str(i)+str(j)+'.ply'
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(rotated_pc)
      o3d.io.write_point_cloud(filename, pcd)
      rotated_pc = pcd

      filename = str(output_path) + '/box_output'+str(i)+str(j)+'.ply'
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(rotated_box)
      o3d.io.write_point_cloud(filename, pcd)

      filename = str(output_path) + '/mesh_frame'+str(i)+str(j)+'.ply'
      mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
      mesh_t = copy.deepcopy(mesh_frame)
      T = pose.camera_T_object
      mesh_t = mesh_t.transform(T)
      o3d.io.write_triangle_mesh(filename,mesh_t)
      pcd_3d.append(mesh_t)
      pcd_3d.append(line_set(rotated_box))
      pcd_3d.append(pcd)
      pcd_3d.append(rotated_pc)

    # print("===============\n")

    # print("predicted class ids", class_ids)
    # print("predicted poses", f_sRT)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # # vis.create_window(window_name=name, width=1920, height=1080)
    # opt.background_color = np.asarray([1, 1, 1])
    # opt.point_size = 10

    name = str(i)

    filename_pc = str(output_path) + '/pc'+str(i)+'.png'

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # # vis.create_window(window_name=name, width=1920, height=1080)
    # opt.background_color = np.asarray([1, 1, 1])
    # opt.point_size = 10

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # pcd_3d.append(mesh_frame)

    for shape in pcd_3d:
        vis.add_geometry(shape)
    opt = vis.get_render_option()
    # vis.create_window(window_name=name, width=1920, height=1080)
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 5
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("/home/zubairirshad/rgbd/fleet/simnet/ScreenCamera_2021-09-20-19-02-34.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    vis.run()
    vis.capture_screen_image(filename_pc, do_render=True)
    vis.destroy_window()
    del opt
    del ctr
    del vis
    

    # def rotate_view(vis):
    #     opt = vis.get_render_option()
    #     vis.create_window()
    #     # vis.create_window(window_name=name, width=1920, height=1080)
    #     opt.background_color = np.asarray([1, 1, 1])
    #     opt.point_size = 15
    #     ctr = vis.get_view_control()
    #     parameters = o3d.io.read_pinhole_camera_parameters("/home/zubairirshad/rgbd/fleet/simnet/ScreenCamera_2021-09-18-20-10-54.json")
    #     ctr.convert_from_pinhole_camera_parameters(parameters)

    # o3d.visualization.draw_geometries_with_animation_callback(pcd_3d, rotate_view)

    shape_end_time = time.time() - shape_start_time
    total_shape_time+=shape_end_time

    # if not actual_pointclouds:
    #   continue
    # for j in range(len(actual_pointclouds)):
    #   shape_out = actual_pointclouds[j]
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(shape_out)
    #   filename = str(output_path) + '/original_actual'+str(i)+str(j)+'.ply'
    #   pcd = o3d.geometry.PointCloud()
    #   pcd.points = o3d.utility.Vector3dVector(shape_out)
    #   o3d.io.write_point_cloud(filename, pcd)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (0,255,255)
    lineType               = 1

    visualize_2d=True
    # colors_box = [(197, 219, 38), (206, 252, 62), (63, 237, 234)]
    colors_box = [(63, 237, 234)]
    im = np.array(np.copy(img_vis)).copy()
    if visualize_2d:
      for k in range(len(colors_box)):
        for points_2d, id, axis in zip(box_obb, class_ids, axes):
          points_2d = np.array(points_2d)
          
          im = draw_bboxes(im, points_2d, axis, colors_box[k])
          # name = object_key_to_name[int(id)]
          # print("name", name)
          # x = np.int32(points_2d[0])[0]
          # y = np.int32(points_2d[0])[1]
          # point = (x,y)
          # cv2.putText(im,name, 
          #     point, 
          #     font, 
          #     fontScale,
          #     (255, 0, 0),
          #     lineType)

        # for points_2d, id in zip(gt_box_obb, gt_ids):
        #   # print("id", id)
        #   points_2d = np.array(points_2d)
        #   # print("points 2d -1",points_2d[-1])
        #   im = draw_bboxes(im, points_2d, (0, 0, 255))
        #   name = object_key_to_name[int(id)] +'GT'
        #   # print("name", name)
        #   x = np.int32(points_2d[0])[0]
        #   y = np.int32(points_2d[0])[1]
        #   point = (x,y)
        #   cv2.putText(im,name, 
        #       point, 
        #       font, 
        #       fontScale,
        #       (0,0,255),
        #       lineType)

        box_plot_name = str(output_path)+'/box3d'+str(i)+str(k)+'.png'
        cv2.imwrite(
            box_plot_name,
            np.copy(im)
        )
        fig, ax1 = plt.subplots(1, figsize=(10,7.5))
        plt.axis('off')

        color_img = np.array(np.copy(img_vis)).copy()
        # color_img = np.copy(img_vis)
        # color_img = color_img/255.0
        plt.xlim((0, color_img.shape[1]))
        plt.ylim((0, color_img.shape[0]))
        # Projections
        color = ['g', 'y', 'b', 'r', 'm', 'c', '#3a7c00', '#3a7cd9', '#8b7cd9', '#211249']
        for g, points_2d_mesh in enumerate(pcd_array):
            plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[g], s=2)

        plt.gca().invert_yaxis()

        # plt.plot(axis[0][0], axis[0][1], axis[1][0], axis[1][1],'g-',linewidth=4)
        # plt.plot(axis[0][0], axis[0][1], axis[3][0], axis[3][1],'r-',linewidth=4)
        # plt.plot(axis[0][0], axis[0][1], axis[2][0], axis[2][1],'b-',linewidth=4)

        # color_img = cv2.arrowedLine(color_img, tuple(axis[0]), tuple(axis[1]), (0, 0, 255), 4)
        # color_img = cv2.arrowedLine(color_img, tuple(axis[0]), tuple(axis[3]), (255, 0, 0), 4)
        # color_img = cv2.arrowedLine(color_img, tuple(axis[0]), tuple(axis[2]), (0, 255, 0), 4)

        color_img = color_img/255.0
        ax1.imshow(color_img[:, :, ::-1])
        plt_name = str(output_path)+'/plot_w_box_mesh'+str(i)
        plt.savefig(plt_name, bbox_inches='tight')
        plt.close()

        # fig, ax1 = plt.subplots(1, figsize=(10,7.5))
        # plt.axis('off')
        # color_img = np.copy(img_vis)
        # color_img = color_img/255.0
        # plt.xlim((0, color_img.shape[1]))
        # plt.ylim((0, color_img.shape[0]))
        # # Projections
        # color = ['g', 'y', 'b', 'r', 'm', 'c', '#3a7c00', '#3a7cd9', '#8b7cd9', '#211249']
        # for g, points_2d_mesh in enumerate(rgbd_array):
        #     plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[g], s=2)
        # plt.gca().invert_yaxis()
        # ax1.imshow(color_img)
        # plt_name = str(output_path)+'/plot_w_rgbd_mesh'+str(i)
        # plt.savefig(plt_name, bbox_inches='tight')
        # plt.close()

        fig, ax1 = plt.subplots(1, figsize=(10,7.5))
        plt.axis('off')
        color_img = np.copy(img_vis)
        color_img = color_img/255.0
        plt.xlim((0, color_img.shape[1]))
        plt.ylim((0, color_img.shape[0]))
        # Projections
        color = ['g', 'y', 'b', 'r', 'm', 'c', '#3a7c00', '#3a7cd9', '#8b7cd9', '#211249']
        for g, points_2d_mesh in enumerate(gt_array):
            plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[g], s=2)
        plt.gca().invert_yaxis()
        ax1.imshow(color_img)
        plt_name = str(output_path)+'/plot_w_gt_mesh'+str(i)
        plt.savefig(plt_name, bbox_inches='tight')
        plt.close()




        # for g, points_2d_mesh in enumerate(gt_pcd_array):
        #     plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[g], s=2)
        # plt.gca().invert_yaxis()
        # ax1.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        # plt_name = str(output_path)+'/plot_w_box_mesh_gt'+str(i)
        # plt.savefig(plt_name, bbox_inches='tight')
        # plt.close()

        # fig, ax2 = plt.subplots(1, figsize=(10,7.5))
        # plt.axis('off')
        # plt.xlim((0, color_img.shape[1]))
        # plt.ylim((0, color_img.shape[0]))
        # for points_2d in box_obb:
        #     for edge in open_3d_lines:
        #         plt.plot(points_2d[edge, 0], points_2d[edge, 1], color='b', linewidth=1.0)
        # plt.gca().invert_yaxis()
        # ax2.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        # plt_name = str(output_path)+'/plot_w_box_obb'+str(i)
        # plt.axis('off')
        # plt.savefig(plt_name, bbox_inches='tight')
        # plt.close()

        # fig, ax2 = plt.subplots(1, figsize=(10,7.5))
        # plt.axis('off')
        # plt.xlim((0, color_img.shape[1]))
        # plt.ylim((0, color_img.shape[0]))
        # for points_2d in box_obb_target:
        #     for edge in open_3d_lines:
        #         plt.plot(points_2d[edge, 0], points_2d[edge, 1], color='b', linewidth=1.0)
        # plt.gca().invert_yaxis()
        # ax2.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        # plt_name = str(output_path)+'/plot_w_box_obb_target'+str(i)
        # plt.axis('off')
        # plt.savefig(plt_name, bbox_inches='tight')
        # plt.close()

        # fig, ax3 = plt.subplots(1, figsize=(10,7.5))
        # plt.axis('off')
        # plt.xlim((0, color_img.shape[1]))
        # plt.ylim((0, color_img.shape[0]))
        # for points_2d in box_abb:
        #     for edge in edges_corners:
        #         plt.plot(points_2d[edge, 0], points_2d[edge, 1], color='b', linewidth=1.0)
        # plt.gca().invert_yaxis()
        # ax3.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        # plt_name = str(output_path)+'/plot_w_box_abb'+str(i)
        # plt.axis('off')
        # plt.savefig(plt_name, bbox_inches='tight')
        # plt.close()
        # plt.show()

    print("shape time per image",shape_end_time)
    print("model time per image", model_end_time)
    print("total inference time per image", shape_end_time+model_end_time)
    
    print("Done eval episode:", i)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
    print("\n")
    # if i ==50:
    #   break

  time_per_image = total_time/num_samples
  shape_time_per_image = total_shape_time/num_samples
  print(" inference time per image", time_per_image)
  print("shape time per image", shape_time_per_image)
  print("ALL CHAMFER DISTANCE", total_chamfer_loss/total_size)
  print("ALL CHAMFER DISTANCE b/w Actual and Target", total_similarity_loss/total_size)
  eval_3d.process_dataset(output_path / 'results_val')
  print("ALL 3D METRICS ", eval_3d.process_all_3D_dataset())
  print("ALL KD 3D METRICS ", eval_3d.process_all_3D_kd_dataset())


if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  app_group = parser.add_argument_group('app')
  app_group.add_argument('--app_output', default='validation_inference', type=str)
  app_group.add_argument(
      '--is_training', default=False, type=bool, help='Is tested on validation data or not.'
  )
  hparams = parser.parse_args()
  overlap_thresh = [0.6, 0.7, 0.75, 0.8, 0.9]
  confidences = [0.01, 0.05, 0.1, 0.2, 0.3]
  results = []

  #for confidence in confidences:
  result_name = 'rgbd_nocs_Cam_Vis'
  result_num = 'norm_23'
  path = 'data/simnet.debug/rgbd_nocs_'+result_name+str(result_num)
  output_path = pathlib.Path(path) / hparams.app_output
  output_path.mkdir(parents=True, exist_ok=True)
  results.append(
      evaluate_dataset(hparams, hparams.val_path, output_path, is_training=hparams.is_training)
  )

  print(overlap_thresh)
  print(results)

  print('\n\n\nHParams used:')
  pprint.pprint(hparams)
  print('\nImages dir:', output_path)
