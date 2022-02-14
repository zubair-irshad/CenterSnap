import open3d as o3d
import os
from pyntcloud import PyntCloud
import numpy as np
import colorsys
import pathlib
import trimesh
from simnet.lib import transform
from pytorch3d.loss import chamfer_distance
import torch
from open3d import *
import cv2
import numpy as np
import trimesh
from trimesh.transformations import transform_points
from simnet.lib import camera
from PIL import Image
from pyobb.obb import OBB

from app.panoptic_tidying.line_set import LineMesh

np.set_printoptions(threshold=np.inf)
from pypoisson import poisson_reconstruction
# from ply_from_array import points_normals_from, ply_from_array

import numpy as np
_CAMERA = camera.NOCS_Real()
import matplotlib.pyplot as plt
# __all__ = ["points_normals_from" ,"ply_from_array"]


def points_normals_from(filename):
    array = np.genfromtxt(filename)
    return array[:,0:3], array[:,3:6]

def ply_from_array(points, faces, output_file):

    num_points = len(points)
    num_triangles = len(faces)

    header = """ply
    format ascii 1.0
    element vertex {0}
    property float x
    property float y
    property float z
    element face {1}
    property list uchar int vertex_indices
    end_header\n""".format(num_points, num_triangles)
    

    with open(output_file,'wb') as f:
        f.write(header.encode())
        for idx, item in enumerate(points):
            f.write("{0:0.6f} {1:0.6f} {2:0.6f}\n".format(item[0],item[1], item[2]).encode())

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row.encode())

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


def create_sphere():
    obj_trimesh = trimesh.creation.uv_sphere()

    # Translated object to center.
    center_to_origin = np.eye(4)
    center_to_origin[0:3, 3] = -transform.compute_trimesh_centroid(obj_trimesh)
    obj_trimesh.apply_transform(center_to_origin)
    # Set largest dimension so that object is contained in a unit cube.
    bounding_box = obj_trimesh.bounds
    current_scale = np.array([
        bounding_box[1][0] - bounding_box[0][0], bounding_box[1][1] - bounding_box[0][1],
        bounding_box[1][2] - bounding_box[0][2]
    ])
    scale_matrix = np.eye(4)
    scale_matrix[0:3, 0:3] = scale_matrix[0:3, 0:3] * 1.0 / np.max(current_scale)
    obj_trimesh.apply_transform(scale_matrix)

    sampled_pc = trimesh.sample.sample_surface(obj_trimesh, 1024)
    return sampled_pc[0]


def unit_cube():
  points = np.array([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
  ]) - 0.5
  lines = [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 3],
      [4, 5],
      [4, 6],
      [5, 7],
      [6, 7],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
  ]

  # colors = [[1, 0, 0] for i in range(len(lines))]
  colors = random_colors(len(lines))
#   line_set = o3d.geometry.LineSet(
#       points=o3d.utility.Vector3dVector(points),
#       lines=o3d.utility.Vector2iVector(lines),
#   )
#   line_set.colors = o3d.utility.Vector3dVector(colors)

  line_set = LineMesh(points, lines,colors=colors, radius=0.0008)
  line_set = line_set.cylinder_segments
  return line_set

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
#   line_set = o3d.geometry.LineSet(
#       points=o3d.utility.Vector3dVector(points_array),
#       lines=o3d.utility.Vector2iVector(open_3d_lines),
#   )
  print("points", points_array.shape)
  print("lines", np.array(open_3d_lines).shape)
  open_3d_lines = np.array(open_3d_lines)
  line_set = LineMesh(points_array, open_3d_lines,colors=colors, radius=0.001)
  line_set = line_set.cylinder_segments
#   line_set.colors = o3d.utility.Vector3dVector(colors)
  return line_set

def custom_draw_geometry_load_option(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for shape in pcd:
        vis.add_geometry(shape)
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("/home/zubairirshad/fleet/simnet/data/simnet.debug/rgbd_nocs_REAL_val_TEST/validation_inference/params.json")
    opt = vis.get_render_option()
    opt.background_color = np.asarray([245/255.0, 255/255.0, 246/255.0])
    vis.run()
    vis.destroy_window()

def custom_draw_geometry_with_rotation(pcd):
    def rotate_view(vis):
        opt = vis.get_render_option()
        vis.create_window()
        # vis.create_window(window_name=name, width=1920, height=1080)
        opt.background_color = np.asarray([1, 1, 1])
        ctr = vis.get_view_control()
        ctr.rotate(1.0, 0.0)
        # return False
    
    o3d.visualization.draw_geometries_with_animation_callback(pcd,
                                                              rotate_view)

def visualize_shape(name, type, shape_list, result_dir):
    """ Visualization and save image.

    Args:
        name: window name
        shape: list of geoemtries

    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=512, height=512, left=50, top=25)
    for shape in shape_list:
        vis.add_geometry(shape)
    ctr = vis.get_view_control()
    ctr.rotate(-300.0, 150.0)
    # if name == 'camera':
    #     ctr.translate(20.0, -20.0)     # (horizontal right +, vertical down +)
    # if name == 'laptop':
    #     ctr.translate(25.0, -60.0)
    vis.run()
    file_name = name+type+'.png'
    vis.capture_screen_image(os.path.join(result_dir,file_name))
    vis.destroy_window()
    
# pointcloud = PyntCloud.from_file("/home/zubairirshad/Documentsindividual0.ply")
# pointcloud.plot()
n=2
pcd=[]
_DEBUG_FILE_PATH = pathlib.Path(f'/home/zubairirshad/o3d/example60')
name = 'WUHWVHJyG4kyPjYH7KebHV'
unitcube = unit_cube()

# for i in range(6):
#     # filename = '/home/zubairirshad/o3d/individual'+str(i)+'.ply'
    
#     # filename = str(_DEBUG_FILE_PATH)+'/obb_'+ name + '_'+str(i)+'.ply'
#     # filename_box = str(_DEBUG_FILE_PATH)+'/box_'+ name + '_'+str(i)+'.ply'

#     filename = str(_DEBUG_FILE_PATH)+'/rotated_'+ name + '_'+str(i)+'.ply'
#     filename_box = str(_DEBUG_FILE_PATH)+'/rotated_box'+ name + '_'+str(i)+'.ply'
    

#     # filename = str(_DEBUG_FILE_PATH)+'/rotatedactual_'+ name + '_'+str(i)+'.ply'
    
#     single_pcd = o3d.io.read_point_cloud(filename)
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    
#     pcd.append(o3d.io.read_point_cloud(filename))

#     # pc_box = o3d.io.read_point_cloud(filename_box)
#     # points_box = np.asarray(pc_box.points)
#     # visualize_shape('pc', 'points', [single_pcd, mesh_frame], '/home/zubairirshad/Documents')
#     # pcd.append(line_set(points_box))

# visualize_shape('pc', 'points', pcd, '/home/zubairirshad/Documents')

# for j in range(30):
name = str(10)
meshes=[]
unitbox=[]

pcd_array=[]
box_array=[]
box_aabb =[]
box_obb=[]

def project(K, p_3d):
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
    return projections_2d

visualize=True

edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

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
edges_corners_pyobb = [[0, 1], [2, 3], [0, 3], [1, 2], [4, 5], [6, 7], [5, 6], [4, 7], [0, 5], [1, 4], [3, 6], [2, 7]]

pc_colors = [(240, 249, 33), (248, 149, 64), (204, 71, 120), (126, 3, 168), (13, 8, 135)]

#pc_colors = [(253, 231, 37), (94, 201, 98), (33, 145, 140), (59, 82, 139), (68, 1, 84)]

pc_colors = np.array(pc_colors).astype(np.float32)
colors_mesh = [(213,185,184), (150,194,187), (125,125,186), (191,183,147), (186,186,214), (234,218,218)]
colors_mesh = np.array(colors_mesh).astype(np.float32)
pcd = []
objs = ['laptop_black', 'bowl_3', 'bowl_3', 'mug_2']
combined_pointclouds = []
name ='8'
#file_dir = '/home/zubairirshad/fleet/simnet/data/simnet.debug/rgbd_nocs_Real_without_ICP_Vis_COLORnorm_24without_ICP/validation_inference/'
file_dir = '/home/zubairirshad/fleet/simnet/data/simnet.debug/rgbd_nocs_rgbd_nocs_Real_with_ICP_Vis_COLOR_texturenorm_24/validation_inference/'
meshes = []
meshes_2 = []
for i in range(6):
    # filename7= '/home/zubairirshad/o3d/example56/mesh_camera_T_centered'+str(i)+'.ply'
    # filename = '/home/zubairirshad/fleet/simnet/data/simnet/zubair_ZED2_pcd_tidy_vis5/decoder_output'+name+'_'+str(i)+'.ply'
    
    
    # filename_original = file_dir+'original_output'+name+str(i)+'.ply'
    filename = file_dir+'rotated_output'+name+str(i)+'.ply'
    filename_box = file_dir+'box_output'+name+str(i)+'.ply'
    # filename_mesh_frame = file_dir+'box_output'+name+str(i)+'.ply'
    filename_original = file_dir+'original_output'+name+str(i)+'.ply'
    
    single_pcd_original = o3d.io.read_point_cloud(filename_original)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
    # custom_draw_geometry_with_rotation([single_pcd_original, mesh_frame])
    # o3d.visualization.draw_geometries([single_pcd_original, mesh_frame])

    # filename = file_dir+'out_rotated_output'+name+str(i)+'.ply'
    # filename_box = file_dir+'out_box_output'+name+str(i)+'.ply'
    filename_mesh_frame = file_dir+'mesh_frame'+name+str(i)+'.ply'
    single_pcd_mesh = o3d.io.read_triangle_mesh(filename_mesh_frame)
    # pcd.append(single_pcd_mesh)

    pc_box = o3d.io.read_point_cloud(filename_box)
    points_box = np.asarray(pc_box.points)
    
    cylinder_segments = line_set(points_box)
    for j in range(len(cylinder_segments)):
        meshes.append(cylinder_segments[j])
        pcd.append(cylinder_segments[j])
    print("points_box", points_box.shape)

    # filename = '/home/zubairirshad/fleet/simnet/data/simnet/zubair_panoptic_NOCS'+name+str(i)+'.ply'
    # filename_box = '/home/zubairirshad/fleet/simnet/data/simnet/zubair_panoptic_NOCS'+name+str(i)+'.ply'
    # filename_mesh_frame = '/home/zubairirshad/fleet/simnet/data/simnet.debug/pc_latent_abs_pose_6/validation_inference/mesh_frame'+name+str(i)+'.ply'

    single_pcd = o3d.io.read_point_cloud(filename)


    pcd.append(single_pcd)
    # print("single pcd", np.array(single_pcd.points).shape)

    single_pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

    single_pcd.estimate_normals()
    single_pcd.orient_normals_consistent_tangent_plane(1000)

    # # # radii = [0.005, 0.01, 0.02, 0.04, 0.08, 0.1]
    # # # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    # # #     single_pcd, o3d.utility.DoubleVector(radii))
    # # # o3d.visualization.draw_geometries([single_pcd, mesh])
    # # # alpha = 0.08
    # # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    # # single_pcd, depth=9)
    # # vertices_to_remove = densities < np.quantile(densities, 0.05)
    # # mesh.remove_vertices_by_mask(vertices_to_remove)

    # # single_pcd_original = single_pcd
    # # single_pcd_original.normals = o3d.utility.Vector3dVector(np.zeros(
    # # (1, 3))) 
    # # single_pcd_original.estimate_normals()
    # # single_pcd_original.orient_normals_consistent_tangent_plane(100)
    alpha = 0.08
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(single_pcd, alpha)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(colors_mesh[i]/255.0)
    # mesh.vertex_colors = mesh.vertex_normals
    # o3d.visualization.draw_geometries([mesh, single_pcd_mesh], mesh_show_back_face=True)
    meshes.append(mesh)
    meshes.append(single_pcd_mesh)
    # meshes.append(single_pcd_mesh)
    pcd.append(single_pcd_mesh)
    # meshes.append(mesh)
    # meshes.append(single_pcd_mesh)
    # mesh.compute_vertex_normals()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(mesh)
    # vis.get_render_option().mesh_color_option = open3d.visualization.MeshColorOption.Normal
    # vis.run()
    # vis.destroy_window()
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh, single_pcd_mesh])



    # custom_draw_geometry_load_option([single_pcd_original, mesh_frame])
    #custom_draw_geometry_load_option([single_pcd_original, *unitcube, mesh_frame])
    # single_pcd.paint_uniform_color(pc_colors[i]/255.0)
    # o3d.visualization.draw_geometries([single_pcd])

    # print("single_pc points", np.array(single_pcd.points).shape)
    single_pcd = o3d.io.read_point_cloud(filename)
    aabb = single_pcd.get_axis_aligned_bounding_box().get_box_points()
    obb = single_pcd.get_oriented_bounding_box().get_box_points()

    # points_aabb = camera.convert_points_to_homopoints(np.array(aabb).T)
    # points_2d_aabb = project(_CAMERA.K_matrix, points_aabb)
    # points_2d_aabb = points_2d_aabb.T
    # box_aabb.append(points_2d_aabb)
    # print("points aabb", points_2d_aabb.shape)
    print("obb", np.array(obb).shape)
    # pcd_obb = o3d.geometry.PointCloud()
    # cylinder_segments = line_set(np.array(obb))
    # for i in range(len(cylinder_segments)):
    #     pcd.append(cylinder_segments[i])
    # pcd_obb.points = o3d.utility.Vector3dVector(line_set(np.array(obb)))

    points_obb = camera.convert_points_to_homopoints(np.array(obb).T)
    # print(points_obb.shape)
    # points_obb_vis = camera.convert_homopoints_to_points(points_obb).T
    points_2d_obb = project(_CAMERA.K_matrix, points_obb)
    #points_2d_obb = project(_CAMERA.proj_matrix, points_obb)
    points_2d_obb = points_2d_obb.T
    box_obb.append(points_2d_obb)
    # pcd.append(pcd_obb)

    # o3d.visualization.draw_geometries([mesh], zoom=0.664,
    #                                 front=[-0.4761, -0.4698, -0.7434],
    #                                 lookat=[1.8900, 3.2596, 0.9284],
    #                                 up=[0.2304, -0.8825, 0.4101])

    # alpha = 0.08
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pointcloud_with_texture, alpha)
    # mesh.compute_vertex_normals()
    # custom_draw_geometry_with_rotation([mesh])
    # o3d.visualization.draw_geometries([mesh, mesh_frame_individual], mesh_show_back_face=True)
    # pc_box_trimesh = trimesh.PointCloud(np.array(line_set(points_box).points))
    # scene.add_geometry(pc_box_trimesh)
    # pcd.append(line_set(points_box))
    color_img = Image.open(file_dir+name+'_image.png')
    color_img = color_img.convert('RGB')
    
    # pcd.append(line_set(points_box))

    points_box = camera.convert_points_to_homopoints(points_box.T)
    #points_2d = project(_CAMERA.proj_matrix, points_box)
    points_2d = project(_CAMERA.K_matrix, points_box)
    points_2d = points_2d.T
    box_array.append(points_2d)
    # print("color ", color_img)

    points_mesh = camera.convert_points_to_homopoints(np.array(single_pcd.points).T)
    #points_2d_mesh = project(_CAMERA.proj_matrix, points_mesh)
    points_2d_mesh = project(_CAMERA.K_matrix, points_mesh)
    points_2d_mesh = points_2d_mesh.T
    colors = []
    print(points_2d_mesh.shape)
    for j in range(points_2d_mesh.shape[0]):
        # im = Image.fromarray(np.uint8(color_img/255.0))
        color = color_img.getpixel((int(points_2d_mesh[j,0]), int(points_2d_mesh[j,1])))
        color = np.array(color)
        colors.append(color/255.0)
    single_pcd = o3d.io.read_point_cloud(filename)
    single_pcd.colors  = o3d.utility.Vector3dVector(np.array(colors))
    single_pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

    single_pcd.estimate_normals()
    single_pcd.orient_normals_consistent_tangent_plane(15)

    # alpha = 0.08

    # alpha = 0.08
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(single_pcd, alpha)
    # mesh.compute_vertex_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    single_pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # o3d.io.write_triangle_mesh(file_dir+objs[i]+'_reconstruction.obj', mesh)
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # pc_box_trimesh = trimesh.PointCloud(np.array(line_set(points_box).points))
    meshes_2.append(mesh)
    meshes_2.append(single_pcd_mesh)
    # custom_draw_geometry_load_option([single_pcd_mesh,single_pcd, *cylinder_segments])

    pcd_array.append(points_2d_mesh)

# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
# pcd.append(mesh_frame)
# custom_draw_geometry_load_option(pcd)

# custom_draw_geometry_load_option(meshes)

custom_draw_geometry_with_rotation(pcd)
# custom_draw_geometry_load_option(pcd)
vis = o3d.visualization.Visualizer()
vis.create_window()
for k in range(len(meshes)):
    vis.add_geometry(meshes[k])
opt = vis.get_render_option()
opt.mesh_show_back_face = True
# opt.background_color = np.asarray([245/255.0, 255/255.0, 246/255.0])
opt.background_color = np.asarray([1, 1, 1])
# opt.background_color = np.asarray([255/255.0, 245/255.0, 244/255.0])

vis.run()
vis.destroy_window()
def rotate_view(vis):
    opt = vis.get_render_option()
    vis.create_window()
    # vis.create_window(window_name=name, width=1920, height=1080)
    opt.background_color = np.asarray([1, 1, 1])
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters
    ctr.rotate(1.0, 0.0)
    # return False

o3d.visualization.draw_geometries_with_animation_callback(meshes_2,
                                                            rotate_view)

o3d.visualization.draw_geometries_with_animation_callback(meshes,
                                                            rotate_view)

o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)
# pc_cat = np.concatenate([np.array(pc.points) for pc in pcd])
# print("pc cat", pc_cat.shape)
# filename = file_dir+'combined_output0.ply'

# print(filename)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pc_cat)
# o3d.io.write_point_cloud(filename, pcd)

# if visualize:
#     # color_img = cv2.imread('/home/zubairirshad/fleet/simnet/data/simnet.debug/pc_latent_output_val_TEST10/validation_inference/10_image.png')
#     color_img = cv2.imread('/home/zubairirshad/fleet/simnet/data/simnet/zubair_ZED2_test_pose/'+name+'img.png')
# # Visualize
#     plt.xlim((0, color_img.shape[1]))
#     plt.ylim((0, color_img.shape[0]))
#     # Projections
#     color = ['g', 'b', 'r', 'y']
#     for points_2d in box_aabb:
#         for j in range(len(points_2d)):
#             plt.text(points_2d[j, 0], points_2d[j, 1],str(j))
#         for edge in open_3d_lines:
#             print(points_2d[i, 0], points_2d[i, 1])
#             plt.plot(points_2d[edge, 0], points_2d[edge, 1], color=color[i], linewidth=3.0)
#             # plt.plot(points_2d[i, 0], points_2d[i, 1],marker='v', color='white')
#     for i, points_2d_mesh in enumerate(pcd_array):
#         for points in points_2d_mesh:
#             plt.scatter(points[0], points[1], color=color[i], s=4)
#     plt.gca().invert_yaxis()
#     plt.imshow(color_img)
#     plt.show()

visualize=True
if visualize:
    color_img = cv2.imread(file_dir+name+'_image.png')
    plt.xlim((0, color_img.shape[1]))
    plt.ylim((0, color_img.shape[0]))
    # Projections
    color = ['g', 'y', 'b', 'r', 'm', 'c', '#3a7c00', '#3a7cd9', '#8b7cd9', '#211249']
    for i, points_2d_mesh in enumerate(pcd_array):
        plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[i], s=2)
        # for points in points_2d_mesh:
        #     plt.scatter(points[0], points[1], color=color[i], s=2)
    plt.gca().invert_yaxis()
    # plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.imshow(color_img)
    # plt_name = name+'plot_points'
    plt.axis('off')
    # plt.show()
    # plt.savefig(file_dir+plt_name+'img_new.png', bbox_inches='tight')

    for points_2d in box_obb:
        for edge in open_3d_lines:
            plt.plot(points_2d[edge, 0], points_2d[edge, 1], color='b', linewidth=1.0)
    plt_name = name+'plot_w_box'
    plt.axis('off')
    plt.show()
    # plt.savefig(file_dir+plt_name+'img_new.png', bbox_inches='tight')

# custom_draw_geometry_with_rotation(pcd)
# visualize_shape('pc', 'points', pcd, '/home/zubairirshad/Documents')



# def show(xy):
#     import matplotlib.pyplot as plt
#     colors = [
#     [0, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ]
#     x, y = xy.T
#     ax = plt.gca()
#     ax.scatter(x, y)
#     plt.show()

# stacked_points = []
# for i in range(len(pcd_array)):
#     points = camera.convert_points_to_homopoints(pcd_array[i].T)
#     stacked_points.append(points)


# stacked_points = np.concatenate(stacked_points, axis=1)
# p_2d = project(_CAMERA.proj_matrix, stacked_points)
# show(p_2d)