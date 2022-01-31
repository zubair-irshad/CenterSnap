import matplotlib.pyplot as plt
import numpy as np
from simnet.lib import camera
import matplotlib.patches as patches
import cv2
import random
import open3d as o3d
import torch
from matplotlib.cm import get_cmap
import copy
import colorsys
import plotly.graph_objects as go

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

def line_set_mesh(points_array):
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
  colors = random_colors(len(open_3d_lines))
  open_3d_lines = np.array(open_3d_lines)
  line_set = LineMesh(points_array, open_3d_lines,colors=colors, radius=0.001)
  line_set = line_set.cylinder_segments
  return line_set


def line_set(points_array):
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
  line_set.colors = o3d.utility.Vector3dVector(colors)
  return line_set


def visualize_projected_points(color_img, pcd_array, box_obb, _DEBUG_FILE_PATH, uid):
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
        plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[i], s=2)
        # for points in points_2d_mesh:
            
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.imshow(color_img)
    # plt_name = name+'plot_points'

    # # plt.show()
    # # plt.savefig('/home/zubairirshad/fleet/simnet/data/simnet/zubair_ZED2_test_pose_4/'+plt_name+'img.png', bbox_inches='tight')

    for points_2d in box_obb:
        for edge in open_3d_lines:
            plt.plot(points_2d[edge, 0], points_2d[edge, 1], color='b', linewidth=1.0)

    plt.savefig(str(_DEBUG_FILE_PATH / f'{uid}_projection.png'))
    # plt.axis('off')
    # plt.show()
    # plt.close('all')


def visualize_projected_points_only(color_img, pcd_array):
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
        plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[i], s=2)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.imshow(color_img)
    plt.show()

def show_projected_points(color_img, pcd_array):
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
    fig = plt.figure()
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    plt.xlim((0, color_img.shape[1]))
    plt.ylim((0, color_img.shape[0]))
    # Projections
    plt.imshow(color_img[...,::-1])
    color = ['g', 'y', 'b', 'r', 'm', 'c', '#3a7c00', '#3a7cd9', '#8b7cd9', '#211249']
    for i, points_2d_mesh in enumerate(pcd_array):
        plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[i], s=2)
    plt.gca().invert_yaxis()
    plt.axis('off')
    fig.canvas.draw()
    color_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    color_img = color_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    return color_img


def save_projected_points(color_img, pcd_array, output_path, uid):
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
    plt.figure()
    plt.xlim((0, color_img.shape[1]))
    plt.ylim((0, color_img.shape[0]))
    # Projections
    color = ['g', 'y', 'b', 'r', 'm', 'c', '#3a7c00', '#3a7cd9', '#8b7cd9', '#211249']
    for i, points_2d_mesh in enumerate(pcd_array):
        plt.scatter(points_2d_mesh[:,0], points_2d_mesh[:,1], color=color[i], s=2)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.imshow(color_img[:,:,::-1])
    plt.savefig(output_path +'/projection'+str(uid)+'.png')

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

def visualize(detections, img, classes, seg_mask, object_key_to_name, filename):
    colors = random_colors(len(classes))
    fig, ax = plt.subplots(1, figsize=(10,7.5))
    plt.axis('off')
    ax.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
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
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]


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


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

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

def display_gird(img_vis, depth, peaks_vis):
    images=[]
    images.append(np.copy(img_vis)[...,::-1])
    images.append(np.copy(peaks_vis)[...,::-1])
    depth_vis = depth2inv(torch.tensor(depth).unsqueeze(0).unsqueeze(0))
    depth_vis = viz_inv_depth(depth_vis)
    images.append(np.copy(depth_vis))
    rows=3
    img_count = 0
    print(images[0].shape)
    fig, axes = plt.subplots(ncols=rows, figsize=(15,15))
    for i in range(rows):     
            if img_count < len(images):
                axes[i].imshow(images[img_count])
                img_count+=1

def draw_geometries(geometries):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()
        
        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=1, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)
            
            mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=triangles[:,0], j=triangles[:,1], k=triangles[:,2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)
        
    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()