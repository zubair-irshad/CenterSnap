from simnet.lib import camera
import open3d as o3d
import os
import numpy as np
_CAMERA = camera.NOCS_Camera()
from app.utils.viz_utils import visualize_projected_points_only
from app.utils.transform_utils import project

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


foldername = '/home/zubair/Downloads/sdf_cid_validation/validation_inference'
num=5
num_meshes = 3

meshes=[]
pcds=[]
test_2d_points = []
for i in range(num_meshes):
    # mesh_name = 'mesh'+str(num)+str(i)+'.ply'
    pcd_file_name = 'pcd_rotated'+str(num)+str(i)+'.ply'
    filename_pcd = os.path.join(foldername, pcd_file_name)
    pcd_load = o3d.io.read_point_cloud(filename_pcd)
    points_mesh = camera.convert_points_to_homopoints(np.array(pcd_load.points).T)
    points_2d_mesh = project(_CAMERA.K_matrix, points_mesh)
    points_2d_mesh = points_2d_mesh.T
    test_2d_points.append(points_2d_mesh)
    pcds.append(pcd_load)
visualize_projected_points_only(image, visualize_projected_points_only)
o3d.visualization.draw_geometries(pcds)