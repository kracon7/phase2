import os
import sys
import copy
import numpy as np
import open3d as o3d
from math import pi



def draw_frame(origin=[0,0,0], q=[1,0,0,0], scale=1):
    # open3d quaternion format qw qx qy qz

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(q), center=origin)
    
    return frame_rot

def draw_camera(origin=[0,0,0], q=[1,0,0,0], scale=0.13):
	# draw original camera aixs
	axis = draw_frame([0,0,0], [1,0,0,0], scale=0.5*scale)
	mesh = [axis]

	# draw the original camera frame
	cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01*scale, height=scale)
	cyld_0.paint_uniform_color([0,0,0])
	cyld_0.translate(np.array([0,0,scale/2]))
	cyld_1 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_zyx(np.array([pi/4, pi/4,0])), center=[0,0,0])
	cyld_2 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_zyx(np.array([-pi/4, pi/4,0])), center=[0,0,0])
	cyld_3 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_zyx(np.array([pi/4, -pi/4,0])), center=[0,0,0])
	cyld_4 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_zyx(np.array([-pi/4, -pi/4,0])), center=[0,0,0])

	mesh += [cyld_1, cyld_2, cyld_3, cyld_4]

	cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01*scale, height=1*scale)
	cyld_0.paint_uniform_color([0,0,0])
	cyld_0.rotate(
		cyld_0.get_rotation_matrix_from_xyz(np.array([0,pi/2,0])), center=[0,0,0])
	cyld_5 = copy.deepcopy(cyld_0).translate(np.array([0, 0.5*scale, 0.7*scale]))
	cyld_6 = copy.deepcopy(cyld_0).translate(np.array([0,-0.5*scale, 0.7*scale]))
	
	cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01*scale, height=1*scale)
	cyld_0.paint_uniform_color([0,0,0])
	cyld_0.rotate(
		cyld_0.get_rotation_matrix_from_xyz(np.array([pi/2,0,0])), center=[0,0,0])
	cyld_7 = copy.deepcopy(cyld_0).translate(np.array([ 0.5*scale, 0, 0.7*scale]))
	cyld_8 = copy.deepcopy(cyld_0).translate(np.array([-0.5*scale, 0, 0.7*scale]))
	
	mesh += [cyld_5, cyld_6, cyld_7, cyld_8]

	# apply frame rotation 
	for m in mesh:
		m.rotate(m.get_rotation_matrix_from_quaternion(q), center=(0,0,0))
	# apply frame translation
	for m in mesh:
		m.translate(np.array(origin), relative=True)
	return mesh


ROOT = os.path.dirname(os.path.abspath(__file__))
pcd_file = os.path.join(ROOT, 'corn.pcd')
pcd = o3d.io.read_point_cloud(pcd_file)

# o3d.visualization.draw_geometries([pcd, cam_frame])

# Transform poincloud to side_view frame
pcd.rotate(pcd.get_rotation_matrix_from_quaternion(np.array([-0.653, 0.270, -0.270, -0.653])), 
		   center=np.array([0,0,0]))
pcd.translate(np.array([0.144, 0.814, 0.045]), relative=True)

# o3d.visualization.draw_geometries([pcd, cam_frame])

# crop point cloud to filter out background
pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-0, -100, -100]), 
                                                   np.array([0.55, 0.1, 100])))
# o3d.visualization.draw_geometries([pcd, cam_frame])
            
# draw two cameras
cam1 = draw_camera(origin=[0, -0.5, 0], q=[0.506, -0.495, 0.500, -0.498])
cam2 = draw_camera(origin=[-0.05, -0.3, 0.02], q=[0.506, -0.495, 0.500, -0.498])

# draw corn 
o3d.visualization.draw_geometries([pcd] + cam1 + cam2)

alpha = 0.006
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
m = cam1 + cam2
for item in m:
	mesh += item
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# o3d.io.write_triangle_mesh("corn_side_view.obj", mesh)