import os
import sys
import copy
import numpy as np
import open3d as o3d
from math import pi


def draw_frame(origin=[0,0,0], q=[0,0,0,1], scale=1):
    # open3d quaternion format qw qx qy qz

    o3d_quat = np.array([q[3], q[0], q[1], q[2]])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(o3d_quat), center=origin)
    
    return frame_rot

def draw_camera(origin=[0,0,0], scale=0.5):


	cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01*scale, height=scale)
	cyld_0.paint_uniform_color([0,0,0])
	cyld_0.translate(np.array([0,0,scale/2]))
	cyld_0.rotate(
		cyld_0.get_rotation_matrix_from_xyz(np.array([0, pi/2,0])), center=[0,0,0])
	cyld_1 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_zyx(np.array([pi/4, pi/5,0])), center=[0,0,0])
	cyld_2 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_zyx(np.array([-pi/4, pi/5,0])), center=[0,0,0])
	cyld_3 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_zyx(np.array([pi/4, -pi/5,0])), center=[0,0,0])
	cyld_4 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_zyx(np.array([-pi/4, -pi/5,0])), center=[0,0,0])

	mesh = [cyld_1, cyld_2, cyld_3, cyld_4]

	cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01*scale, height=1.16*scale)
	cyld_0.paint_uniform_color([0,0,0])
	cyld_5 = copy.deepcopy(cyld_0).translate(np.array([0.5773*scale, -0.5773*scale, 0]))
	cyld_6 = copy.deepcopy(cyld_0).translate(np.array([0.5773*scale, 0.5773*scale, 0]))
	
	cyld_0.rotate(
		cyld_0.get_rotation_matrix_from_xyz(np.array([pi/2,0,0])), center=[0,0,0])
	cyld_7 = copy.deepcopy(cyld_0).translate(np.array([0.5773*scale,0, -0.5773*scale]))
	cyld_8 = copy.deepcopy(cyld_0).translate(np.array([0.5773*scale, 0,0.5773*scale]))
	
	mesh += [cyld_5, cyld_6, cyld_7, cyld_8]

	for m in mesh:
		m.translate(np.array(origin), relative=True)
	return mesh

def draw_plane(corners, nh, nw, color=[0,0,1]):
	top_right, top_left, bottom_right, bottom_left = corners
	top = np.linspace(top_right, top_left, nw)
	bottom = np.linspace(bottom_right, bottom_left, nw)
	right = np.linspace(top_right, bottom_right, nh)
	left = np.linspace(top_left, bottom_left, nh)

	lines = [[i, i+nw] for i in range(nw)] + [[2*nw+i, 2*nw+i+nh] for i in range(nh)]
	coord = np.concatenate([top, bottom, right, left], axis=0)
	colors = [color for i in range(2*nw+2*nh)]
	line_plane = o3d.geometry.LineSet(
	    points=o3d.utility.Vector3dVector(coord),
	    lines=o3d.utility.Vector2iVector(lines),
	)
	line_plane.colors = o3d.utility.Vector3dVector(colors)
	return line_plane

# cam_frame = draw_frame(scale=0.3)

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
cam_frame_1 = draw_frame(origin=[0, -0.5, 0], q=[0.506, -0.495, 0.500, -0.498], scale=0.1)
cam1 = draw_camera(origin=[0, -0.5, 0], scale=0.2)

cam_frame_2 = draw_frame(origin=[-0.05, -0.3, 0.02], q=[0.506, -0.495, 0.500, -0.498], scale=0.1)
cam2 = draw_camera(origin=[-0.05, -0.3, 0.02], scale=0.2)

# draw corn plane
corners = [[0.42, 0.1, 0.25],
		   [0.5, -0.85, 0.25],
		   [0.42, 0.1, -0.1],
		   [0.5, -0.85, -0.1]]
line_plane = draw_plane(corners, 10, 20)

# draw corn 
# o3d.visualization.draw_geometries([pcd, cam_frame_1, cam_frame_2, line_plane] + cam1 + cam2)

alpha = 0.006
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
m = [cam_frame_1, cam_frame_2] + cam1 + cam2
for item in m:
	mesh += item
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

o3d.io.write_triangle_mesh("corn_side_view.obj", mesh)