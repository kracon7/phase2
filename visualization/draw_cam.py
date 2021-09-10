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
	mesh = axis

	# draw the original camera frame
	color = [0,0,0]
	cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01*scale, height=scale)
	cyld_0.paint_uniform_color(color)
	cyld_0.translate(np.array([0,0,scale/2]))
	cyld_1 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_xzy(np.array([0, pi/5, pi/5])), center=[0,0,0])
	cyld_2 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_xzy(np.array([0, -pi/5, pi/5])), center=[0,0,0])
	cyld_3 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_xzy(np.array([0, pi/5, -pi/5])), center=[0,0,0])
	cyld_4 = copy.deepcopy(cyld_0).rotate(
			cyld_0.get_rotation_matrix_from_xzy(np.array([0, -pi/5, -pi/5])), center=[0,0,0])

	for m in [cyld_1, cyld_2, cyld_3, cyld_4]: 
		mesh += m

	cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01*scale, height=0.96*scale)
	cyld_0.paint_uniform_color(color)
	cyld_0.rotate(
		cyld_0.get_rotation_matrix_from_xyz(np.array([0,pi/2,0])), center=[0,0,0])
	cyld_5 = copy.deepcopy(cyld_0).translate(np.array([0, 0.35*scale, 0.81*scale]))
	cyld_6 = copy.deepcopy(cyld_0).translate(np.array([0,-0.35*scale, 0.81*scale]))
	
	cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01*scale, height=0.7*scale)
	cyld_0.paint_uniform_color(color)
	cyld_0.rotate(
		cyld_0.get_rotation_matrix_from_xyz(np.array([pi/2,0,0])), center=[0,0,0])
	cyld_7 = copy.deepcopy(cyld_0).translate(np.array([ 0.48*scale, 0, 0.81*scale]))
	cyld_8 = copy.deepcopy(cyld_0).translate(np.array([-0.48*scale, 0, 0.81*scale]))
	
	for m in [cyld_5, cyld_6, cyld_7, cyld_8]:
		mesh +=  m

	# apply frame rotation 
	mesh.rotate(mesh.get_rotation_matrix_from_quaternion(q), center=(0,0,0))
	# apply frame translation
	mesh.translate(np.array(origin), relative=True)

	return mesh


cam0 = draw_camera()
o3d.visualization.draw_geometries([cam0])
o3d.io.write_triangle_mesh("camera.ply", cam0)
