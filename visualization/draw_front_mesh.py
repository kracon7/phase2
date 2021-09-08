import os
import sys
import copy
import pickle
import numpy as np
import open3d as o3d
from math import pi


class Frame():
    """sync-ed frame for side and front view"""
    def __init__(self, front_color, front_depth, side_color, stamp, pose):
        self.front_color = front_color
        self.front_depth = front_depth
        self.side_color = side_color
        self.stamp = stamp
        self.pose = pose



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

def rgbd_to_pcd(color, depth):
	# compute rays in advance
    K = np.array([[615.311279296875,   0.0,             430.1778869628906],
                       [  0.0,            615.4699096679688, 240.68307495117188],
                       [  0.0,              0.0,               1.0]])
    im_w = 848
    im_h = 480
    x, y = np.arange(im_w), np.arange(im_h)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy], axis=2).reshape(-1,2)
    rays = np.dot(np.insert(points, 2, 1, axis=1), 
    				   np.linalg.inv(K).T).reshape(im_h, im_w, 3)
    points = rays * depth.reshape(im_h, im_w, 1)
    return points.reshape(-1, 3)

frame_dir = '/home/jc/tmp/offline_frames/frame'
idx = 70
frame = pickle.load(open(os.path.join(frame_dir, 'frame_%07d.pkl'%idx),'rb'))
points = rgbd_to_pcd(frame.front_color, frame.front_depth)
front_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
color = frame.front_color.reshape(-1,3).astype('float') / 255
front_pcd.colors = o3d.utility.Vector3dVector(color)
front_pcd.rotate(front_pcd.get_rotation_matrix_from_quaternion(
					np.array([0.9990482, -0.0436194, 0, 0])), center=[0,0,0])
front_pcd = front_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-0.6, -0.15, 0.5]), 
                  			                                   np.array([0.6, 0.6, 6])))
cam1 = draw_camera(origin=[0, 0, 0], q=[1,0,0,0])
o3d.visualization.draw_geometries([front_pcd]+cam1)
o3d.io.write_point_cloud("front.ply", front_pcd)
