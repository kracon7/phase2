import os
import sys
import argparse
import time
import pickle
import copy
import numpy as np
import datetime as dt
import open3d as o3d
import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from torch.utils.data.dataset import Dataset
# from model import CorridorNet

np.random.seed(0)

def points2heightmap(surface_pts, heightmap_size, ws_limits=[[-0.5,0.5],[-0.4,0.25],[0.1, 0.7]]):

    resol_x = (ws_limits[0][1] - ws_limits[0][0]) / heightmap_size[1]
    resol_y = (ws_limits[1][1] - ws_limits[1][0]) / heightmap_size[0]

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = (surface_pts[:,0] > ws_limits[0][0]) & (surface_pts[:,0] < ws_limits[0][1]) & \
                          (surface_pts[:,1] > ws_limits[1][0]) & (surface_pts[:,1] < ws_limits[1][1]) & \
                          (surface_pts[:,2] > ws_limits[2][0]) & (surface_pts[:,2] < ws_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - ws_limits[0][0]) / resol_x).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - ws_limits[1][0]) / resol_y).astype(int)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = ws_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    # depth_heightmap[depth_heightmap == -z_bottom] = np.nan
    return depth_heightmap

def draw_frame(origin=[0,0,0], q=[0,0,0,1], scale=1):
    # open3d quaternion format qw qx qy qz

    o3d_quat = np.array([q[3], q[0], q[1], q[2]])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(o3d_quat))
    
    return frame_rot

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def main(args):
    
    plt.ion()
    flist = os.listdir(os.path.join(args.load_from, 'front_rgbd'))
    flist.sort()
    output_dir = os.path.join(os.path.dirname(args.load_from), 'front_rectified')

    cam_frame = draw_frame(scale=0.3)

    fig, ax = plt.subplots(2,1)
    
    for idx in range(len(flist)):
        try:
            # idx = np.random.randint(len(flist))
            # idx = 10 * i
            print(idx)

            xyzrgb = np.load(os.path.join(args.load_from, 'front_rgbd', flist[idx])).reshape(-1,6)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,:3]))
            pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:].astype('float') / 255)
            # o3d.visualization.draw_geometries([pcd, cam_frame])

            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-0.6, -0.2, 0]), 
                                                               np.array([0.6, 0.35, 8])))
            # o3d.visualization.draw_geometries([pcd, cam_frame])

            pcd = pcd.voxel_down_sample(voxel_size=0.04)
            # o3d.visualization.draw_geometries([pcd, cam_frame])

            plane_model, inliers = pcd.segment_plane(distance_threshold=0.04,
                                                     ransac_n=3,
                                                     num_iterations=100)
            y_axis = plane_model[:3]
            if y_axis[1] < 0:
                y_axis = - y_axis
            # display_inlier_outlier(pcd, inliers)

            # use ground points to do PCA to find z axis
            ground_points = pcd.select_by_index(inliers)
            _, cov = ground_points.compute_mean_and_covariance()
            eigen_values, eigen_vectors = np.linalg.eig(cov)
            z_axis = eigen_vectors[:,0]
            if z_axis[2] < 0:
                z_axis = -z_axis

            x_axis = np.cross(y_axis, z_axis)

            # rectify the point cloud
            R = np.stack([x_axis, y_axis, z_axis])

            corn_points = pcd.select_by_index(inliers, invert=True)
            rectified_corn = corn_points.rotate(R, center=np.array([0., 0., 0.]))
            # o3d.visualization.draw_geometries([rectified_corn, cam_frame])

            # randomly downsample the point cloud
            num_points = np.asarray(rectified_corn.points).shape[0]
            sampling_ratio = 1000./num_points
            rectified_corn = rectified_corn.random_down_sample(sampling_ratio)

            rectified_x = np.asarray(rectified_corn.points)[:,0]
            bins = np.linspace(-0.6, 0.6, 61)
            hist, _ = np.histogram(rectified_x, bins)
            hist[hist<20] = 0
            ptx = bins[np.nonzero(hist)[0][0]] - 0.1
            pt = R.T @ np.array([ptx, -plane_model[3], 0])

            pickle.dump([pt, -y_axis, z_axis], 
                        open(os.path.join(args.load_from, 'plane', flist[idx].split('.')[0]+'.pkl'), 'wb'))


    
        except (RuntimeError, IndexError):
            print('Failed to find ground')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test depth filter and projection for corn images')
    parser.add_argument('--load_from', default='/home/jc/tmp', help='directory to load images')
    parser.add_argument('--color', default=0, type=int, help='load color or not')
    args = parser.parse_args()
    
    main(args)