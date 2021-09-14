# import os
# import sys
# import numpy as np
# import open3d as o3d

# cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
# side_view = o3d.io.read_point_cloud(pcd_fname)
# side_view.rotate(side_view.get_rotation_matrix_from_zyx(np.array([0, 0.1,0])), center=[0,0,0])

# o3d.visualization.draw_geometries([side_view, cam_frame])

import os
import sys
import numpy as np
import time
import open3d as o3d
import cv2
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from collections import defaultdict

import torch
import torchvision.transforms as T
import argparse
from plane_estimation import PlaneEstimator
from corn_tracker import MOT_Tracker
from localization import *
from point_cloud import PointCloud

plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="model/faster-rcnn-corn_bgr8_ep100.pt",
                help="path to the model")
parser.add_argument("-c", "--confidence", type=float, default=0.8, 
                help="confidence to keep predictions")
parser.add_argument("-d", "--data_dir", default="tmp/offline_frames")
parser.add_argument('--reconstruct', type=int, default=0)
parser.add_argument('--build_mesh', type=int, default=0)
parser.add_argument("--pcd_name", type=str, default="side_view")
args = parser.parse_args()


class Frame():
    """sync-ed frame for side and front view"""
    def __init__(self, front_color, front_depth, side_color, side_depth, stamp, pose):
        self.front_color = front_color
        self.front_depth = front_depth
        self.side_color = side_color
        self.side_depth = side_depth
        self.stamp = stamp
        self.pose = pose

pcd_fname = args.pcd_name + '.pcd'

if args.reconstruct:
    CLASS_NAMES = ["__background__", "corn_stem"]
    ROOT = os.path.dirname(os.path.abspath(__file__))
    HOME = str(Path.home())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(os.path.join(ROOT, args.model))
    model.to(device)

    data_dir = os.path.join(HOME, args.data_dir)
    frame_dir = os.path.join(data_dir, 'frame')
    front_color_dir = os.path.join(data_dir, 'front_color')
    side_color_dir = os.path.join(data_dir, 'side_color')

    plane_estimator = PlaneEstimator(args, model)
    tracker = MOT_Tracker(args, model)
    side_pcd = PointCloud(vis=False)
    history = defaultdict(list)

    ################ MAIN LOOP, READ FRAMES ONE BY ONE  ###############
    num_frames = len(os.listdir(frame_dir))
    print("Found %d frames, start loading now....")

    i_start = 1
    num_frames = 95

    for i in range(i_start, i_start+num_frames):
        frame = pickle.load(open(os.path.join(frame_dir, 'frame_%07d.pkl'%(i)), 'rb'))
        print('Loaded frame number %d'%i)

        # merge pointcloud every 5 frames
        if (i-i_start) % 20 == 0:
            points, depth_mask = side_pcd.depth_to_points(frame.side_depth, plane_estimator.K)

            # color thresholding
            color = frame.side_color
            ratio1, ratio2 = color[:,:,1] / color[:,:,0] , color[:,:,2] / color[:,:,0] 
            color_mask = (color[:,:,0]<60) & (ratio1>0.9) & (ratio1<1.1) & (ratio2>0.9) & (ratio2<1.1)

            mask = depth_mask & (~color_mask.reshape(-1))

            side_pcd.merge(points[mask], frame.side_color.reshape(-1,3)[mask], frame.pose)
            o3d.io.write_point_cloud(pcd_fname, side_pcd.point_cloud)


    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    pcd = o3d.io.read_point_cloud(pcd_fname)
    o3d.visualization.draw_geometries([pcd, cam_frame])
    
    pcd.rotate(pcd.get_rotation_matrix_from_zyx(np.array([0, 0.09,0])), center=[0,0,0])
    pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-100, -100, 0]), 
                                                       np.array([100, 100, 0.49])))
    pcd.rotate(pcd.get_rotation_matrix_from_zyx(np.array([0, -0.09,0])), center=[0,0,0])
    o3d.visualization.draw_geometries([pcd, cam_frame])
    o3d.io.write_point_cloud(pcd_fname, pcd)

if args.build_mesh:
    pcd = o3d.io.read_point_cloud(pcd_fname)
    print("Loaded %d points"%(np.asarray(pcd.points).shape[0]))
    # pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-1, -0.25, 0.3]), 
    #                                                    np.array([1.5, 100, 1])))
    o3d.visualization.draw_geometries([pcd])
    print("Cropped down to %d points"%(np.asarray(pcd.points).shape[0]))
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    print('surface normal estimation finished')

    alpha = 0.005
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("side_view.ply", mesh)