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
side_pcd = PointCloud()
history = defaultdict(list)

################ MAIN LOOP, READ FRAMES ONE BY ONE  ###############
num_frames = len(os.listdir(frame_dir))
print("Found %d frames, start loading now....")

for i in range(1, num_frames):
    frame = pickle.load(open(os.path.join(frame_dir, 'frame_%07d.pkl'%(i)), 'rb'))
    print('Loaded frame number %d'%i)

    # estimate plane
    plane_estimator.update(frame)

    # update tracker
    corn_id_bbox = tracker.corn_tracking_sort(frame.side_color)

    # use bbox and plane to find 3D position
    if plane_estimator.d_plane is not None:
        loc_3d = compute_loc_3d([0,0,1,plane_estimator.d_plane], [0,1,0,-0.1], 
                        corn_id_bbox, plane_estimator.K)

        merge_measurements(history, loc_3d, frame.pose)

        print(history)

    # merge pointcloud every 5 frames
    if i % 5 == 1:
        points = side_pcd.depth_to_points(frame.side_depth, plane_estimator.K)
        points_map = side_pcd.transform_side_to_map(points, frame.pose)
        side_pcd.merge(points_map, frame.side_color.reshape(-1,3))
        side_pcd.save_as_mesh('side_view.ply')

