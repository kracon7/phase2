# import os
# import sys
# import numpy as np
# import open3d as o3d

# cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
# side_view = o3d.io.read_point_cloud('side_view.pcd')
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
args = parser.parse_args()

history = pickle.load(open('history.pkl', 'wb'))

cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
pcd = o3d.io.read_point_cloud('side_view.pcd')
pcd.rotate(pcd.get_rotation_matrix_from_zyx(np.array([0, 0.09,0])), center=[0,0,0])
pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-100, -100, 0]), 
                                                   np.array([100, 100, 0.49])))
pcd.rotate(pcd.get_rotation_matrix_from_zyx(np.array([0, -0.09,0])), center=[0,0,0])

# average history to compute target positions