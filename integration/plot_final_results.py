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
parser.add_argument("-p", "--pcd_path", 
                    default="/home/jc/tmp/phase2_visualization/side_view.pcd")
parser.add_argument("-l", "--history_path", 
                    default="/home/jc/jiacheng/phase2/integration/history.pkl")
args = parser.parse_args()

history = pickle.load(open(args.l, 'wb'))

cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
pcd = o3d.io.read_point_cloud(args.p)

# average history to compute target positions