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
import copy
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from collections import defaultdict
from math import pi

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
                    default="/home/jc/jiacheng/phase2/integration/side_view.pcd")
parser.add_argument("-l", "--history_path", 
                    default="/home/jc/jiacheng/phase2/integration/history.pkl")
args = parser.parse_args()

history = pickle.load(open(args.history_path, 'rb'))

cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
pcd = o3d.io.read_point_cloud(args.pcd_path)    
pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-0.2, -100, 0]), 
                                                   np.array([100, 100, 1])))
mesh = [pcd]

all_pos = []
for idx in history.keys():
    if len(history[idx]) > 6:
        all_pos.append(np.mean(np.stack(history[idx]), axis=0))
all_pos = np.stack(all_pos)

sorted_idx = np.argsort(all_pos[:,0])
N = all_pos.shape[0]
removed = np.array([0, 2, 4, 5, 6, 11, 13,14,15, 17,18,19,20,21,22,23, 25,26,27,28,
                    33,34, 37, 38, 39, 40, 41, 42, 43, 44, 45])
print(all_pos[sorted_idx[removed]], all_pos[sorted_idx])
sorted_idx = np.array([sorted_idx[i] for i in range(N) if i not in removed])

for i in range(sorted_idx.shape[0]):
    pos = all_pos[sorted_idx[i]]
    pos[1] = -i * 0.002  -0.1
    cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=0.4)
    cyld_0.rotate(cyld_0.get_rotation_matrix_from_xzy(np.array([pi/2,0,0])), center=[0,0,0])
    cyld_0.translate(pos, relative=True)
    cyld_0.paint_uniform_color([1,0,0])
    mesh.append(copy.deepcopy(cyld_0))
o3d.visualization.draw_geometries(mesh)


# for i in range(all_pos.shape[0]):
#     pos = all_pos[i]
#     pos[1] = -i * 0.002  -0.1

#     cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=0.4)
#     cyld_0.rotate(cyld_0.get_rotation_matrix_from_xzy(np.array([pi/2,0,0])), center=[0,0,0])
#     cyld_0.translate(pos, relative=True)
#     cyld_0.paint_uniform_color([1,0,0])

#     mesh.append(copy.deepcopy(cyld_0))

# o3d.visualization.draw_geometries(mesh)