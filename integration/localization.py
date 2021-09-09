import scipy
import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_loc_3d(side_plane, ground_plane, corn_id_bbox, K):
    '''
    line_plane intersection to get the corn 
    Input:
        side_plane -- plane equation [sa, sb, sc, sd]
        ground_plane -- plane equation [ga, gb, gc, gd]
        corn_id_bbox -- dict object for bounding boxed, corn id is key
        K -- intrinsic matrix
    '''
    ns = np.array(side_plane[:3])     # side plane normal
    ng = np.array(ground_plane[:3])   # ground plane normal

    corn_idx = corn_id_bbox.keys()
    loc_3d = {}
    for idx in corn_idx:
        bbox = corn_id_bbox[idx]
        center = [ (bbox[0][0] + bbox[1][0])/2, (bbox[0][1] + bbox[1][1])/2 ]
        ray = np.linalg.inv(K) @ np.array([center[0], center[1], 1])
        pt = - side_plane[3] / (ns @ ray) * ray
        loc_3d[idx] = pt

    return loc_3d

def merge_measurements(history, loc_3d, pose):
    '''
    merge single measurement with all history measurements
    Input:
        hist -- history measurements
        corn_id_bbox -- dict object for bounding boxed, corn id is key
        pose -- position and quaternion of current camera frame
    '''
    p1, q1 = pose[:3], pose[3:]
    R1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
    rot, C = R1.T, -R1.T @ p1
    
    for idx in loc_3d.keys():
        pt = loc_3d[idx]
        pt_map = rot @ pt + C 

        history[idx].append(pt_map)
