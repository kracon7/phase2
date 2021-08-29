import os
import sys
import numpy as np
import cv2
from scipy.spatial import cKDTree


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    x1, x2, ind1 = [], [], []
    ratio = 0.7
    tree1, tree2 = cKDTree(des1), cKDTree(des2)

    N = loc1.shape[0]

    for i in range(N):
        ft_1 = des1[i]
        dd, ii = tree2.query([ft_1], k=2, n_jobs=-1)
        if dd[0,0] / dd[0,1] < ratio:
            # the correspoinding index of matched feature in des2 from des1
            idx2 = ii[0, 0]
            
            # query back from feature 2 to tree1
            ft_2 = des2[idx2]
            dd, ii = tree1.query([ft_2], k=2, n_jobs=-1)
            if dd[0,0] / dd[0,1] < ratio:
                if ii[0, 0] == i:
                    x1.append(loc1[i])
                    x2.append(loc2[idx2])
                    ind1.append(i)
    x1 = np.stack(x1)
    x2 = np.stack(x2)
    ind1 = np.array(ind1)
    return x1, x2, ind1

def load_data(data_dir, frame_idx):
	front_rgbd_dir = os.path.join(data_dir, 'front_rgbd')
	side_color_dir = os.path.join(data_dir, 'side_color')
	transform_dir = os.path.join(data_dir, 'transform')

	data = {'front_rgbd': np.load(os.path.join(front_rgbd_dir, 'frame_%07d.npy'%(frame_idx))),
			'side_color': np.load(os.path.join(side_color_dir, 'frame_%07d.npy'%(frame_idx))),
			'transform': np.load(os.path.join(transform_dir, 'frame_%07d.npy'%(frame_idx))),}
	return data


ROOT = os.path.dirname(os.path.abspath(__file__))
frame_idx_1 = 1
frame_idx_2 = 21
data_dir = '/home/jc/tmp/pred_distance'

frame1 = load_data(data_dir, frame_idx_1)
frame2 = load_data(data_dir, frame_idx_2)