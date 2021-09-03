import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import torch
import torchvision.transforms as T
import argparse
from corridor_ransac import find_ground_plane

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

def quaternion_division(self, q, r):
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r

    tw = rw*qw + rx*qx + ry*qy + rz*qz
    tx = rw*qx - rx*qw - ry*qz + rz*qy
    ty = rw*qy + rx*qz - ry*qw - rz*qx
    tz = rw*qz - rx*qy + ry*qx - rz*qw
    return [tw, tx, ty, tz]

def load_data(data_dir, frame_idx):
    front_rgbd_dir = os.path.join(data_dir, 'front_rgbd')
    side_color_dir = os.path.join(data_dir, 'side_color')
    transform_dir = os.path.join(data_dir, 'transform')

    data = {'front_rgbd': np.load(os.path.join(front_rgbd_dir, 'frame_%07d.npy'%(frame_idx))),
            'side_color': np.load(os.path.join(side_color_dir, 'frame_%07d.npy'%(frame_idx))),
            'transform': np.load(os.path.join(transform_dir, 'frame_%07d.npy'%(frame_idx))),}
    return data

def get_rel_trans(frame1, frame2):
    '''
    Compute the relative transformation between frame1 and frame2
    Input
        frame1 -- dictionary object, stores front rgbd, side color, absolute transformation
    Output
        T -- transformation matrix from frame2 to frame1
    '''
    trans1 = frame1['transform']
    trans2 = frame2['transform']
    p1, q1 = trans1[:3], trans1[3:]
    p2, q2 = trans2[:3], trans2[3:]
    R1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
    R2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]]).as_matrix()
    
    T_map_1, T_2_map = np.eye(4), np.eye(4)
    T_map_1[:3,:3], T_map_1[:3,3] = R1.T, -R1.T @ p1
    T_2_map[:3,:3], T_2_map[:3,3] = R2, p2
    T = T_2_map @ T_map_1
    return T

def get_bbox(model, frame, confidence=0.8):
    '''
    Get the bounding box for side view corn detection
    Input
        model -- pytorch model object
        frame -- dictionary object, stores front rgbd, side color, absolute transformation
    Output
        bbox -- list object, bounding box position and sise
    '''
    transform = T.Compose([T.ToTensor()])
    img = transform(frame['side_color']).to(device)
    pred = model([img])
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    if len([x for x in pred_score if x>confidence])!=0:
        pred_t = [pred_score.index(s) for s, c in zip(pred_score, pred_class) 
                    if s>confidence and c=='corn_stem'][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]
    else:
        pred_boxes, pred_class, pred_score = None, None, None

    return pred_boxes, pred_class, pred_score

def bbox_to_mask(bbox, im_h, im_w):
    '''
    generate binary mask according to bounding boxes for feature detection
    '''
    mask = np.zeros((im_h, im_w), dtype='uint8')
    for box in bbox:
        top, bottom, left, right = int(box[0][1]), int(box[1][1]), int(box[0][0]), int(box[1][0])
        mask[top:bottom, left:right] = 1
    return mask

def estimate_distance(kp1, kp2, K, T, normal):
    '''
    Least square estimation of plane distance 
    Input
        kp1 -- matched key points of frame1
        kp2 -- matched key points of frame2
        K -- camera intrinsic matrix
        T -- camera pose transformation, from frame2 to frame1
        normal -- estimated plane normal direction, numpy array
    '''
    A, b = [], []
    R, C = T[:3, :3], T[:3, 3]
    for pt1, pt2 in zip(kp1, kp2):
        u1, v1, u2, v2 = pt1[0], pt1[1], pt2[0], pt2[1]
        L = K @ R @ np.linalg.inv(K) @ np.array([u1,v1,1]) / \
            (normal @ np.linalg.inv(K) @ np.array([u1,v1,1]))
        S = K @ C

        A += [ u2*L[2] - L[0]]
        b += [ u2*S[2] - S[0]]

    A, b = np.stack(A), np.stack(b)
    d = 1/(A @ A) * A @ b
    return d

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="model/faster-rcnn-corn_bgr8_ep100.pt",
                help="path to the model")
parser.add_argument("-c", "--confidence", type=float, default=0.8, 
                help="confidence to keep predictions")
args = parser.parse_args()

CLASS_NAMES = ["__background__", "corn_stem"]
ROOT = os.path.dirname(os.path.abspath(__file__))

frame_idx_1 = 1
frame_idx_2 = 16
data_dir = '/home/jc/tmp/pred_distance'

frame1 = load_data(data_dir, frame_idx_1)
frame2 = load_data(data_dir, frame_idx_2)

rel_trans = get_rel_trans(frame1, frame2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load(os.path.join(ROOT, args.model))
model.to(device)

bbox1, pred_cls_1, pred_score_1 = get_bbox(model, frame1)
bbox2, pred_cls_2, pred_score_2 = get_bbox(model, frame2)

# # create orb descriptor function
# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(frame1['side_color'], None)
# kp2, des2 = orb.detectAndCompute(frame2['side_color'], None)

# create sift feature
sift = cv2.SIFT_create()
mask1 = bbox_to_mask(bbox1, 480, 848)
mask2 = bbox_to_mask(bbox2, 480, 848)
kp1, des1 = sift.detectAndCompute(frame1['side_color'], mask1)
kp2, des2 = sift.detectAndCompute(frame2['side_color'], mask2)

# # visualize masked features
# img1 = cv2.drawKeypoints(frame1['side_color'], kp1, None, color=(0,255,0), flags=0)
# img2 = cv2.drawKeypoints(frame2['side_color'], kp2, None, color=(0,255,0), flags=0)
# img = np.concatenate([img1, img2], axis=0)
# plt.imshow(img)
# plt.show()


# match sift key points
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,2)
dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,2)  
# # visualize matched pairs
# img3 = cv2.drawMatchesKnn(frame1['side_color'], kp1, frame2['side_color'], kp2, 
#             good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

R = find_ground_plane(frame1['front_rgbd'])
ground_normal = R[2]

K = np.array([[615.311279296875,   0.0,             430.1778869628906],
              [  0.0,            615.4699096679688, 240.68307495117188],
              [  0.0,              0.0,               1.0]])

d = estimate_distance(src_pts, dst_pts, K, rel_trans, ground_normal)

print(d)