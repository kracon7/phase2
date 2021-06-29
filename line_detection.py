import os
import sys
import argparse
import numpy as np
import open3d as o3d
import math
import copy

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils.corn_utils import find_stalks


def draw_frame(origin, q, scale=1):
    # Input quaternion format: qx qy qz qw
    # open3d quaternion format qw qx qy qz

    o3d_quat = np.array([q[3], q[0], q[1], q[2]])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(o3d_quat))
    
    return frame_rot


def compute_point_cloud(corn_img, K, rgb=None):
    '''
    compute the point cloud in the camera frame
    Input
    	corn_img -- ndarray of shape (im_h, im_w)
    	K -- camera intrinsics (3, 3)
    Output
    	pc_cam -- pointcloud in camera frame, (im_h*im_w, 3)
    '''

    # pixels of image grid
    im_h, im_w = corn_img.shape[:2]
    x, y = np.arange(im_w), np.arange(im_h)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy], axis=2).reshape(-1,2)

    # project pixels
    rays = np.insert(points, 2, 1, axis=1) @ np.linalg.inv(K).T
    pc_cam = rays.reshape(im_h, im_w, 3) * np.expand_dims(corn_img, axis=-1)  # im_h x im_w x 3
    
    return pc_cam.reshape(-1, 3)

def visualize_pcd(cloud):
    '''
    visualize the point cloud and the origin coordiante frame
    Input
    	pcd -- pointcloud (N, 3)
    	color -- rgb channel  (im_h, im_w, 3)
    '''
    q0 = np.array([0., 0., 0., 1.])
    frame_base = draw_frame(np.zeros(3), q0, scale=0.2)

    o3d.visualization.draw_geometries([cloud, frame_base])


def angle_calculate_pcd(cloud, mask=None, rnsc_thresh=0.02, rnsc_iter=200):
    '''
    Input
    	pcd -- pointcloud (N, 3)
    '''
    if mask is not None:
        cloud = cloud.select_down_sample(mask)
    plane_model, inliers = cloud.segment_plane(distance_threshold=rnsc_thresh,
                                         ransac_n=3,
                                         num_iterations=rnsc_iter)
    [a1, b1, c1, d1] = plane_model
    print("Plane equation: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(a1, b1, c1, d1))
    a2, b2, c2, d2 = 0, 0, 1, 0
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    print("Angle is", A, "degree")
    return plane_model, inliers

def points2pixel(points, K):
    '''
    Project points back to image plane
    Input
        points -- 3D coordinates (m, 3)
    Output
        pixel_coord -- pixel coordinates for corresponding points (m, 2)
    '''
    ray = points / points[:, -1:]
    pixel_coord = ray @ K.T
    pixel_coord = np.rint(pixel_coord[:, :2]).astype('int')
    return pixel_coord


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
    corn_img_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - ws_limits[0][0]) / resol_x).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - ws_limits[1][0]) / resol_y).astype(int)
    corn_img_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = ws_limits[2][0]
    corn_img_heightmap = corn_img_heightmap - z_bottom
    corn_img_heightmap[corn_img_heightmap < 0] = 0
    # corn_img_heightmap[corn_img_heightmap == -z_bottom] = np.nan
    return corn_img_heightmap

def detect_lines(img, minLineLength=100, bin_width=16):
    im_h, im_w = img.shape[:2]
    img = img[int(im_h/2):im_h, 0:im_w]

    # this is for image closing
    kernel = np.ones((37,37),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # used img instead of closing
    edges = cv2.Canny(img,50,150,apertureSize = 3)

    # LineLength determined based on observation
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=50,lines=np.array([]), 
                            minLineLength=minLineLength,maxLineGap=80)

    a,b,c = lines.shape
    cent_pos = []
    for i in range(a):
        # [0][0], and [0][2] are x (column locations)
        angle =  math.atan((lines[i][0][1]-lines[i][0][3])/(lines[i][0][0]-lines[i][0][2]))
        # angle between 75 and 90 degree
        if  (angle > 1.3 and angle <= math.pi/2) or (angle < -1.3 and angle >= -math.pi/2) and abs((lines[i][0][1]-lines[i][0][3]))>100:
          cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
          #print(i,(lines[i][0][0], lines[i][0][2]))
          cent_pos.append((lines[i][0][0]+lines[i][0][2])/2.0)
    cent_pos.sort()
    cent_pos = np.array(cent_pos)

    hist, bin_edges = np.histogram(cent_pos, bins=int(im_w/bin_width), range=(0, im_w))
    return hist, bin_edges



def main(args):

    os.system('mkdir -p %s'%(args.output_dir))
    flist = os.listdir(args.load_from)
    flist.sort()
    num_images = len(flist)

    fig, ax = plt.subplots(2,1, sharex='col')
    plt.subplots_adjust(wspace=0, hspace=0)

    for img_index in range(num_images):
        print(flist[img_index])
        corn_img = cv2.imread(os.path.join(args.load_from, flist[img_index]))

        im_h, im_w, _ = corn_img.shape
        if corn_img.shape[0] == 720:
            # camera intrinsics for Realsense D455 at 720 x 1280 resolution
            K=np.array([[643.014,      0.0,  638.611],
                        [0.0,      643.014,  365.586],
                        [0.0,          0.0,    1.0]])
        elif corn_img.shape[0] == 480:
            # camera intrinsics for Realsense D455 at 480 x 848 resolution
            K=np.array([[425.997,      0.0,  423.08],
                        [0.0,      425.997,  243.701],
                        [0.0,          0.0,    1.0]])

        bin_width = 16
        line_hist, bin_edges = detect_lines(corn_img, bin_width=bin_width)

        ax[1].set_ylim(0, 5)

        ax[0].imshow(corn_img, aspect="auto")
        ax[1].plot(bin_edges[:-1]+bin_width/2, line_hist)

        ax[0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[1].tick_params(bottom=False, labelbottom=False)

        plt.savefig(os.path.join(args.output_dir, flist[img_index]), bbox_inches='tight')

        ax[0].clear()
        ax[1].clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test corn_img filter and projection for corn images')
    parser.add_argument('--load_from', default='jiacheng/data', help='directory to load segmented corn images')
    parser.add_argument('--output_dir', default='line_detection', help='directory to output bayesian filtered results')
    parser.add_argument('--stitch', default=0, type=int, help='sticth the corn segmentation and original picture together')
    args = parser.parse_args()
    
    main(args)