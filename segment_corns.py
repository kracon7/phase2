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


def compute_point_cloud(depth, K, rgb=None):
    '''
    compute the point cloud in the camera frame
    Input
    	depth -- ndarray of shape (im_h, im_w)
    	K -- camera intrinsics (3, 3)
    Output
    	pc_cam -- pointcloud in camera frame, (im_h*im_w, 3)
    '''

    # pixels of image grid
    im_h, im_w = depth.shape[:2]
    x, y = np.arange(im_w), np.arange(im_h)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy], axis=2).reshape(-1,2)

    # project pixels
    rays = np.insert(points, 2, 1, axis=1) @ np.linalg.inv(K).T
    pc_cam = rays.reshape(im_h, im_w, 3) * np.expand_dims(depth, axis=-1)  # im_h x im_w x 3
    
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
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - ws_limits[0][0]) / resol_x).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - ws_limits[1][0]) / resol_y).astype(int)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = ws_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    # depth_heightmap[depth_heightmap == -z_bottom] = np.nan
    return depth_heightmap

def main(args):

    os.system('mkdir -p %s'%(args.output_dir))
    flist = os.listdir(args.load_from)
    num_images = int(len(flist) / 3)

    for img_index in range(num_images):
        print(img_index)
        depth = np.load(os.path.join(args.load_from, 'depth_%07d.npy'%(img_index)))
        rgb = np.load(os.path.join(args.load_from, 'color_%07d.npy'%(img_index))).astype('float')/255

        im_h, im_w = depth.shape
        if depth.shape[0] == 720:
            # camera intrinsics for Realsense D455 at 720 x 1280 resolution
            K=np.array([[643.014,      0.0,  638.611],
                        [0.0,      643.014,  365.586],
                        [0.0,          0.0,    1.0]])
        elif depth.shape[0] == 480:
            # camera intrinsics for Realsense D455 at 480 x 848 resolution
            K=np.array([[425.997,      0.0,  423.08],
                        [0.0,      425.997,  243.701],
                        [0.0,          0.0,    1.0]])

        points = compute_point_cloud(depth, K)

        # filter in camera z-axis
        z_mask = (points[:,2]>0.2) & (points[:,2]<0.8)
        points = points[z_mask]
        rgb = rgb.reshape(-1, 3)[z_mask]
        
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # plot ground plane
        ground_candid_ind = np.where((points[:,1]>0) & (points[:,1]<0.4))[0]
        ground_candid_cloud = pcd.select_down_sample(ground_candid_ind)

        plane_ground, ind = angle_calculate_pcd(ground_candid_cloud, rnsc_thresh=0.03)
        a1, b1, c1, d1 = plane_ground

        ground_inlier_ind = ground_candid_ind[np.array(ind)]

        ground_inlier_cloud = pcd.select_down_sample(ground_inlier_ind)
        ground_outlier_cloud = pcd.select_down_sample(ground_inlier_ind, invert=True)

        if args.vis_pcd:
            if args.semantics:
                print("Showing outliers (red) and inliers (gray): ")
                ground_outlier_cloud.paint_uniform_color([1, 0, 0])
                ground_inlier_cloud.paint_uniform_color([0.6, 0.3, 0])
                o3d.visualization.draw_geometries([ground_inlier_cloud, ground_outlier_cloud])
            else:
                o3d.visualization.draw_geometries([ground_inlier_cloud])

        # fit corn plane
        corn_candid_cloud = ground_outlier_cloud
        plane_corn, ind = angle_calculate_pcd(corn_candid_cloud, rnsc_thresh=0.04)
        a2, b2, c2, d2 = plane_corn

        corn_inlier_cloud = corn_candid_cloud.select_down_sample(ind)
        corn_outlier_cloud = corn_candid_cloud.select_down_sample(ind, invert=True)

        if args.vis_pcd:
            if args.semantics:
                print("Showing outliers (red) and inliers (gray): ")
                corn_outlier_cloud.paint_uniform_color([1, 0, 0])
                corn_inlier_cloud.paint_uniform_color([0.2, 0.8, 0.2])
                o3d.visualization.draw_geometries([ground_inlier_cloud, corn_inlier_cloud, corn_outlier_cloud])
            else:
                o3d.visualization.draw_geometries([corn_inlier_cloud])

            q0 = np.array([0., 0., 0., 1.])
            frame_base = draw_frame(np.zeros(3), q0, scale=0.2)

            if args.semantics:
                o3d.visualization.draw_geometries([ground_inlier_cloud, corn_inlier_cloud, corn_outlier_cloud, 
                                                    frame_base])
            else:
                o3d.visualization.draw_geometries([corn_inlier_cloud, frame_base, line_ground, line_corn])

        # save segmented corns in image plane
        corn_points = np.asarray(corn_inlier_cloud.points)
        pixel_coord = points2pixel(corn_points, K)
        corn_img = np.zeros((im_h, im_w, 3)).astype('uint8')
        corn_img[pixel_coord[:,1], pixel_coord[:,0], :] = (255*np.asarray(corn_inlier_cloud.colors)).astype('uint8')
        
        # find corn stalk positions
        peak_ind = find_stalks(pixel_coord, im_h, im_w)
        # plot rectangles
        img = np.load(os.path.join(args.load_from, 'color_%07d.npy'%(img_index)))
        for ind in peak_ind:
            cv2.rectangle(img,(ind-10, 10),(ind+10, im_h-100),(0, 69,255),2)

        # find the intersection line between ground plane and corn plane
        fov_x = np.linalg.inv(K)[0] @ np.array([0, 0, 1])

        A1 = np.array([[a1, b1,    c1],
                       [a2, b2,    c2],
                       [1,   0, fov_x]])
        A2 = np.array([[a1, b1,    c1],
                       [a2, b2,    c2],
                       [1,   0, -fov_x]])
        # 3d point position where the intersection line intersect with the fov boundary
        pt1 = np.linalg.inv(A1) @ np.array([d1, d2, 0])
        pt2 = np.linalg.inv(A2) @ np.array([d1, d2, 0])

        pix1 = (K @ pt1 / pt1[2])[:2].astype('int')
        pix2 = (K @ pt2 / pt2[2])[:2].astype('int')

        cv2.line(img, (pix1[0], pix1[1]), (pix2[0], pix2[1]), (255, 0, 0), 2)

        if args.stitch:
            img = np.concatenate([img, corn_img], axis=0)
        cv2.imwrite(os.path.join(args.output_dir, 'corn_%07d.png'%(img_index)), corn_img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test depth filter and projection for corn images')
    parser.add_argument('--load_from', default='jiacheng/data', help='directory to load images')
    parser.add_argument('--output_dir', default='segmentation', help='directory to output segmented images')
    parser.add_argument('--semantics', default=0, type=int, help='add semantics color or not')
    parser.add_argument('--vis_pcd', default=0, type=int, help='visualize pointcloud or not')
    parser.add_argument('--stitch', default=0, type=int, help='sticth the corn segmentation and original picture together')
    args = parser.parse_args()
    
    main(args)