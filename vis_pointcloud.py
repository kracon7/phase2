import os
import argparse
import numpy as np
import open3d as o3d
import math
import copy
from PIL import Image
import matplotlib.pyplot as plt

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
    pixel_coord = pixel_coord[:, :2].astype('int')
    return pixel_coord

def main(args):

    img_index = 100
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

    plane_ground, ind = angle_calculate_pcd(ground_candid_cloud)

    ground_inlier_ind = ground_candid_ind[np.array(ind)]

    ground_inlier_cloud = pcd.select_down_sample(ground_inlier_ind)
    ground_outlier_cloud = pcd.select_down_sample(ground_inlier_ind, invert=True)

    if args.sementics:
        print("Showing outliers (red) and inliers (gray): ")
        ground_outlier_cloud.paint_uniform_color([1, 0, 0])
        ground_inlier_cloud.paint_uniform_color([0.6, 0.3, 0])
        o3d.visualization.draw_geometries([ground_inlier_cloud, ground_outlier_cloud])
    else:
        o3d.visualization.draw_geometries([ground_inlier_cloud])

    # draw ground plane
    [a1, b1, c1, d1] = plane_ground
    # xy coordinates
    xy = np.array([[0.55, 0.2],
                   [0.55, 0.16],
                   [0.55, 0.12],
                   [0.55, 0.08],
                   [0.55, 0.04],
                   [-0.55, 0.04],
                   [-0.55, 0.08],
                   [-0.55, 0.12],
                   [-0.55, 0.16],
                   [-0.55, 0.2]])
    z = (-d1 - a1 * xy[:,0] - b1 * xy[:,1]) / c1
    coord = np.hstack([xy, z.reshape(-1, 1)])
    # draw plane ground
    lines = [
        [0, 4],
        [4, 5],
        [5, 9],
        [9, 0],
        [1, 8],
        [2, 7],
        [3, 6]]
    colors = [[0, 0, 1] for i in range(len(lines))]
    line_ground = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(coord),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_ground.colors = o3d.utility.Vector3dVector(colors)

    # fit corn plane
    corn_candid_cloud = ground_outlier_cloud
    plane_corn, ind = angle_calculate_pcd(corn_candid_cloud, rnsc_thresh=0.04)

    corn_inlier_cloud = corn_candid_cloud.select_down_sample(ind)
    corn_outlier_cloud = corn_candid_cloud.select_down_sample(ind, invert=True)

    if args.sementics:
        print("Showing outliers (red) and inliers (gray): ")
        corn_outlier_cloud.paint_uniform_color([1, 0, 0])
        corn_inlier_cloud.paint_uniform_color([0.2, 0.8, 0.2])
        o3d.visualization.draw_geometries([ground_inlier_cloud, corn_inlier_cloud, corn_outlier_cloud])
    else:
        o3d.visualization.draw_geometries([corn_inlier_cloud])

    # draw corn plane
    [a1, b1, c1, d1] = plane_corn
    # xy coordinates
    xy = np.array([[0.55, 0.15],
                   [0.55, 0.05],
                   [0.55, -0.05],
                   [0.55, -0.15],
                   [0.55, -0.26],
                   [-0.55, -0.26],
                   [-0.55, -0.15],
                   [-0.55, -0.05],
                   [-0.55, 0.05],
                   [-0.55, 0.15]])
    z = (-d1 - a1 * xy[:,0] - b1 * xy[:,1]) / c1
    coord = np.hstack([xy, z.reshape(-1, 1)])
    # draw plane ground
    lines = [
        [0, 4],
        [4, 5],
        [5, 9],
        [9, 0],
        [1, 8],
        [2, 7],
        [3, 6]]
    colors = [[0, 1, 0] for i in range(len(lines))]
    line_corn = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(coord),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_corn.colors = o3d.utility.Vector3dVector(colors)

    q0 = np.array([0., 0., 0., 1.])
    frame_base = draw_frame(np.zeros(3), q0, scale=0.2)

    if args.sementics:
        o3d.visualization.draw_geometries([ground_inlier_cloud, corn_inlier_cloud, corn_outlier_cloud, 
                                            frame_base, line_ground, line_corn])
    else:
        o3d.visualization.draw_geometries([corn_inlier_cloud, frame_base, line_ground, line_corn])
    o3d.visualization.draw_geometries([pcd, frame_base, line_ground, line_corn])

    # view segmented corns in image plane
    fig, ax = plt.subplots(1,1)
    corn_points = np.asarray(corn_inlier_cloud.points)
    pixel_coord = points2pixel(corn_points, K)
    corn_img = np.zeros((im_h, im_w, 3))
    corn_img[pixel_coord[:,1], pixel_coord[:,0], :] = np.asarray(corn_inlier_cloud.colors)
    ax.imshow(corn_img)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test depth filter and projection for corn images')
    parser.add_argument('--load_from', default='jiacheng/data', help='directory to load images')
    parser.add_argument('--image_index', default=100, type=int, help='image index to load rgb and depth')
    parser.add_argument('--sementics', default=0, type=int, help='add sementics color or not')
    args = parser.parse_args()
    
    main(args)