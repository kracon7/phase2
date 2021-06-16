import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


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

    if rgb is not None:
        pc_cam = np.concatenate([pc_cam, rgb], axis=2).reshape(-1, 6)
    else:
        pc_cam = pc_cam.reshape(-1, 3)
    
    return pc_cam


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
    plt.ion()
    if args.color:
        fig, ax = plt.subplots(3,1)
    else:
        fig, ax = plt.subplots(2,1)

    # start from index idx
    idx = 100

    # check if it is compressed files or not
    flist = os.listdir(args.load_from)
    if '.npy' in flist[0]:
        f_format = 'npy'
    else:
        f_format = 'npz'

    for i in range(50):
        fname = 'depth_%07d.'%(idx) + f_format
        print(fname)
        depth = np.load(os.path.join(args.load_from, fname))
        if f_format == 'npz':
            depth = depth['arr_0']

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
        heightmap = points2heightmap(points, [720, 1280])

        # load color image
        if args.color:
            color = np.load(os.path.join(args.load_from, 'color_%07d.npy'%(idx)))

        ax[0].imshow(depth, vmax=0.8)
        ax[1].imshow(heightmap)
        if args.color:
            ax[2].imshow(color)

        a = input()
        idx += 100
        ax[0].clear()
        ax[1].clear()
        if args.color:
            ax[2].clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test depth filter and projection for corn images')
    parser.add_argument('--load_from', default='jiacheng/data', help='directory to load images')
    parser.add_argument('--color', default=0, type=int, help='load color or not')
    args = parser.parse_args()
    
    main(args)