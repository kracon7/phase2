import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class PointCloud():
    def __init__(self, vis=False):
        self.point_cloud = None
        self.voxel_size = 0.005
        self.visualize = vis

    def merge(self, new_points, new_colors):
        '''
        merge new point array into existing point cloud
        '''
        if new_colors.dtype == 'uint8':
            new_colors = new_colors.astype('float') / 255
            
        if self.point_cloud is None:
            self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_points))
            self.point_cloud.colors = o3d.utility.Vector3dVector(new_colors)
        else:
            exising_points = np.asarray(self.point_cloud.points)
            exising_colors = np.asarray(self.point_cloud.colors)

            points = np.concatenate((exising_points, new_points), axis=0)
            colors = np.concatenate((exising_colors, new_colors), axis=0)

            self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

            self.point_cloud.voxel_down_sample(voxel_size=self.voxel_size)

        if self.visualize:
            o3d.visualization.draw_geometries([self.point_cloud])

    def save_as_mesh(self, fpath, alpha=0.01):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                            self.point_cloud, alpha)
        o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh(fpath, mesh)

    def transform_side_to_map(self, points, pose):
        '''
        Input:
            points -- numpy array (N x 3)
            pose -- position and quaternion qx qy qz qw, correspoinding to T_map_1
        '''
        p1, q1 = pose[:3], pose[3:]
        R1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
        # rot, C = R1.T, -R1.T @ p1

        rot, C = R1, p1

        new_points = points @ rot.T + C
        return new_points

    def depth_to_points(self, depth, K):
        im_h, im_w = depth.shape
        x, y = np.arange(im_w), np.arange(im_h)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx, yy], axis=2).reshape(-1,2)
        rays = np.dot(np.insert(points, 2, 1, axis=1), np.linalg.inv(K).T).reshape(im_h, im_w, 3)
        points = rays * depth.reshape(im_h, im_w, 1)
        return points.reshape(-1,3)