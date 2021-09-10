import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=5, suppress=True)

def draw_frame(origin=[0,0,0], q=[1,0,0,0], scale=1):
    # open3d quaternion format qw qx qy qz

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(
                mesh_frame.get_rotation_matrix_from_quaternion(q), center=origin)
    
    return frame_rot


class PointCloud():
    def __init__(self, voxel_size=0.005, vis=False):
        self.point_cloud = None
        self.voxel_size = voxel_size
        self.visualize = vis
        self.cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

    def merge(self, new_points, new_colors, pose):
        '''
        merge new point array into existing point cloud
        '''
        if new_colors.dtype == 'uint8':
            new_colors = new_colors.astype('float') / 255
            
        if self.point_cloud is None:
            self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_points))
            self.point_cloud.colors = o3d.utility.Vector3dVector(new_colors)
            self.pose0 = pose
        else:
            rel_trans = self.get_rel_trans(self.pose0, pose)
            new_points = self.transform_points(new_points, rel_trans)
            exising_points = np.asarray(self.point_cloud.points)
            exising_colors = np.asarray(self.point_cloud.colors)

            points = np.concatenate((exising_points, new_points), axis=0)
            colors = np.concatenate((exising_colors, new_colors), axis=0)

            self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

            self.point_cloud.voxel_down_sample(voxel_size=self.voxel_size)

        if self.visualize:
            o3d.visualization.draw_geometries([self.point_cloud, self.cam_frame])

    def save_as_mesh(self, fpath, alpha=0.01):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                            self.point_cloud, alpha)
        o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh(fpath, mesh)

    def get_rel_trans(self, pose1, pose2):
        '''
        Compute the relative transformation between frame1 and frame2
        Input
            pose1 -- position and quaternion
        Output
            T -- transformation matrix from frame2 to frame1
        '''
        p1, q1 = pose1[:3], pose1[3:]
        p2, q2 = pose2[:3], pose2[3:]
        R1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
        R2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]]).as_matrix()
        
        T_1_map, T_map_2 = np.eye(4), np.eye(4)
        T_1_map[:3,:3], T_1_map[:3,3] = R1, p1
        T_map_2[:3,:3], T_map_2[:3,3] = R2.T, -R2.T @ p2
        T = T_1_map @ T_map_2
        return T

    def transform_points(self, points, T):
        '''
        Input:
            points -- numpy array (N x 3)
            pose -- position and quaternion qx qy qz qw, correspoinding to T_1_map
        '''
        rot, C = T[:3,:3], T[:3,3]

        new_points = points @ rot.T + C
        return new_points

    def depth_to_points(self, depth, K, clamp=1):
        im_h, im_w = depth.shape
        x, y = np.arange(im_w), np.arange(im_h)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx, yy], axis=2).reshape(-1,2)
        rays = np.dot(np.insert(points, 2, 1, axis=1), np.linalg.inv(K).T).reshape(im_h, im_w, 3)
        points = rays * depth.reshape(im_h, im_w, 1)

        mask = ((depth < clamp) & (depth > 0)).reshape(-1)
        return points.reshape(-1,3), mask

