import numpy as np
import open3d as o3d

def find_ground_plane(xyzrgb):
	'''
	Use RANSAC and PCA to detect ground plane
	Input
		xyzrgb -- numpy array of point cloud
	Output
		R -- [x_axis, y_axis, z_axis], 
			 y is ground plane normal pointing up
			 z is corn line vector pointing front
	'''
	xyzrgb = xyzrgb.reshape(-1,6)
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzrgb[:,:3]))
	pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:].astype('float') / 255)
	# o3d.visualization.draw_geometries([pcd, cam_frame])

	pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-0.6, -0.2, 0]), 
	                                                   np.array([0.6, 0.35, 8])))
	# o3d.visualization.draw_geometries([pcd, cam_frame])

	pcd = pcd.voxel_down_sample(voxel_size=0.04)
	# o3d.visualization.draw_geometries([pcd, cam_frame])

	plane_model, inliers = pcd.segment_plane(distance_threshold=0.04,
	                                         ransac_n=3,
	                                         num_iterations=100)
	y_axis, d = plane_model[:3], plane_model[3]
	if y_axis[1] < 0:
	    y_axis = - y_axis
	    d = -d
	# display_inlier_outlier(pcd, inliers)

	# use ground points to do PCA to find z axis
	ground_points = pcd.select_by_index(inliers)
	_, cov = ground_points.compute_mean_and_covariance()
	eigen_values, eigen_vectors = np.linalg.eig(cov)
	z_axis = eigen_vectors[:,0]
	if z_axis[2] < 0:
	    z_axis = -z_axis

	x_axis = np.cross(y_axis, z_axis)

	# rectify the point cloud
	R = np.stack([x_axis, y_axis, z_axis])

	return R, d
