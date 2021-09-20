import open3d as o3d
import numpy as np

cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
pcd = o3d.io.read_point_cloud('side_view.pcd')
pts = np.asarray(pcd.points)
N = pts.shape[0]
print('Original pcd has %d points'%(N))

o3d.visualization.draw_geometries([pcd, cam_frame])

pcd.rotate(pcd.get_rotation_matrix_from_zyx(np.array([0, 0.02,0])), center=[0,0,0])
    
pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-100, -100, 0.2]), 
                                                   np.array([100, 100, 0.5])))
pts = np.asarray(pcd.points)
mask = (pts[:,1] > 0) | (pts[:,2] > 0.32)
print('%d points left after cropping'%(np.sum(mask)))
ind = np.where(mask)[0]
pcd = pcd.select_by_index(ind)

pcd.rotate(pcd.get_rotation_matrix_from_zyx(np.array([0, -0.02,0])), center=[0,0,0])
o3d.visualization.draw_geometries([pcd, cam_frame])
    
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50,
#                                          std_ratio=2.0)
cl, ind = pcd.remove_radius_outlier(nb_points=200, radius=0.05)
pcd = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd, cam_frame])

o3d.io.write_point_cloud('side_view.pcd', pcd)
