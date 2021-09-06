import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import torch
import torchvision.transforms as Transforms

class PlaneEstimator():
    def __init__(self, args, rcnn_model):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = rcnn_model
        self.CLASS_NAMES = ["__background__", "corn_stem"]
        self.torch_trans = Transforms.Compose([Transforms.ToTensor()])
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher()
        self.frame_buffer = []
        
        # compute rays in advance
        self.K = np.array([[615.311279296875,   0.0,             430.1778869628906],
                           [  0.0,            615.4699096679688, 240.68307495117188],
                           [  0.0,              0.0,               1.0]])
        self.im_w = 848
        self.im_h = 480
        x, y = np.arange(self.im_w), np.arange(self.im_h)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx, yy], axis=2).reshape(-1,2)
        self.rays = np.dot(np.insert(points, 2, 1, axis=1), np.linalg.inv(self.K).T).reshape(self.im_h, self.im_w, 3)
        # intrinsic matrix for realsense d435 480 x 848

    def get_rel_trans(self, pose1, pose2):
        '''
        Compute the relative transformation between frame1 and frame2
        Input
            frame1 -- dictionary object, stores front rgbd, side color, absolute transformation
        Output
            T -- transformation matrix from frame2 to frame1
        '''
        p1, q1 = pose1[:3], pose1[3:]
        p2, q2 = pose2[:3], pose2[3:]
        R1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
        R2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]]).as_matrix()
        
        T_map_1, T_2_map = np.eye(4), np.eye(4)
        T_map_1[:3,:3], T_map_1[:3,3] = R1.T, -R1.T @ p1
        T_2_map[:3,:3], T_2_map[:3,3] = R2, p2
        T = T_2_map @ T_map_1
        return T

    def get_bbox(self, img, confidence=0.8):
        '''
        Get the bounding box for side view corn detection
        Input
            img -- numpy array of rgb image
        Output
            bbox -- list object, bounding box position and sise
        '''
        img = self.torch_trans(img).to(self.device)
        pred = self.model([img])
        pred_class = [self.CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
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

    def bbox_to_mask(self, bbox, im_h, im_w):
        '''
        generate binary mask according to bounding boxes for feature detection
        '''
        mask = np.zeros((im_h, im_w), dtype='uint8')
        for box in bbox:
            top, bottom, left, right = int(box[0][1]), int(box[1][1]), int(box[0][0]), int(box[1][0])
            mask[top:bottom, left:right] = 1
        return mask

    def estimate_distance(self, kp1, kp2, T, normal):
        '''
        Least square estimation of plane distance 
        Input
            kp1 -- matched key points of frame1
            kp2 -- matched key points of frame2
            K -- camera intrinsic matrix
            T -- camera pose transformation, from frame2 to frame1
            normal -- estimated plane normal direction, numpy array
        '''
        K = self.K
        A, b = [], []
        R, C = T[:3, :3], T[:3, 3]
        S = K @ C
        E = K @ R @ np.linalg.inv(K)                    # 3 x 3
        F = (normal @ np.linalg.inv(K)).reshape(1,3)    # 1 x 3
        X1 = np.insert(kp1, 2, 1, axis=1).T             # 3 x N
        L = E @ X1 / (F @ X1)                           # 3 x N
        A = kp2[:,0] * L[2] - L[0]
        b = kp2[:,0] * S[2] - S[0]
        d = 1/(A@A) * A @ b
        
        return d

    def estimate_distance_ransac(self, kp1, kp2, T, normal, ransac_thr, ransac_iter):
        '''
        use RANSAC to esitmate plane distance
        '''
        K = self.K
        R, C = T[:3, :3], T[:3, 3]
        S = K @ C
        E = K @ R @ np.linalg.inv(K)                    # 3 x 3
        F = (normal @ np.linalg.inv(K)).reshape(1,3)    # 1 x 3
        X1 = np.insert(kp1, 2, 1, axis=1).T             # 3 x N
        L = E @ X1 / (F @ X1)                           # 3 x N
        
        num_ft, num_choice = kp1.shape[0], 8
        max_inlier = 0
        d_result = 0
        if num_choice >= num_ft:
            d_result = self.estimate_distance(kp1, kp2, T, normal)
        else:
            for i in range(ransac_iter):
                sample_idx = np.random.choice(num_ft, num_choice) 
                sampled_kp1 = kp1[sample_idx]
                sampled_kp2 = kp2[sample_idx]

                d = self.estimate_distance(sampled_kp1, sampled_kp2, T, normal)

                # compute reprojection error and count inliers
                u2_err = (-d * L[0] + S[0]) / (-d * L[2] + S[2]) - kp2[:,0]
                n_inlier = np.sum(np.abs(u2_err) < 5)
                if n_inlier > max_inlier:
                    max_inlier = n_inlier
                    d_result = d
                
                # if no outliers then break loop
                if max_inlier == num_ft:
                    break

        u2_err = (-d_result * L[0] + S[0]) / (-d_result * L[2] + S[2]) - kp2[:,0]
        inliers = u2_err < 5
        dpx = np.average(kp1[inliers, 0] - kp2[inliers, 0])
        print("translation in x: %8.4f average dx: %5.1f, ratio: %10.2f, d_result: %8.4f "%(
                T[0,3], dpx, dpx/T[0,3], -d_result*100), end='')

        return d_result

    def find_ground_plane(self, xyzrgb):
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

    def process_frames(self, frame1, frame2):
        '''
        process two frames to compute corn plane equation
        '''
        pose1, pose2 = frame1.pose, frame2.pose
        rel_trans = self.get_rel_trans(pose1, pose2)

        bbox1, pred_cls_1, pred_score_1 = self.get_bbox(frame1.side_color)
        bbox2, pred_cls_2, pred_score_2 = self.get_bbox(frame2.side_color)

        # create sift feature
        mask1 = self.bbox_to_mask(bbox1, 480, 848)
        mask2 = self.bbox_to_mask(bbox2, 480, 848)
        kp1, des1 = self.sift.detectAndCompute(frame1.side_color, mask1)
        kp2, des2 = self.sift.detectAndCompute(frame2.side_color, mask2)

        # match sift key points
        matches = self.bf_matcher.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append([m])

        if len(good) > 5:
            src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,2)
            dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,2)  

            points = self.rays * frame1.front_depth.reshape(self.im_h, self.im_w, 1)
            front_xyzrgb = np.concatenate([points, frame1.front_color], axis=2)
            R_ground, d_ground = self.find_ground_plane(front_xyzrgb)
            ground_normal = R_ground[2]

            d_plane = self.estimate_distance_ransac(src_pts, dst_pts, rel_trans, ground_normal, 5, 10)

            print(d_plane)
            # draw grid in side view images
            d_ground_side = - d_ground - 0.099
            drawn_side_img = draw_side_plane(frame1['side_color'], K, R_ground, d_plane, d_ground_side)

            # draw vanishing point z in front view images
            ground_z = R_ground[2]
            vpz = (K @ (ground_z / ground_z[2]))[:2]
            drawn_front_img = cv2.circle(frame1['front_rgbd'][:,:,3:].astype('uint8'), (int(vpz[0]), int(vpz[1])), 4, (0,255,255), 2)

            drawn_img = np.concatenate([drawn_side_img, drawn_front_img], axis=0)
            cv2.imwrite(os.path.join(output_dir, 'frame_%07d.png'%(i)), drawn_img)

