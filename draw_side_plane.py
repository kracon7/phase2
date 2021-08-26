import os
import sys
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

def draw_corridor(img, label, color=(255,0,0)):
    '''
    draw two lines without vanishing point
    Input:
        img - RGB image
        label - [x1, y1, theta1, theta2]
    Output:
        image - 
    '''
    vpzx, vpzy = int(label[0]), int(label[1])
    theta1, theta2 = np.deg2rad(label[2]), np.deg2rad(label[3])
    l = 300
    # find pt1
    dx = l * np.cos(theta1)
    dy = l * np.sin(theta1)
    if dy < 0:
        dx, dy = -dx, -dy
    pt1 = (int(vpzx+dx), int(vpzy+dy))

    # find pt2
    dx = l * np.cos(theta2)
    dy = l * np.sin(theta2)
    if dy < 0:
        dx, dy = -dx, -dy
    pt2 = (int(vpzx+dx), int(vpzy+dy))

    img = cv2.line(img, (vpzx, vpzy), (pt1[0], pt1[1]), color, 2)
    img = cv2.line(img, (vpzx, vpzy), (pt2[0], pt2[1]), color, 2)

    return img

def draw_side_plane(img, K, pt_side, vx_side, vy_side, color=(255,255,0)):
    bottom_center = pt_side - (pt_side[0])/(vx_side[0]) * vx_side
    bottom_right = bottom_center + 0.2 * vx_side
    bottom_left = bottom_center - 0.2 * vx_side
    top_right = bottom_right - 0.2 * vy_side
    top_left = bottom_left - 0.2 * vy_side
    
    
    bottom = np.linspace(bottom_left, bottom_right, 6)
    top = np.linspace(top_left, top_right, 6)
    right = np.linspace(bottom_right, top_right, 6)
    left = np.linspace(bottom_left, top_left, 6)
    
    bottom_pixels = bottom @ K.T
    top_pixels = top @ K.T
    right_pixels = right @ K.T
    left_pixels = left @ K.T
    
    bottom_pixels = bottom_pixels / bottom_pixels[:,2].reshape(-1,1)
    top_pixels = top_pixels / top_pixels[:,2].reshape(-1,1)
    right_pixels = right_pixels / right_pixels[:,2].reshape(-1,1)
    left_pixels = left_pixels / left_pixels[:,2].reshape(-1,1)

    bottom_pixels = bottom_pixels[:,:2].astype('int')
    top_pixels = top_pixels[:,:2].astype('int')
    right_pixels = right_pixels[:,:2].astype('int')
    left_pixels = left_pixels[:,:2].astype('int')

    drawn_img = img.copy()
    color = (255,255,0)
    for p1, p2 in zip(bottom_pixels, top_pixels):
        drawn_img = cv2.line(drawn_img, (p1[0], p1[1]), (p2[0], p2[1]), color, 2)
    for p1, p2 in zip(right_pixels, left_pixels):
        drawn_img = cv2.line(drawn_img, (p1[0], p1[1]), (p2[0], p2[1]), color, 2)
    
    return drawn_img


K = np.array([[615.311279296875,   0.0,             430.1778869628906],
              [  0.0,            615.4699096679688, 240.68307495117188],
              [  0.0,              0.0,               1.0]])

front_dir = '/home/jc/tmp/front_rgbd'
side_dir = '/home/jc/tmp/side'
plane_dir = '/home/jc/tmp/plane'
drawn_dir = '/home/jc/tmp/drawn'

front_imgs = os.listdir(front_dir)
front_imgs.sort()

for i in range(len(front_imgs)):
    print(i)
    front = np.load(os.path.join(front_dir, front_imgs[i]))[:,:,3:].astype('uint8')
    side = np.load(os.path.join(side_dir, front_imgs[i]))
    plane = pickle.load(open(os.path.join(plane_dir, front_imgs[i].split('.')[0]+'.pkl'), 'rb'))
    pt, vy, vz = plane

    # pt[2] -

    # draw vanishing point and corn line
    vpz = K @ (vz / vz[2])
    vpz = vpz[:2].astype('int')

    pixel_pt = K @ (pt+vz)
    pixel_pt = (pixel_pt / pixel_pt[2]).astype('int')
    drawn_front = cv2.line(front, (pixel_pt[0], pixel_pt[1]), (vpz[0], vpz[1]), (255,255,0), 3)

    T = np.array([[ 0, 0, 1, 0.03858],
                  [ 0, 1, 0, -0.099],
                  [-1, 0, 0, 0.11184],
                  [ 0, 0, 0,  1]])
    pt_side = T[:3,:3] @ pt + T[:3,3]
    vy_side = - T[:3,:3] @ vy
    vx_side = T[:3,:3] @ vz
    drawn_side = draw_side_plane(side, K, pt_side, vx_side, vy_side)

    img = np.concatenate([drawn_front, drawn_side], axis=0)
    
    cv2.imwrite(os.path.join(drawn_dir, front_imgs[i].split('.')[0]+'.png'), img)