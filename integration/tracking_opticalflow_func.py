# Modified https://github.com/yashs97/object_tracker/blob/master/multi_label_tracking.py

import numpy as np
import argparse
import cv2 
import time
from imutils.video import FPS 
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import os
import random

import operator


CLASS_NAMES = ["__background__", "corn_stem"]


def corn_tracking_opticalflow(frame_dir, model_path = "./output/faster-rcnn-corn_bgr8_ep100.pt", frame_count=80, output_file=None):
    # Labels of Network.
    labels = { 0: 'background', 1: 'corn'}

    lk_params = dict(winSize = (50,50), maxLevel = 4, 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    total_frames = 1


    # prev_frame_number = format(int(frame_path[-11:-4])-1, '07d')
    # prev_frame_path = frame_path[:-11] + prev_frame_number
    # prev_frame = cv2.imread(prev_frame_path)
    # frame = cv2.imread(frame_path)

    prev_frame_number = format(1, '07d')
    prev_frame_path = os.path.join(frame_dir, 'frame_'+prev_frame_number+'.png')
    prev_frame = cv2.imread(prev_frame_path)


    fps = FPS().start()
    tracking_started = False

    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_file, fourcc, 100,(prev_frame.shape[1], prev_frame.shape[0]), True)

    # color_dict = dict()

    frame_number = 2

    corn_id_bbox_dict = dict()

    while True:
        frame = cv2.imread(os.path.join(frame_dir, 'frame_'+format(frame_number, '07d')+'.png'))
        if frame is None: #end of video file
            break
        # running the object detector every nth frame 
        if total_frames % int(frame_count)-1 == 0:
            pred_boxes, pred_class, pred_score = get_prediction(frame, model_path, 0.5)
            
            centroids = np.zeros([1, 1, 2], dtype=np.float32)

            # only if there are predictions
            if pred_boxes != None:
                corn_dict = dict()
                for i in range(len(pred_boxes)):
                    corn_dict[i]=dict()
                corn_dict['centroids']=dict()

                for i in range(len(pred_boxes)):
                # cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
                    color = list(np.random.random(size=3) * 256)
                    # print("i color", i, color)
                    tracking_id = int(i)
                    confidence = pred_score[i]

                    xLeftBottom = int(pred_boxes[i][0][0]) 
                    yLeftBottom = int(pred_boxes[i][0][1])
                    xRightTop   = int(pred_boxes[i][1][0])
                    yRightTop   = int(pred_boxes[i][1][1])

                    # print class and confidence          
                    label = pred_class[i] +": "+ str(confidence)             
                    # print(label)

                    x = (xLeftBottom + xRightTop)/2
                    y = (yLeftBottom + yRightTop)/2

                    corn_dict[i]['bbox'] = [(xLeftBottom,yLeftBottom),(xRightTop,yRightTop)]
                    corn_dict[i]['centroid'] =[(x,y)]
                    corn_dict['centroids'][tuple((x,y))]=[]

                    frame = cv2.rectangle(frame,(xLeftBottom,yLeftBottom),(xRightTop,yRightTop), color, thickness=2) ### added today
                    # draw the centroid on the frame
                    frame = cv2.circle(frame, (int(x),int(y)), 15, color, -1)
                    print("before if STATE i %d frame %d x y: %d %d" % (i, total_frames, x, y))
                    tracking_started = True
                    if i == 0:
                        color_dict = dict()
                        centroids[0,0,0] = x
                        centroids[0,0,1] = y
                        color_dict[tuple(color)]=[(x,y)]

                    else:
                        centroid = np.array([[[x,y]]],dtype=np.float32)
                        centroids = np.append(centroids,centroid,axis = 0)
                        color_dict[tuple(color)]=[(x,y)]


        else:   # track an object only if it has been detected
            if centroids.sum() != 0 and tracking_started:
                next1, st, error = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                                centroids, None, **lk_params)

                good_new = next1[st==1]
                good_old = centroids[st==1]


                # print("color dict", color_dict)

                corn_id_bbox = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    # Returns a contiguous flattened array as (x, y) coordinates for new point
                    a, b = new.ravel()
                    c, d = old.ravel()
                    distance = np.sqrt((a-c)**2 + (b-d)**2)
                    # distance between new and old points should be less than
                    # 200 for 2 points to be same the object
                    if distance < 200 :
                        corn_dict['centroids'][corn_dict[i]['centroid'][0]].append((a,b))
                        for color, centroids_list in color_dict.items():
                            for centroids in centroids_list:
                                if centroids==(c,d):
                                    color_dict[color].append((a,b))
                                    color_old = color
                                    frame = cv2.circle(frame, (a, b), 15, color_old, -1)
                        
                        ## TODO: global corn ID?
                        res = tuple(map(operator.sub, (c,d),corn_dict[i]['centroid'][0]))
                        new_bbox_coor1 = tuple(map(operator.add, corn_dict[i]['bbox'][0], res))
                        new_bbox_coor2 = tuple(map(operator.add, corn_dict[i]['bbox'][1], res))
                        new_bbox_coor1 = tuple(map(int, new_bbox_coor1))
                        new_bbox_coor2 = tuple(map(int, new_bbox_coor2))

                        frame = cv2.rectangle(frame, new_bbox_coor1, new_bbox_coor2, color_old, thickness=2) ### added today                    
                        frame = cv2.putText(frame, str(total_frames), (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 10, cv2.LINE_AA)
                        corn_id_bbox.append([new_bbox_coor1, new_bbox_coor2])
                print("total fr", total_frames,"corn_id", corn_id_bbox)
                corn_id_bbox_dict[total_frames] = corn_id_bbox
                centroids = good_new.reshape(-1, 1, 2)

        total_frames += 1
        frame_number += 1
        fps.update()
        fps.stop()
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        if output_file:
            writer.write(frame)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", frame)
        prev_frame = frame
        if cv2.waitKey(1) >= 0:  # Break with ESC 
            break
    return corn_id_bbox_dict

def get_prediction(img, model_path, confidence=0.5):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(model_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if len([x for x in pred_score if x>confidence])!=0:
      pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
      pred_boxes = pred_boxes[:pred_t+1]
      pred_class = pred_class[:pred_t+1]
      pred_score = pred_score[:pred_t+1]
    else:
      pred_boxes, pred_class, pred_score = None, None, None

    return pred_boxes, pred_class, pred_score

