from utils import *
import os
import sys
import time
import datetime
import random
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms
from torch.autograd import Variable
from imutils.video import FPS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import cv2

from sort import *
import argparse


cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# fps = FPS().start()

class MOT_Tracker():
    def __init__(self, args, rcnn_model):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = rcnn_model
        self.CLASS_NAMES = ["__background__", "corn_stem"]
        self.torch_trans = Transforms.Compose([Transforms.ToTensor()])
        
        self.mot_tracker = Sort()   
        self.frame_number = 1
        self.corn_id_bbox_dict = dict()

    def get_prediction(self, img, confidence=0.5):
        """
        get_prediction
          parameters:
            - img - input image
            - confidence - threshold value for prediction score
          method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - class, box coordinates are obtained, but only prediction score > threshold
              are chosen.
        
        """
        img = self.torch_trans(img).to(self.device)
        pred = self.model([img])
        pred_class = [self.CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                      for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        if len([x for x in pred_score if x > confidence]) != 0:
          pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]
          pred_boxes = pred_boxes[:pred_t+1]
          pred_class = pred_class[:pred_t+1]
          pred_score = pred_score[:pred_t+1]
        else:
          pred_boxes, pred_class, pred_score = None, None, None

        return pred_boxes, pred_class, pred_score


    def corn_tracking_sort(self, img):
        
        pred_boxes, pred_class, pred_score = self.get_prediction(img, 0.8)
        
        detections_list = []
        for i in range(len(pred_score)):
            detections_list.append([pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i]
                                   [1][0], pred_boxes[i][1][1], pred_score[i], pred_score[i], 1])

        detections = torch.FloatTensor(detections_list)

        
        if detections is not None:
            tracked_objects = self.mot_tracker.update(detections.cpu())

            # n_cls_preds = len(unique_labels)
            n_cls_preds = 2

            corn_id_bbox = dict()
            drawn_img = img.copy()
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = self.CLASS_NAMES[int(cls_pred)]
                drawn_img = cv2.rectangle(drawn_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                # cv2.rectangle(frame, (int(x1), int(y1)-35), (int(x1)+len(cls)*19+60, int(y1)), color, -1)
                # cv2.putText(frame, cls + "-" + str(int(obj_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                drawn_img = cv2.rectangle(drawn_img, (int(x1), int(y2)-35),
                              (int(x1)+len(cls)*19+60, int(y2)), color, -1)
                drawn_img = cv2.putText(drawn_img, cls + "-" + str(int(obj_id)), (int(x1),
                            int(y2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                corn_id_bbox[int(obj_id)] = [tuple((int(x1), int(y1))), tuple((int(x2), int(y2)))]
        self.corn_id_bbox_dict[self.frame_number] = corn_id_bbox
        self.frame_number +=1

        # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", drawn_img)
        time.sleep(0.1)
        # cv2.waitKey(0)
        # if cv2.waitKey(1) >= 0:  # Break with ESC
        #     break

        return corn_id_bbox