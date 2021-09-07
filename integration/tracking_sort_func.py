from models import *
from utils import *
import os
import sys
import time
import datetime
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.transforms as T
from imutils.video import FPS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import cv2

from sort import *
import argparse


cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# initialize Sort object and video capture
# videopath = '../corn_detection_torch/8-6_10-21.avi'
# videopath = args.video
# vid = cv2.VideoCapture(videopath)
mot_tracker = Sort()
fps = FPS().start()


# if args.output:
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#     writer = cv2.VideoWriter(args.output, fourcc, 100,
#                              (prev_frame.shape[1], prev_frame.shape[0]), True)

Tensor = torch.cuda.FloatTensor
CLASS_NAMES = ["__background__", "corn_stem"]


def get_prediction(img, model_path, confidence=0.5):
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
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(model_path)

    transform = T.Compose([T.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])
    pred_class = [CLASS_NAMES[i]
                  for i in list(pred[0]['labels'].cpu().numpy())]
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


def corn_tracking_sort(frame_dir, model_path="./weights/faster-rcnn-corn_bgr8_ep100.pt", output_file=None):

  frame_number = 1
  frame = cv2.imread(os.path.join(frame_dir, 'frame_' + format(frame_number, '07d')+'.png'))

  if output_file:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_file, fourcc, 100,(frame.shape[1], frame.shape[0]), True)

  corn_id_bbox_dict = dict()
  while True:
    frame = cv2.imread(os.path.join(frame_dir, 'frame_' +
                       format(frame_number, '07d')+'.png'))

    pred_boxes, pred_class, pred_score = get_prediction(frame, model_path, 0.5)
    detections_list = []
    for i in range(len(pred_score)):
        detections_list.append([pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i]
                               [1][0], pred_boxes[i][1][1], pred_score[i], pred_score[i], 1])

    detections = torch.FloatTensor(detections_list)

    
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        # n_cls_preds = len(unique_labels)
        n_cls_preds = 2

        corn_id_bbox = dict()
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cls = CLASS_NAMES[int(cls_pred)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            # cv2.rectangle(frame, (int(x1), int(y1)-35), (int(x1)+len(cls)*19+60, int(y1)), color, -1)
            # cv2.putText(frame, cls + "-" + str(int(obj_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            cv2.rectangle(frame, (int(x1), int(y2)-35),
                          (int(x1)+len(cls)*19+60, int(y2)), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (int(x1),
                        int(y2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            corn_id_bbox[int(obj_id)] = [tuple((int(x1), int(y1))), tuple((int(x2), int(y2)))]
    corn_id_bbox_dict[frame_number] = corn_id_bbox
    frame_number +=1

    fps.update()
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if output_file:
        writer.write(frame)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    # prev_frame = frame
    if cv2.waitKey(1) >= 0:  # Break with ESC
        break
  return corn_id_bbox_dict