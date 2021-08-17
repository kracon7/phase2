import os
import sys
import argparse
import pickle
import cv2
import time
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from model import CorridorNet


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


class CorridorDataset(Dataset):
    def __init__(self, csv_file, stat_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the voxels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file, header=None)
        stat = pickle.load(open(stat_file, 'rb'))
        self.mean = stat['mean']
        self.std = stat['std']
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        torch_data = self.transform(Image.open(im_path))

        # label
        label = np.array([float(self.data_frame.iloc[idx, 1]), float(self.data_frame.iloc[idx, 2]),
                     float(self.data_frame.iloc[idx, 3]), float(self.data_frame.iloc[idx, 4])])
        label = (label - self.mean) / self.std
        torch_label = torch.from_numpy(label).float()

        sample = {'data': torch_data, 'label': torch_label, 'fnames': self.data_frame.iloc[idx, 0]}
        return sample


plt.ion()
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corn_front_data')

if __name__ == '__main__':

    # Add Parameters from argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--result_path', type=str, default='resnet_result')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epoch_checkpoint', required=True, type=int, default=0)
    args = parser.parse_args()

    save_path = os.path.join(ROOT_DIR,  args.result_path, 'img')
    # Create a directory if not exist.
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_csv_path = os.path.join(ROOT_DIR, 'train.csv')
    test_csv_path = os.path.join(ROOT_DIR, 'test.csv')
    stat_path = os.path.join(ROOT_DIR, 'corridor_stat.pkl')
    stat = pickle.load(open(stat_path, 'rb'))
    
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    T = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                           ])

    train_dataset = CorridorDataset(csv_file=train_csv_path, 
                                    root_dir=ROOT_DIR,
                                    stat_file=stat_path, 
                                    transform=T)
    train_data_loader = DataLoader(train_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=args.num_workers)

    test_dataset = CorridorDataset(csv_file=test_csv_path, 
                                   root_dir=ROOT_DIR,
                                   stat_file=stat_path,  
                                   transform=T)
    test_data_loader = DataLoader(test_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers)

    net = CorridorNet().to(args.device)
    l1_loss = nn.L1Loss().to(args.device)

    # Read models from checkpoint
    fname_checkpoint = os.path.join(ROOT_DIR, args.result_path, 'chckpt_%i.pt' % args.epoch_checkpoint)
    print("Loading checkpoint from path: " + fname_checkpoint)
    checkpoint = torch.load(fname_checkpoint, map_location=args.device)
    net.load_state_dict(checkpoint['state_dict'])

    loss_train = checkpoint['loss_train']
    loss_test = checkpoint['loss_test']
    epoch_checkpoint = checkpoint['epoch']
    

    num_batch_train = train_dataset.__len__() / args.batch_size
    num_batch_test = test_dataset.__len__() / args.batch_size

    
    net.eval()
    # test cycle
    epoch_loss = []

    t = time.time()
    num_img = 0
    for i, sample in enumerate(test_data_loader):
        data = sample['data'].to(args.device)
        label = sample['label'].to(args.device)

        pred = net(data)

        b = data.shape[0]
        num_img += b
        for k in range(b):
            gt_lines = label[k].detach().cpu().numpy().reshape(-1) * stat['std'] + stat['mean']
            pred_lines = pred[k].detach().cpu().numpy().reshape(-1) * stat['std'] + stat['mean']
            img = cv2.imread(os.path.join(ROOT_DIR, sample['fnames'][k]))

            img = draw_corridor(img, gt_lines, (0, 0, 255))
            img = draw_corridor(img, pred_lines, (0, 255, 0))
            cv2.imwrite(os.path.join(save_path, 'batch_%d_%d.png'%(i, k)), img)

    print('Total number of image: %d, total evaluation time: %.4f, average: %4f'%(
            num_img, time.time()-t, (time.time()-t)/num_img))