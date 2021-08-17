import os
import sys
import argparse
import pickle
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

        sample = {'data': torch_data, 'label': torch_label}
        return sample


plt.ion()
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corn_front_data')

if __name__ == '__main__':

    # Add Parameters from argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='resnet_result')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--has_checkpoint', action='store_true', default=False)
    parser.add_argument('--epoch_checkpoint', type=int, default=0)
    args = parser.parse_args()

    save_path = os.path.join(ROOT_DIR,  args.save_path)
    # Create a directory if not exist.
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_csv_path = os.path.join(ROOT_DIR, 'train.csv')
    test_csv_path = os.path.join(ROOT_DIR, 'test.csv')
    stat_path = os.path.join(ROOT_DIR, 'corridor_stat.pkl')
    
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
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    l1_loss = nn.L1Loss().to(args.device)

    # Read models from checkpoint
    fname_checkpoint = os.path.join(save_path, 'chckpt_%i.pt' % args.epoch_checkpoint)
    if args.epoch_checkpoint > 1:
        print("Loading checkpoint from path: " + fname_checkpoint)
        checkpoint = torch.load(fname_checkpoint, map_location=args.device)
        net.load_state_dict(checkpoint['state_dict'])

        loss_train = checkpoint['loss_train']
        loss_test = checkpoint['loss_test']
        epoch_checkpoint = checkpoint['epoch']
    else:
        loss_train = []
        loss_test = []
        epoch_checkpoint = 1

    viz_data = False
    if viz_data:
        fig, ax = plt.subplots(1, args.batch_size)

    loss_fig, loss_ax = plt.subplots(1,1)

    num_batch_train = train_dataset.__len__() / args.batch_size
    num_batch_test = test_dataset.__len__() / args.batch_size

    for epoch in range(epoch_checkpoint, epoch_checkpoint + args.num_epochs):
        train_stat, test_stat = np.zeros(2).astype('float'), np.zeros(2).astype('float')

        net.train()
        # train cycle
        epoch_loss = []
        for i, sample in enumerate(train_data_loader):
            data = sample['data'].to(args.device)
            label = sample['label'].to(args.device)

            # visualize dataset
            if viz_data:
                for i in range(args.batch_size):
                    ax[i].imshow(data[i].detach().cpu().numpy().transpose(1,2,0))
                    ax[i].xaxis.set_visible(False)
                    ax[i].yaxis.set_visible(False)
                    ax[i].set_title(sample['name'][i])
                plt.pause(0.01)

            optim.zero_grad()
            pred = net(data)
            loss = l1_loss(pred, label)
            loss.backward()

            for param in net.parameters():
                param.grad.data.clamp_(-.1, .1)
            optim.step()

            # update the accuracy
            train_stat[0] += loss.detach().item() * pred.shape[0]
            train_stat[1] += pred.shape[0]
            # record the loss
            epoch_loss.append(loss.item())
            print('[%6d: %6d/%6d] train loss: %f, average training loss: %f' % \
                    (epoch, i, num_batch_train, loss.item(), train_stat[0]/train_stat[1]))


        loss_train.append(np.average(epoch_loss))

        net.eval()
        # test cycle
        epoch_loss = []
        for i, sample in enumerate(test_data_loader):
            data = sample['data'].to(args.device)
            label = sample['label'].to(args.device)

            pred = net(data)
            loss = l1_loss(pred, label)

            # update the accuracy
            test_stat[0] += loss.detach().item() * pred.shape[0]
            test_stat[1] += pred.shape[0]
            # record the loss
            epoch_loss.append(loss.item())
            print('[%6d: %6d/%6d] test loss: %f, average training loss: %f' % \
                    (epoch, i, num_batch_train, loss.item(), test_stat[0]/test_stat[1]))

        loss_test.append(np.average(epoch_loss))

        # plot result
        loss_ax.plot(loss_train, 'r')
        loss_ax.plot(loss_test, 'b')
        plt.pause(0.01)

        if (epoch) % args.save_every == 0:
            torch.save({    
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'loss_train': loss_train,
                    'loss_test': loss_test
                }, '%s/chckpt_%i.pt' % (save_path, epoch))
            loss_fig.savefig('%s/loss_%d.png'% (save_path, epoch))
