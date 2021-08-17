import os 
import sys
import cv2
import csv
import pickle
import argparse
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()


def get_lines(annotations):
    '''
    extract the image file names and their corridor lines position
    Input:
        annotations - pandas data frame
    Output:
        lines_loc - dict object, contains the two lines coordinates [x1, x2, y1, y2]
    '''
    
    lines_loc = collections.defaultdict(list)

    N = len(annotations)
    for i in range(N):
        f = annotations.iloc[i, 0]
        assert annotations.iloc[i, 3] == 2

        lines = annotations.iloc[i, 5]
        xpoints = lines.split('[')[1].split(']')[0].split(',')
        ypoints = lines.split('[')[2].split(']')[0].split(',')

        xpoints = [int(item) for item in xpoints]
        ypoints = [int(item) for item in ypoints]

        lines_loc[f].append(xpoints + ypoints)

    return lines_loc

def format_labels(lines_loc):
    '''
    reformat label to vanishing point location and two slopes
    Input:
        lines_loc - dict object, contains the two lines coordinates [x1, x2, y1, y2]
    Output:
        labels - 
    '''
    labels = collections.defaultdict(list)
    for fname in lines_loc.keys():
        x1, x2, y1, y2 = lines_loc[fname][0]
        x3, x4, y3, y4 = lines_loc[fname][1]
        # compute intersection point
        x = ((x1*y2-x2*y1)*(x3-x4) - (x3*y4-x4*y3)*(x1-x2)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        y = ((x1*y2-x2*y1)*(y3-y4) - (x3*y4-x4*y3)*(y1-y2)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
        # compute two slopes
        theta1 = np.rad2deg(np.arctan2(y2-y1, x2-x1))
        theta2 = np.rad2deg(np.arctan2(y4-y3, x4-x3))

        if theta1 < 0:
            theta1 += 180
        if theta2 < 0:
            theta2 += 180

        if theta1 > theta2:
            labels[fname] = [x, y, theta1, theta2]
        else:
            labels[fname] = [x, y, theta2, theta1]

    return labels

def draw_corridor(img, label):
    '''
    draw vanishing point and two lines
    Input:
        img - RGB image
        label - [x1, y1, theta1, theta2]
    Output:
        image - 
    '''
    vpzx, vpzy = int(label[0]), int(label[1])
    img = cv2.circle(img, (vpzx, vpzy), 4, (0, 255, 255), 2)

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

    img = cv2.line(img, (vpzx, vpzy), (pt1[0], pt1[1]), (255,0,0), 2)
    img = cv2.line(img, (vpzx, vpzy), (pt2[0], pt2[1]), (255,0,0), 2)

    return img


def main(args):
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corn_front_data')
    output_dir = os.path.join(ROOT, 'labeled_img')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    csv_path = os.path.join(ROOT, 'dataset.csv')
    result_csv = csv.writer(open(csv_path, 'w'), delimiter=',')
        
    annotations = pd.read_csv(args.annotation_csv)
    lines_loc = get_lines(annotations)
    labels = format_labels(lines_loc)

    # tran test split
    if args.split:
        train_csv_path = os.path.join(ROOT, 'train.csv')
        test_csv_path = os.path.join(ROOT, 'test.csv')
        train_csv = csv.writer(open(train_csv_path, 'w'), delimiter=',')
        test_csv = csv.writer(open(test_csv_path, 'w'), delimiter=',')
        ratio = 0.8
        N = len(labels.keys())

    print("Saving corridor labels")
    pickle.dump(labels, open(os.path.join(ROOT, 'corridor_labels.pkl'), 'wb'))

    count = 0
    for f in labels.keys():
        print('Working on %dth image: %s'%(count, f))
        img = cv2.imread(os.path.join(ROOT, args.img_dir, f))
        img = draw_corridor(img, labels[f])
        cv2.imwrite(os.path.join(output_dir, f), img)

        data = [os.path.join('color', f)] + labels[f]
        result_csv.writerow(data)

        if args.split:
            if count < N * ratio:
                train_csv.writerow(data)
            else:
                test_csv.writerow(data)

        count += 1

    print("Saving corridor stats...")
    # standardization for the label
    labels_list = np.array([item for item in labels.values()])
    mean, std = np.mean(labels_list, axis=0), np.std(labels_list, axis=0)
    stat = {'mean': mean, 'std': std}
    print('Mean: %.2f %.2f %.2f %.2f, \nDeviation: %.2f %.2f %.2f %.2f'%(
            mean[0], mean[1], mean[2], mean[3], std[0], std[1], std[2], std[3]))
    pickle.dump(stat, open(os.path.join(ROOT, 'corridor_stat.pkl'), 'wb'))

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract corn image annotation labels')
    parser.add_argument('--annotation_csv', required=True, help='annotation_csv to load bounding box info')
    parser.add_argument('--img_dir', default='color', help='directory to load images')
    parser.add_argument('--split', default=1, type=int, help='split into train and test set')
    args = parser.parse_args()

    main(args)