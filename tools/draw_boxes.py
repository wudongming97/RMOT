# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import load_model
from main import get_args_parser
from torch.nn.functional import interpolate
from typing import List
from util.evaluation import Evaluator
import motmetrics as mm
import shutil
import json
import matplotlib.pyplot as plt

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader

np.random.seed(2020)

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img.numpy(), c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img.numpy(), c1, c2, color, -1)  # filled
        cv2.putText(img.numpy(),
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img.numpy(), score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)
    return img


'''
deep sort 中的画图方法，在原图上进行作画
'''


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=None)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


def load_label(label_path: str, img_size: tuple) -> dict:
    labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
    h, w = img_size
    # Normalized cewh to pixel xyxy format
    labels = labels0.copy()
    labels[:, 2] = w * (labels0[:, 2])
    labels[:, 3] = h * (labels0[:, 3])
    labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4])
    labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5])
    targets = {'boxes': [], 'labels': [], 'area': []}
    num_boxes = len(labels)

    frame_id = int(label_path.split('/')[-1].split('.')[0])
    if frame_id < 320:
        ref_ids = [29]
    else:
        ref_ids = [29, 31]

    visited_ids = set()
    for label in labels[:num_boxes]:
        obj_id = label[1]
        if obj_id not in ref_ids:
            continue
        if obj_id in visited_ids:
            continue
        visited_ids.add(obj_id)
        targets['boxes'].append(label[2:6].tolist())
        targets['area'].append(label[4] * label[5])
        targets['labels'].append(obj_id)
    targets['boxes'] = np.asarray(targets['boxes'])
    targets['area'] = np.asarray(targets['area'])
    targets['labels'] = np.asarray(targets['labels'])
    return targets

class ListImgDataset(Dataset):
    def __init__(self, img_list) -> None:
        super().__init__()
        self.img_list = img_list

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        label_path = f_path.replace('training', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        # print(label_path)
        cur_img = cv2.imread(f_path)
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        targets = load_label(label_path, cur_img.shape[:2]) if os.path.exists(label_path) else None
        # img = draw_bboxes(torch.tensor(cur_img), targets['boxes'])
        return cur_img, targets

    def init_img(self, img, targets):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, targets

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, targets = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, targets)


class Detector(object):
    def __init__(self, args, model=None, seq_num=2):

        self.args = args
        self.detr = model

        self.seq_num = seq_num
        img_list = os.listdir(os.path.join(self.args.mot_path, 'KITTI/training/image_02', self.seq_num[0]))
        img_list = [os.path.join(self.args.mot_path, 'KITTI/training/image_02', self.seq_num[0], _) for _ in img_list
                    if ('jpg' in _) or ('png' in _)]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.json_path = os.path.join('/data/Dataset/KITTI/expression_clean/', seq_num[0], seq_num[1])
        with open(self.json_path, 'r') as f:
            json_info = json.load(f)
        self.json_info = json_info
        self.sentence = [json_info['sentence']]

        self.save_path = os.path.join(self.args.output_dir,
                                      'results_epoch_all/{}/{}'.format(seq_num[0], seq_num[1].split('.')[0]))
        os.makedirs(self.save_path, exist_ok=True)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)
        if os.path.exists(os.path.join(self.predict_path, f'{self.seq_num}.txt')):
            os.remove(os.path.join(self.predict_path, f'{self.seq_num}.txt'))

    @staticmethod
    def visualize_img_with_bbox(img_path, img, gt_boxes=None, identities=None):
        if gt_boxes is not None:
            img_show = draw_bboxes(img, gt_boxes, identities=identities)
        cv2.imwrite(img_path, cv2.cvtColor(img_show.numpy(), cv2.COLOR_RGB2BGR))

    def detect(self, prob_threshold=0.7, area_threshold=100):
        print('Results are saved into {}'.format(self.save_path))

        # meta_info = open(os.path.join(data_root, self.seq_num[0], 'seqinfo.ini')).read()
        # im_width = int(meta_info[meta_info.find('imWidth=') + 8:meta_info.find('\nimHeight')])
        # im_height = int(meta_info[meta_info.find('imHeight=') + 9:meta_info.find('\nimExt')])

        # track_instances = None
        loader = DataLoader(ListImgDataset(self.img_list), 1, num_workers=2)
        for i, (cur_img, ori_img, targets) in enumerate(tqdm(loader)):
            cur_img, ori_img = cur_img[0], ori_img[0]

            # for visual
            cur_vis_img_path = os.path.join(self.save_path, 'frame_{}.jpg'.format(i))
            gt_boxes = targets['boxes'][0].numpy()
            gt_ids = targets['labels'][0].numpy()
            self.visualize_img_with_bbox(cur_vis_img_path, ori_img, gt_boxes=gt_boxes, identities=gt_ids)



if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    # detr, _, _ = build_model(args)
    # checkpoint = torch.load(args.resume, map_location='cpu')
    # detr = load_model(detr, args.resume)
    # detr.eval()
    # detr = detr.cuda()

    # # '''for MOT17 submit'''
    # sub_dir = 'MOT17/images/test'
    # seq_nums = ['MOT17-01-SDP',
    #             'MOT17-03-SDP',
    #             'MOT17-06-SDP',
    #             'MOT17-07-SDP',
    #             'MOT17-08-SDP',
    #             'MOT17-12-SDP',
    #             'MOT17-14-SDP']
    data_root = '/data/Dataset/KITTI/training/image_02/'
    expressions_root = '/data/Dataset/KITTI/expression_clean/'
    # video_ids = sorted(os.listdir(expressions_root))
    # video_ids = ['0005', '0011', '0013']
    video_ids = ['0015']
    # video_ids = ['MOT17-11-SDP']

    seq_nums = []
    for video_id in video_ids:  # we have multiple videos
        expression_jsons = sorted(os.listdir(os.path.join(expressions_root, video_id)))
        for expression_json in expression_jsons:  # each video has multiple expression json files
            # with open(osp.join(expressions_root, video_id, expression_json), 'r') as f:
            #     a = json.load(f)
            seq_nums.append([video_id, expression_json])

    for seq_num in seq_nums:
        if 'cars-in-the-left' not in seq_num[1]:
            continue
        # if seq_num[1] == 'males.json':
        print('Evaluating seq {}'.format(seq_num))
        det = Detector(args, model=None, seq_num=seq_num)
        det.detect()

