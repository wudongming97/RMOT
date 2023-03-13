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
MOT dataset which returns image_id for evaluation.
"""
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import json
import datasets.transforms as T
from models.structures import Instances


class DetMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = args.vis
        self.video_dict = {}

        with open(data_txt_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [(x.replace('images', 'labels_with_ids').replace('training', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        # The number of images per sample: 1 + (num_frames - 1) * interval.
        # The number of valid samples: num_images - num_image_per_sample + 1.
        self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval

        self._register_videos()

        # video sampler.
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.lengths) > 0
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]
            self.current_epoch = 0

    def _register_videos(self):
        for label_name in self.label_files:
            video_name = '/'.join(label_name.split('/')[:-1])
            if video_name not in self.video_dict:
                print("register {}-th video: {} ".format(len(self.video_dict) + 1, video_name))
                self.video_dict[video_name] = len(self.video_dict)
                # assert len(self.video_dict) <= 300

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        gt_instances.is_ref = targets['is_ref']
        return gt_instances

    def _pre_single_frame(self, idx: int, expression_infos):

        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        ref_ids = []
        for expression_info in expression_infos:
            if expression_info is not None:
                frame_id = int(img_path.split('/')[-1].split('.')[0])
                if str(frame_id) in expression_info['label'].keys():
                    ref_ids += expression_info['label'][str(frame_id)]
        ref_ids = [int(ref_id) for ref_id in ref_ids]

        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        if osp.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            if 'KITTI' in label_path:
                # normalized x1y1wh to pixel xyxy format
                labels = labels0.copy()
                labels[:, 2] = w * (labels0[:, 2])
                labels[:, 3] = h * (labels0[:, 3])
                labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4])
                labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5])
            else:
                # normalized cewh to pixel xyxy format
                labels = labels0.copy()
                labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
                labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
                labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
                labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
        else:
            raise ValueError('invalid label path: {}'.format(label_path))
        video_name = '/'.join(label_path.split('/')[:-1])
        obj_idx_offset = self.video_dict[video_name] * 1000000  # 1000000 unique ids is enough for a video.

        if 'KITTI' in img_path:
            targets['dataset'] = 'KITTI'
        else:
            raise NotImplementedError()

        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        targets['is_ref'] = []

        # print(labels[0])
        for label in labels:
            targets['boxes'].append(label[2:6].tolist())
            targets['area'].append(label[4] * label[5])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)
            # targets['labels'].append(label[0] - 1)  # category start from 0
            obj_id = label[1] + obj_idx_offset if label[1] >= 0 else label[1]
            targets['obj_ids'].append(obj_id)  # relative id
            if int(label[1]) in ref_ids:  # if an id is in ref_ids, then mask it as 1, else mask 0
                targets['is_ref'].append(1.0)
            else:
                targets['is_ref'].append(0.0)

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'], dtype=torch.int64)
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['is_ref'] = torch.as_tensor(targets['is_ref'])
        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, start, end, interval=1, expression_info=None):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i, expression_info)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def __getitem__(self, idx):
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)

        img_path = self.img_files[sample_start]

        if 'KITTI' in img_path:
            video_id = img_path.split('/')[-2]
            expression_list = os.listdir(osp.join(self.args.rmot_path, 'expression', video_id))
            expression_random = random.choice(expression_list)
            expression_path = osp.join(self.args.rmot_path, 'expression', video_id, expression_random)
            with open(expression_path, 'r') as f:
                expression_info = json.load(f)
            sentence = [expression_info['sentence']]
            expression_info = [expression_info]
        else:
            raise NotImplementedError()

        images, targets = self.pre_continuous_frames(sample_start, sample_end, sample_interval, expression_info)
        data = {}
        dataset_name = targets[0]['dataset']
        transform = self.dataset2transform[dataset_name]
        if transform is not None:
            images, targets = transform(images, targets)
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
        data.update({
            'imgs': images,
            'sentences': sentence,
            'gt_instances': gt_instances,
            'dataset_name': dataset_name,
        })
        if self.args.vis:
            data['ori_img'] = [target_i['ori_img'] for target_i in targets]
        return data

    def __len__(self):
        return self.item_num



class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, dataset2transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)


def make_transforms_for_kitti(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset2transform(args, image_set):

    kitti_train = make_transforms_for_kitti('train', args)
    dataset2transform_train = {'KITTI': kitti_train}
    if image_set == 'train':
        return dataset2transform_train
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.rmot_path)
    assert root.exists(), f'provided RMOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    return dataset

