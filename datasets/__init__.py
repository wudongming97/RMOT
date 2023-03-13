# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision

from .coco import build as build_coco
from .detmot import build as build_e2e_mot
from .dance import build as build_e2e_dance
from .static_detmot import build as build_e2e_static_mot
from .torchvision_datasets import CocoDetection
from .refer_kitti import build as build_rmot

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'e2e_static_mot':
        return build_e2e_static_mot(image_set, args)
    if args.dataset_file == 'e2e_mot':
        return build_e2e_mot(image_set, args)
    if args.dataset_file == 'e2e_dance':
        return build_e2e_dance(image_set, args)
    if args.dataset_file == 'e2e_rmot':
        return build_rmot(image_set, args)
        # return build_rmot_plus(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
