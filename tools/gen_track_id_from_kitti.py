import os.path as osp
import os
import numpy as np
from pycocotools import mask
import torch


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = './data/MOTS_KITTI/instances_txt'
label_root = './data/MOTS_KITTI/labels_with_ids'
mkdirs(label_root)

current = -1


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    y = [0, 0, 0, 0]
    y[0] = x[0]
    y[1] = x[1]
    y[2] = (x[0] + x[2])
    y[3] = (x[1] + x[3])
    return y

mapper = {}
for idx in range(21):
    ids = {}

    seq_width = 1242
    seq_height = 375


    gts = open("./data/MOTS_KITTI/instances_txt/{:04d}.txt".format(idx)).readlines()

    seq_label_root = osp.join(label_root, "KITTI-{}".format(idx), 'img1')
    mkdirs(seq_label_root)

    for l in gts:
        # fid, tid, x, y, w, h, mark, label, _ =
        fid, tid, class_id, img_height, img_width, rle = l.strip().split(" ")
        class_id = int(class_id)
        if class_id == 10:
            continue

        img_height = int(img_height)
        img_width = int(img_width)
        mask_rle = {'size': [img_height, img_width], 'counts': rle.encode(encoding='UTF-8')}
        x, y, w, h = mask.toBbox(mask_rle)

        fid = int(fid)
        tid = int(tid)
        if not tid in ids:
            current += 1
            ids[tid] = current

        tid_curr = ids[tid]

        mapper["KITTI-{}_{}".format(idx, tid)] = tid_curr

        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)

torch.save(mapper, "mapper.pth")
