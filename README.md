# Referring Multi-Object Tracking

This repository is an official implementation of the paper [Referring Multi-Object Tracking](https://arxiv.org/abs/2303.03366).

## Introduction


<div style="align: center">
<img src=./figs/TransRMOT.png/>
</div>

**Abstract.** 
Existing referring understanding tasks tend to involve the detection of a single text-referred object. In this paper, we propose a new and general referring understanding task, termed referring multi-object tracking (RMOT). Its core idea is to employ a  language expression as a semantic cue to guide the prediction of multi-object tracking. To the best of our knowledge, it is the first work to achieve an arbitrary number of referent object predictions in videos. To push forward RMOT, we construct one benchmark with scalable expressions based on KITTI, named Refer-KITTI. Specifically, it provides 18 videos with 818 expressions, and each expression in a video is annotated with an average of 10.7 objects. Further, we develop a transformer-based architecture TransRMOT to tackle the new task in an online manner, which achieves impressive detection performance.

## Updates
- (2023/03/17) RMOT dataset and code are released.
- (2023/03/07) RMOT paper is available on [arxiv](https://arxiv.org/abs/2303.03366).
- (2023/02/28) RMOT is accepted by CVPR2023! The dataset and code is coming soon!



## Getting started
### Installation

The basic environment setup is on top of [MOTR](https://github.com/megvii-research/MOTR), including conda environment, pytorch version and other requirements. 

### Dataset
You can download our created expression and labels_with_ids from [Google Drive](https://drive.google.com/drive/folders/1CjX1Y5XJ2zRloTEQM1OHQsF1RCuBLXbc?usp=sharing) and the official KITTI images from [here](https://www.cvlibs.net/datasets/kitti/eval_tracking.php).
The Refer-KITTI is organized as follows:

```
.
├── refer-kitti
│   ├── KITTI
│           ├── training
│           ├── labels_with_ids
│   └── Expression
```


### Training
You can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Then training MOTR on 8 GPUs as following:
```bash 
sh configs/r50_rmot_train.sh
```

### Evaluation
You can download the pretrained model of TransRMOT (the link is in "Main Results" session), then run following command to generate and save prediction boxes:
```bash
sh configs/r50_rmot_test.sh
```

You can get the main results by runing the evaluation part. You can also use our prediction and gt file from [Google Drive](https://drive.google.com/drive/folders/1CjX1Y5XJ2zRloTEQM1OHQsF1RCuBLXbc?usp=sharing). 
```bash
cd TrackEval/script
sh evaluate_rmot.sh
```

## Results


| **Method** | **Dataset** | **HOTA** | **DetA** | **AssA** | **DetRe** | **DetPr** | **AssRe** | **AssRe** | **LocA** |                                           **URL**                                           |
|:----------:|:-----------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|-----------|----------| :-----------------------------------------------------------------------------------------: |
| TransRMOT  | Refer-KITTI |  38.06   |  29.28   |  50.83   |   40.19   |   47.36   |   55.43   | 81.36     | 79.93    | [model](https://drive.google.com/drive/folders/1CjX1Y5XJ2zRloTEQM1OHQsF1RCuBLXbc?usp=sharing) |





## Citing RMOT
If you find RMOT useful in your research, please consider citing:

```bibtex
@inproceedings{wu2023referring,
  title={Referring Multi-Object Tracking},
  author={Dongming Wu, Wencheng Han, Tiancai Wang, Xingping Dong, Xiangyu Zhang and Jianbing Shen},
  booktitle={CVPR},
  year={2023}
}
```


## Acknowledgements

- [MOTR](https://github.com/megvii-research/MOTR)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)


