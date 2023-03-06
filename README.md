# Referring Multi-Object Tracking

This repository is an official implementation of the paper [Referring Multi-Object Tracking](https://arxiv.org/pdf/2105.03247.pdf).

## Introduction

**TL; DR.** 
RMOT is a new referring understanding task. 

<div style="align: center">
<img src=./figs/TransRMOT.png/>
</div>

**Abstract.** 
Existing referring understanding tasks tend to involve the detection of a single text-referred object. In this paper, we propose a new and general referring understanding task, termed referring multi-object tracking (RMOT). Its core idea is to employ a  language expression as a semantic cue to guide the prediction of multi-object tracking. To the best of our knowledge, it is the first work to achieve an arbitrary number of referent object predictions in videos. To push forward RMOT, we construct one benchmark with scalable expressions based on KITTI, named Refer-KITTI. Specifically, it provides 18 videos with 818 expressions, and each expression in a video is annotated with an average of 10.7 objects. Further, we develop a transformer-based architecture TransRMOT to tackle the new task in an online manner, which achieves impressive detection performance.
## Updates
- (2023/03/06) Our paper is accepted by CVPR2023! The dataset and code is coming soon!



## Citing RMOT
If you find MOTR useful in your research, please consider citing:
```bibtex
@article{zeng2021motr,
  title={MOTR: End-to-End Multiple-Object Tracking with TRansformer},
  author={Zeng, Fangao and Dong, Bin and Zhang, Yuang and Wang, Tiancai and Zhang, Xiangyu and Wei, Yichen},
  journal={arXiv preprint arXiv:2105.03247},
  year={2021}
}
```
