# ConResNet
<p align="left">
    <img src="models/conresnet.png" width="85%" height="85%">
</p>


This repo holds the pytorch implementation of ConResNet:<br />

**Paper: Inter-slice Context Residual Learning for 3D Medical Image Segmentation.** 
(https://ieeexplore.ieee.org/abstract/document/9245569) 
(https://arxiv.org/pdf/2011.14155.pdf)

## Requirements
Python 3.6<br />
Torch==1.4.0<br />
Apex==0.1<br />

## Usage

### 0. Installation
* Clone this repo
```
git clone https://github.com/jianpengz/ConResNet.git
cd ConResNet
```
### 1. Data Preparation
* Put the data and image_id_list in `dataset/`.

### 2. Training
* Run `run.sh` to start the training. 

### 3. Evaluation
* Run `python test.py` to start the evaluation.

### 7. Citation
If this code is helpful for your study, please cite:

```
@article{zhang2020conresnet,
  title={Inter-slice Context Residual Learning for 3D Medical Image Segmentation},
  author={Zhang, Jianpeng and Xie, Yutong and Wang, Yan and Xia, Yong},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={2},
  pages={661-672},
  year={2021},
  publisher={IEEE}
}
```

### Contact
Jianpeng Zhang (james.zhang@mail.nwpu.edu.cn)
