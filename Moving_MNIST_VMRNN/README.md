# VMRNN: Integrating Vision Mamba and LSTM for Efficient and Accurate Spatiotemporal Forecasting


## Introduction

Combining CNNs or ViTs, with RNNs for spatiotemporal forecasting, has yielded unparalleled results in predicting temporal and spatial dynamics. However, modeling extensive global information remains a formidable challenge; CNNs are limited by their narrow receptive fields, and ViTs struggle with the intensive computational demands of their attention mechanisms. The emergence of recent Mamba-based architectures has been met with enthusiasm for their exceptional long-sequence modeling capabilities, surpassing established vision models in efficiency, accuracy, and computational footprint, which motivates us to develop an innovative architecture tailored for spatiotemporal forecasting. In this paper, we propose the VMRNN cell, a new recurrent unit that integrates the strengths of Vision Mamba blocks with LSTM. We construct a network centered on VMRNN cells to tackle spatiotemporal prediction tasks effectively. Our extensive evaluations show that our proposed approach secures competitive results on a variety of pivot benchmarks while maintaining a smaller model size.


## Overview
- `Pretrained/` contains pretrained weights on MovingMNIST.
- `data/` contains the MNIST dataset and the MovingMNIST test set download link.
- `VMRNN_B.py` contains the model with a single VMRNN cell.
- `VMRNN_D.py` contains the model with a multiple VMRNN cell.
- `dataset.py` contains training and validation dataloaders.
- `functions.py` contains train and test functions.
- `train.py` is the core file for training pipeline.
- `test.py` is a file for a quick test.

## Requirements
- python >= 3.8
- torch == 1.11.0
- torchvision == 0.12.0
- numpy
- matplotlib
- skimage == 0.19.2
- timm == 0.4.12
- einops == 0.4.1

## Citation
If you find this work useful in your research, please cite the paper:
```


```

## Acknowledgment
These codes are based on [SwinLSTM](https://github.com/SongTang-x/SwinLSTM). We extend our sincere appreciation for their valuable contributions.

