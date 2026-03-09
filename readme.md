# Multi-scene Separation and Reconstruction from Fused Random Compressed Measurements

This repository provides implementable code for multi-scene separation and reconstruction from fused random compressed measurements.

## How to Run

The main function can be accessed from:  
`Dual-view-CS/CS_DualView/main/main.m`

## Requirements

This code uses a Matlab-Python bridge and tests with the following Python environment:

- **Python**: 3.7.16
- **PyTorch**: 1.12.0+cu116
- **torchvision**: 0.13.0+cu116 (installed with PyTorch)
- **numpy**: 1.21.6
- **opencv-python**: 4.8.1.78
- **einops**: 0.6.1
- **PyYAML**: 6.0
- **basicsr**: >=1.4.2 (as used in the original Restormer implementation)

## Pretrained models

This repository does **not** ship large pretrained weights.  

Please download the following models manually and place them at the
indicated paths under `Trained_Weights/Restormer/pretrained_models`:

- Download the Gaussian denoising Restormer model from HuggingFace  
  https://huggingface.co/deepinv/Restormer/tree/main

Please download the following models manually and place them at the
indicated paths under `Trained_Weights/MWCNN`:
  **Please download them from the authors’ shared storage and update the links below:**
  [https://drive.google.com/drive/folders/1T5yvuDCToA_NU11GLnZKoVaXq2at7fSL?usp=drive_link](https://drive.google.com/drive/folders/1p6MAShg5g5J3Ip-2NZu_zW2wCThDdEkP?usp=drive_link)
These MWCNN models are the ones used in our paper

> Zan Chen, Tao Wang, Jun Li, Wenlong Guo, Yuanjing Feng,  
> Xueming Qian, and Xingsong Hou.  
> **Discard Significant Bits of Compressed Sensing: A Robust Image Coding for Resource-Limited Contexts.**  
> ACM Trans. Multimedia Comput. Commun. Appl. 21, 1, Article 31 (January 2025), 25 pages. 


