<div align="center" markdown>
<img src="https://github.com/user-attachments/assets/1416d5b0-0d5e-456a-9a5b-c6ddf164b6d0"/>  

# Fine-tune Segment Anything 2.1

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-segment-anything-2/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-segment-anything-2)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/serve-segment-anything-2/train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/serve-segment-anything-2/train.png)](https://supervise.ly)

</div>

# Overview

Segment Anything Model 2 is a foundation model for interactive instance segmentation in images and videos. It is based on transformer architecture with streaming memory for real-time video processing. SAM 2 is a generalization of the first version of SAM to the video domain, it processes video frame-by-frame and uses a memory attention module to attend to the previous memories of the target object. When SAM 2 is applied to images, the memory is empty and the model behaves like usual SAM.

Unlike the first version of Segment Anything, the frame embedding used by the SAM 2 decoder is conditioned on memories of past predictions and prompted frames (instead of being taken directly from an image decoder). Memory encoder creates "memories" of frames based on the current prediction, these "memories" are stored in model's memory bank for use in subsequent frames.  The memory attention operation takes the per-frame embedding from the image encoder and conditions it on the memory bank to produce an embedding that is then passed to the mask decoder. 

![sam2 architecture](https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_architecture.png)

This app allows you to fine-tune SAM 2 on custom dataset. You can define model checkpoints, data split methods, training hyperparameters and many other features related to model training.

# How To Run

Select images project, select GPU device in "Agent" field, click on `RUN` button:
  
# Screenshot



# Acknowledgment

This app is based on the great work `Segment Anything 2`: [github](https://github.com/facebookresearch/segment-anything-2). ![GitHub Org's stars](https://img.shields.io/github/stars/facebookresearch/segment-anything-2?style=social)