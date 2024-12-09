<div align="center" markdown>
<img src="https://github.com/user-attachments/assets/642ee655-2e73-4d59-8d45-411e2cedac24"/>  

# Serve Segment Anything Model 2.1
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Run">How to Run</a> •
  <a href="#Model-application-examples">Model application examples</a> •
  <a href="#Controls">Controls</a> •
  <a href="#Acknowledgment">Acknowledgment</a> 
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-segment-anything-2)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-segment-anything-2)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/serve-segment-anything-2.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/serve-segment-anything-2.png)](https://supervise.ly)
 
</div>

# Overview

Segment Anything Model 2 is a foundation model for interactive instance segmentation in images and videos. It is based on transformer architecture with streaming memory for real-time video processing. SAM 2 is a generalization of the first version of SAM to the video domain, it processes video frame-by-frame and uses a memory attention module to attend to the previous memories of the target object. When SAM 2 is applied to images, the memory is empty and the model behaves like usual SAM.

Unlike the first version of Segment Anything, the frame embedding used by the SAM 2 decoder is conditioned on memories of past predictions and prompted frames (instead of being taken directly from an image decoder). Memory encoder creates "memories" of frames based on the current prediction, these "memories" are stored in model's memory bank for use in subsequent frames.  The memory attention operation takes the per-frame embedding from the image encoder and conditions it on the memory bank to produce an embedding that is then passed to the mask decoder. 

![sam2 architecture](https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_architecture.png)

# How To Run

**Step 1** Select pretrained model architecture and press the **Serve** button

![sam2 pretrained models](https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_pretrained_models.png)

Alternatively, you can load your custom SAM 2 checkpoint:

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_load_custom_checkpoint.mp4" type="video/mp4">
</video>

**Step 2.** Wait for the model to deploy

![sam2 deployed](https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_deployed.png)

# Model application examples

Usage of Segment Anything 2 as a Smart Tool for image labeling:

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_smart_tool.mp4" type="video/mp4">
</video>

Video object segmentation and tracking:

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_video_tracking.mp4" type="video/mp4">
</video>

Automatic image mask generation without any prompts via [NN Image Labeling app](https://ecosystem.supervisely.com/apps/nn-image-labeling/annotation-tool):

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_auto_mask_gen_speedup.mp4" type="video/mp4">
</video>

Applying model to object in bounding box:

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_bbox.mp4" type="video/mp4">
</video>

Fast labeling of images batch via [Batched Smart Tool](https://ecosystem.supervisely.com/apps/dev-smart-tool-batched):

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/serve-segment-anything-2/releases/download/v0.0.1/sam2_batched_smart_tool.mp4" type="video/mp4">
</video>

# Controls

| Key                                                           | Description                               |
| ------------------------------------------------------------- | ------------------------------------------|
| <kbd>Left Mouse Button</kbd>                                  | Place a positive click                    |
| <kbd>Shift + Left Mouse Button</kbd>                          | Place a negative click                    |
| <kbd>Scroll Wheel</kbd>                                       | Zoom an image in and out                  |
| <kbd>Right Mouse Button</kbd> + <br> <kbd>Move Mouse</kbd>    | Move an image                             |
| <kbd>Space</kbd>                                              | Finish the current object mask            |
| <kbd>Shift + H</kbd>                                          | Higlight instances with random colors     |
| <kbd>Ctrl + H</kbd>                                           | Hide all labels                           |


<p align="left"> <img align="center" src="https://i.imgur.com/jxySekj.png" width="50"> <b>—</b> Auto add positivie point to rectangle button (<b>ON</b> by default for SmartTool apps) </p>

<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/119248312/229995019-9a9dece7-516f-4b44-8b73-cdd01c1a4178.jpg" width="90%"/>
</div>

<p align="left"> <img align="center" src="https://user-images.githubusercontent.com/119248312/229998670-21ced133-903f-48ce-babb-e22408d2580c.png" width="150"> <b>—</b> SmartTool selector button, switch between SmartTool apps and models</p>

<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/119248312/229995028-d33b0423-6510-4747-a929-e0e860ccabff.jpg" width="90%"/>
</div>

# Acknowledgment

This app is based on the great work `Segment Anything 2`: [github](https://github.com/facebookresearch/segment-anything-2). ![GitHub Org's stars](https://img.shields.io/github/stars/facebookresearch/segment-anything-2?style=social)
