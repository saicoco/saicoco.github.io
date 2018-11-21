---
title: Pixle-Anchor,TextBoxes++与EAST的结合
layout: post
date: 2018-11-21
categories: 
- paper_reading
tag: paper
blog: true
start: true
author: karl
description: Scene Text Detection
header:
   image_fullwidth: "../downloads/pa/1.png"
feature: ../downloads/pa/1.png
---  

## 前言
pixel-anchor是云从科技昨天放出来的论文，文章提出了east和Textboxes++的合体模型，通过结合anchor-based和pixel-based的检测方法的特性，达到了SOTA。不过就整个框架而言，创新点虽然不多，但是预感会带起一波检测与分割结合的文字检测方法。

## 文章脉络
- anchor-based和pixel-based方法的优缺点
- 网络结构
- 结果分析
- 模型分析

### anchor-based和pixel-based的检测方法  
anchor-based的方法可以分为两个派系，一类是faster-rcnn,另外一类是SSD系列。  
其中，基于faster-rcnn的方法具体代表有如下方法：
- RRPN：提出带角度的anchor,设计的anchor需要足够多的角度以及尺度覆盖场景中的文本
- R2CNN：提出不同感受野的roipooling:7x7,3x11, 11x3用来检测多角度的文本
- CTPN：更改RPN anchor为水平方向的序列anchor,利用后处理将anchor连接成行
- InceptText：基于FCIS，加入Deformble conv和deformable PSROIpooing回去较强的感受野，同时加入inception module获取丰富感受野，检测文字
- FTSN：InceptText的平民版本
- etc  


而基于SSD的方法具体代表如下：
- TextBoxes and TextBoxes++：基于SSD,更改多尺度层中国的anchor尺度，并加入对倾斜框支持
- SegLink：基于SSD，将文本分为segments和Links，多尺度预测segments和link，利用后处理方法连接成文本
- etc  

这类方法主要依赖anchor的选取，因为文本的尺度变化剧烈，使得此类方法anchor的数量较多，进而效率较低。同时由于anchor的匹配机制，使得recall通常较高；对比Pixel-based的方法，如：
- pixellink：基于FCN分割网络，加入对pixel score Map的预测和当前像素与周围像素link的预测，后处理获得文本实例
- sotd：纯分割网络，加入border，用来分割密集文本。后处理通过minAreaRect获得检测框
- PSENet：FPN,预测不同大小的kernel,通过扩张算法得到各自的文本实例
- EAST：resnet50-based Unet，加入geo_map和score_map的预测，最后通过每个像素预测框Nms得到最后的预测框
- DDR：EAST的孪生版本
- FOTS：基于EAST改进，加入OHEM, ROIRotate以及识别分支
- etc  

通常为分割+回归，或者单独的分割接后处理的形式。这类方法基础网络多为Unet或者FPN，因此对小目标分割具有一定的优势，但是其回归的方式多依赖网络的感受野：如east，DDR， FOTS。虽然通过一些样本挖掘的方法可以获得一定的提升，但是感受野不足导致此类模型在回归较长文本或者较大文本时，容易出现检测不全或者丢失的情况。

上述方法的回归方式可以用下图表示：  
![img](../downloads/pa/2.png)  