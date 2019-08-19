---
title: Cascade-RCNN
layout: post
date: 2019-06-28
categories: 
- paper_reading
tag: paper
blog: true
start: true
author: karl
description: Scene Text Recognition



---



Cascade-RCNN发现了训练样本与detector之间的关系：特定的iou阈值得到的proposal可以训练得到对应的detector。



### 发现问题  

![image-20190628130757158](../downloads/cascade_rcnn/image-20190628130757158.png)

文章给出第一张图，为了说明不同iou的proposal与detector的关系，作者首先论述了测试阶段不同iou阈值会带来不同的检测结果，如图a,b所示。