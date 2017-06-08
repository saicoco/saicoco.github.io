---
title: "由MCB想到的"
layout: page
date: 2017-01-09
tag: thinking
blog: true
start: true
author: karl
category: thinking
description: 多模态特征融合
---    

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

最近一直在努力将MCB应用与语音和人脸的特征融合，在一遍一遍的阅读源码之后，想在github上issue，但觉得不足以表达想法，因此还是以博文形式记录，以防以后用到。　　

## MCB步骤解剖　　

### Input    

    * bottom1: [N, C1, H, W]  
    * bottom2: [N, C2, H, W]  

这里的输入为两个四维张量，但是在计算时，代码中体现为[N, C, HW],即将向量转换为一维向量进行计算，这样也就可以很好的理解文章中在卷积层沿着通道方向的外积手段。　　

### Forward    

#### 降维　　

降低维度的目的旨在避免维度灾难，而降维方向在通道方向：输入为C个通道，输出为num_outputs_。具体如下：　　

* 获得两个随机索引randh, rands,其中randh_为C个数，每个位置的数值为来自[0, num_outputs_)的随机数；rands_为C个数，值为[-1, 1]。降维逼近公式为：　　

$$Z_{h_i} = Z_{randh_i} + rands_i \times bottomx_{i}$$  


由上式，可以看出的是，如果操作向量是两个一维向量，那么降维后的向量的每个位置上的元素值由整个过程中抽中的randh_i和rands_i决定的；如果操作向量是张量，那么最后得到的向量上每个
位置上为一个二维向量，其来自于randh_i,rands_i的对应原始向量中元素的组合。**可以注意到的是，这个阶段是不可学习的，即没有对应的映射函数，完全随机逼近**。　　

#### 频域内积　　

利用上述得到的逼近向量b1, b2,转换至频域复数，对其进行内积：　　

$$complexZ = FFT(b1) \cdot FFT(b2)$$  

然后通过iFFT得到外积的后的值：$$Z = iFFT(complexZ)$$,其中Z为$$(N, num_outputs_, HW)$$。　　

----  

### 想法　　

由上述一系列操作，可以看到MCB的整个过程是不涉及学习参数的，只是单纯做了外积操作。因为外积操作也可以视为认为定义的一种操作，而且MCB的作用最小范围为通道，并没有涉及到一个通道内feature map上值之间的交互，是不是可以将这部分操作再做细化，即为机器提供更为丰富的关系，使得在数据充足的情况下驱动机器从中选取良好的关系。　　

对于来自两个bottom的每个feature map，将其reshape为一维向量，通过两个映射w：  


$$Z=F(\sigma (w_1 \cdot b_1), \sigma (w_2 \cdot b_2))$$

当然，这里可以加入bias,然后对于这种映射关系加入损失函数，目的使得映射后的分布与原始分布尽可能的一致(易分类)，或者是距离尽可能的近，同时保证Z的大小(HW)与b相同，这样可以用作加入注意力。　　

以上为局部修改的想法，未经证实，记录以开学来尝试。
