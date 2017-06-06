---
title: Fast Style Transfer
layout: page
date: 2017-06-06
categories: 
- paper_reading
tag: paper
blog: true
start: false
author: karl
description: image_caption
feature: /downloads/fast_style/head.png
--- 

Style Transfer是比较火的一个算法，而这篇文章出来很久了，拾起来读一读，谈谈想法．　　

## 引言　　
以往的style transfer的工作多为利用content和style两张图片进行迭代生成风格化图片，而这也是以往工作速度较慢的原因．
最近有GAN的做法，如CycleGAN, Pix2Pix等，这类工作多为寻找一个生成模型，使得generator得到的图片具有style的风格，
训练过程相当于两组数据的对抗，对抗结果就是输入generator的图像可以以假乱真，即具有了discriminator认可图片的特性．
当然，这是GAN的做法;那么对于常规的做法，是不是可以训练一个模型，使得一张图片输入其中，可以得到其对应的style transfer
image. 答案是肯定的．　　

## Architecture  
模型结构如下图所示：　
![arch](/downloads/fast_style/arch.png)  

对于一张输入图片$$\mathbf{x}$$而言，通过Image Transform Net得到$$\mathbf{\hat{y}}$$,这里的Transfor Net就
是我们要学习的模型，这个模型相当于encoder2decoder的结构, 而$$\mathbf{\hat{y}}$$大小应该与VGG输入图片大小一致．
在得到$$\mathbf{\hat{y}}$$之后，图中的后半部分为loss network, 顾名思义，transform net的损失函数是由后半部分网络
定义的．  

这里分别定义了三种损失, 按照文中说法分别如下：　　
* Pixel loss  
* style reconstruction loss  
* feature reconstruction loss  

### Pixel loss  
Pixel loss主要为了防止内容的丢失，即需要生成的风格图像与目标图像像素匹配时，使用该损失函数．当然，这里公式比较好理解：　　

$$
\begin{equation}
l_{pixel}(\hat{y},y) = \frac{||\hat{y}-y||_{2}^{2}}{CHW}
\end{equation}
$$    

就是对应位置像素的损失．　　

### Style Reconstruction Loss  
风格重构损失即transform net的输出图像通过vgg得到各层gram matrix与提供风格图像得到各层gram matrix之间的重构损失，这里gram matrix用来表示风格，纹理，在Gatys et al[^2]中又提到．类似的公式：　　

$$
\begin{equation}
l_{style}^{f, j}(\hat{y},y)=||G_{j}^{f}(\hat{y} - G_{j}^{f}(y)||_{F}^{2}
\end{equation}
$$  

这里的f表示VGG,j对应VGG的提取层，虽然使用F范数，道理和欧式距离类似．G表示gram matrix，对于VGG feature maps,维度通常为$$(N, H, W, C)$$,求gram matrix时需要reshape为$$(N, C, HW)$$, gram matrix求解如下：　　

$$
\begin{equation}
G_{j}^{f}(x) = \frac{MM^T}{C_{j}H_{j}W_{j}}
\end{equation}
$$  

### Content Reconstruction Loss  

内容重构损失主要为了防止原始内容的丢失，因为希望最终得到的图片具有原始图片的内容，兼具风格图像的风格．公式如下：　　

$$
\begin{equation}
l_{feat}^{f, j}(\hat{y},y)=\frac{1}{C_{j}H_{j}W_{j}}||F_{j}^{f}(\hat{y} - F_{j}^{f}(y)||_{2}^{2}
\end{equation}
$$  

典型的欧式距离，这里F表示VGG从输入到feature map的变换．　　

## Fast Style Transfer  
介绍完基本部件，该如何style transfer. 归根到底，只要学习好transform net,对于一张图片，我们就可以得到对应的风格化图片．
其实这里也暗含一点，faste style transfer[^2]只能学习一种风格，即一种风格对应一个模型．　　

对于CNN来说，低层包含像素信息较多，高层则像素信息损失严重，因为pooling的存在，高层视野较为开阔，因此高层包含较多语音信息，
可以理解为高层的风格信息较多．因此文章做了想Gatys et al.[^1]的做法，对于不同层设置不同的权重，低层的内容重构损失设置较大
权重，高层风格重构损失设置较大的权重，也就是style和centent的trade-off. 最终的损失函数如下：　　

$$
\begin{equation}
\hat{y} = \arg\min_{y} \lambda_{c}l_{feat}^{F,j}(y, y_c) + \lambda_{s}l_{style}^{f,j}(y, y_s) + \lambda_{TV}l_{TV}
\end{equation}
$$   


其中$$\lambda_{c, s}$$表示的是content,style的权重，j表示的是VGG的feature层．可以看到，整个模型的优化就是寻找transform net的权重，使得$$x$$输入之后，得到的$$y$$可以使得整个损失最小．　　  

就写到这里吧，想跑实验的可以戳[faste style transfer](https://github.com/lengstrom/fast-style-transfer)


## References  
[^1]: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
[^2]: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/)

