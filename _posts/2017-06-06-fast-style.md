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

{% highlight html %}
<figure class="third">
	<img src="/downloads/fast_style/arch.png">
	<figcaption>模型Pipeline</figcaption>
</figure>
{% endhighlight %}　


