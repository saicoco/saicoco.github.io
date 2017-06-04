---
layout: post
title: Python generator 在读取大数据时的应用
tag: python
category: python
comments: true
blog: true
date: 2016-07-13
---

最近苦于电脑内存较小16G,在做深度学习时，由于keras中fit方法需一口气将数据读入内存，此时当数据较大时，占用系统交换分区的同时，还会带来运行较慢速度，甚至内存不足。经过几天的折腾，发现生成器(Generator)可以很好的解决此类问题，先贴代码吧：

```
# -*- coding: utf-8 -*-
# author = sai
import numpy as np
import h5py
import random
from PIL import Image
import threading
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def load_audio(labels, count, audio, tags):
    '''
    根据标签生成与人脸数据对应的语音数据
    '''
    # 得到标签相等得行
    idx = [np.where(np.argmax(tags, axis=1) == labels)[0]]
    # print idx
    np.random.seed(count)
    audio_data = np.zeros((375,))
    # rn = np.random.sample(idx[0], 5)
    rn = [np.random.choice(idx[0]) for i in xrange(5)]  # 从item中随机选取5个数，即选取5行
    # print audio[rn].shape
    audio_data = audio[rn].reshape(375)
    return audio_data
@threadsafe_generator
def image_generator(batch_size, train=True):
    images = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)
    labels = []
    count = 0
    if train:
        path = './BingBang/BBT-train-file-list.txt'
    else:
        path = './BingBang/BBT-test-file-list.txt'
    with open(path, 'r') as f:
        readlines = f.readlines()
        print 'shuffle'
        random.shuffle(readlines)
        while 1:
            for i, item in enumerate(readlines):
                line = item.split(' ')
                fileuint = line[0].split('\\')
                label = int(line[1])
            #        print fileuint
            # windows 文件路径格式转换为 Linux 文件格式
                filename = './BingBang/' + fileuint[0] + '/' + fileuint[1] + '/' + fileuint[2] + '/' + fileuint[3]
                # img = cv2.imread(filename)
                # img = cv2.resize(img, dsize=(50, 50))
                img = Image.open(filename)
                img = img.resize((50, 50))
                images[count] = img
                labels.append(label)
                count += 1

                if count==batch_size:
                    images = images.transpose(0, 3, 1, 2)/255.
                    yield (images, labels)
                    count = 0
                    labels = []
                    images = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)

def av_generator(batch_size, train=True):
    images = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)
    audio_data = np.zeros((batch_size, 375))
    labels = []
    count = 0
    if train:
        path = './BingBang/BBT-train-file-list.txt'
    else:
        path = './BingBang/BBT-test-file-list.txt'
    with open(path, 'r') as f:
        audio_path = './data/BigBang/audio_merge/audio_samples.mat'
        with h5py.File(audio_path, 'r') as ff:
            audio = ff[ff.keys()[0]][:]
            tags = ff[ff.keys()[1]][:]
        readlines = f.readlines()
        print 'shuffle'
        random.shuffle(readlines)
        while 1:
            for i, item in enumerate(readlines):
                line = item.split(' ')
                fileuint = line[0].split('\\')
                label = int(line[1])
                audio_data[count] = load_audio(label, count, audio, tags)
            #        print fileuint
            # windows 文件路径格式转换为 Linux 文件格式
                filename = './BingBang/' + fileuint[0] + '/' + fileuint[1] + '/' + fileuint[2] + '/' + fileuint[3]
                # img = cv2.imread(filename)
                # img = cv2.resize(img, dsize=(50, 50))
                img = Image.open(filename)
                img = img.resize((50, 50))
                images[count] = img
                labels.append(label)
                count += 1
                if count==batch_size:
                    images = images.transpose(0, 3, 1, 2)/255.
                    yield ([images, audio_data], labels)
                    count = 0
                    labels = []
                    images = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)
                    audio_data = np.zeros((batch_size, 375))
```  

由于已经由保存好的h5数据，因此想着利用双层生成器达到不断生成数据的效果，但遇到的问题就是不知道何时会停止，造成难以管理等问题，翻看[keras issue](https://github.com/fchollet/keras/issues/1627),可以通过生成路径来获得对应数据，也算机智了一把。
代码中文本文件用来生成的图片名和对应label，利用yield函数，达到占用较小内存，却可以实现读取较大数据的效果，算是时间换空间的做法,在最近版本keras中加入多线程，pickle_safe和nb_worker来实现多线程，此处参看网上线程安全装饰器，可以保证线程安全。同时keras中[ImageDataGenerator](http://keras.io/preprocessing/image/)提供了生成器接口，使得这一方法使用较为方便。
在此过程中发现，可以利用多线程加速读取数据，但设计线程安全问题，此处就不深究了。
