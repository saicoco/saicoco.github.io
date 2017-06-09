---
title: PyQt从UI设计到style transefer实现
layout: post
date: 2017-06-09
categories: 
- qt
tag: qt
blog: true
start: false
author: karl
feature: /downloads/qt/qt.jpg
---

最近想做个界面，把上篇post的功能简单实现一下．　　

## 引言　　
　
PyQt快速制作桌面界面需要以下几步：　　

* 设计UI  
* .ui转换.py文件　　
* 调用.py实现功能细节．　　

### 设计UI  
通常，UI设计可以有两种方式，一种是纯手coding实现各种功能，另外一种就是我要写的这种，对于我这中小白还是比较好使的．如下图所示，就是工作区：　　

![ui](/downloads/qt/1.jpg)  

这里使用的是QTcreator,首先创建Qt工程，选择界面设计类，这样得到的工程文件中包含.ui文件，这样我们就可以使用"设计"选项来编辑UI文件．如上图所示，
主要分为五个区域，左边区域为组件区：按钮，滚动条等都在这里;中间为UI设计区，右边和右下角为属性，包括名字等各种属性，底部为按钮等组件动作的设置区．
UI设计真是技术活，我是干不来，很丑．　　

### ui转py文件　　

假设刚才UI文件保存为mainwindow.ui,那么可以通过下面命令将其转化为python的ui代码
```
pyuic4 -o ui.py mainwindow.ui
```

### 实现功能　　

线上一下我们的界面图：　　

![3](/downloads/qt/3.jpg)  

如上图所示，我们想做的是：打开content图片，然后在空白区域显示图片，然后在右上角style choices选项中选择风格（这里名字瞎起的，别在意），然后点击RUN按钮，
开始进行style transfer.  
因此，我们就需要在点击对应的按钮实现对应功能：　　

* 点击open content弹出文件对话框，选择文件并将文件显示在空白区域　　
* 点击style choices中任意一个之后可以选中我们要迁移的风格模型　　
* 点击run开始运行模型　　　　

直接贴代码了：　　
```python
# coding=utf-8
__author__ = 'JiaGeng'
from neural_windows.ui import Ui_MainWindow
from PyQt4.QtCore import  *
from PyQt4.QtGui import *
import sys, os
from transfer.evaluate import ffwd_to_img

class neural_style_transfer(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(neural_style_transfer, self).__init__(parent)
        self.setupUi(self)
        QObject.connect(self.shanshui, SIGNAL("clicked()"), self.fshanshui)
        QObject.connect(self.youhua, SIGNAL("clicked()"), self.fyouhua)
        QObject.connect(self.shuimohua, SIGNAL("clicked()"), self.fshuimohua)
        QObject.connect(self.contentButton, SIGNAL("clicked()"), self.show_contDialog)
        # QObject.connect(self.contentSlider, SIGNAL("valueChanged(int)"), self.get_content_slider)
        # QObject.connect(self.styleSlider, SIGNAL("valueChanged(int)"), self.get_style_slider)
        QObject.connect(self.runButton, SIGNAL("clicked()"), self.transfer)

        self.ckpt = ""
        self.style_weight = 0.
        self.content_weight = 1e-3
        self.content_img = None
        self.out_path = './out'
        self.outputs = []

    def fshanshui(self):
        self.ckpt = 'transfer/checkpoint/rain_princess.ckpt'

    def fyouhua(self):
        self.ckpt = 'transfer/checkpoint/scream.ckpt'

    def fshuimohua(self):
        self.ckpt = 'transfer/checkpoint/wreck.ckpt'

    def get_style_slider(self):
        value = self.contentSlider.value()
        self.style_weight = value * 0.1

    def get_content_slider(self):
        value =  self.contentSlider.value()
        self.content_weight = value * 0.1

    def show_contDialog(self):
        filename = QFileDialog.getOpenFileName(self, 'open file', './')
        assert filename.split('.')[-1] in ['jpg', 'png']
        self.content_img = str(filename)
        self.content_label.setPixmap(QPixmap(filename))

    def transfer(self):
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)
        device = '/cpu:0'
        out_path = self.out_path + '/out' + self.content_img.split('/')[-1]
        self.outputs = ffwd_to_img(self.content_img, out_path, self.ckpt, device=device)
        self.content_label.setPixmap(QPixmap(out_path))

if __name__=='__main__':
    app = QApplication(sys.argv)
    neural_st = neural_style_transfer()
    neural_st.show()
    app.exec_()
```

总的来说，首先我们的`neural_style_transfer`继承刚才我们转换的ui.py中的属性，其中每个按钮的名字在.ui文件中已经设置．这里将
每个按钮在点击之后调用对应的方法．如style choices点击时，获取pre-train模型权重; 点击run时，调用上篇Post中的算法得到结果，并
将结果显示在界面中;对于打开content文件，直接调用文件对话框就好．一下是运行后的结果：　　

![4](/downloads/qt/4.jpg)  

有一点不好，就是transfer结束之后模型仍在内存中，没有释放，这里有待解决．不过pyqt搭界面还是比较好用的，尤其对于我这中c plus plus不行的人．




