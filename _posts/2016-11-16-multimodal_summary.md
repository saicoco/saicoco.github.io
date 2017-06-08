---
layout: post
title: ubuntu安装matlab2015b
tag: ubuntu
category: software
comments: true
blog: true
data: 2016-11-16
---    
因为matlab代码需要在ubuntu15.10下运行，因此需要安装matlab，刚好在别的群里边溜达的时候遇到一些人的分享，我保存在[百度云盘](https://pan.baidu.com/s/1kUCe2R9)了,安装过程还算顺利，这里小记一下，因为还是遇到一些小问题，之前遇到然后知难而退了，话不多说，开始步骤。  

### 1  

上述下载的文件包含.iso和crack文件，这里首先将.iso利用磁盘挂载器挂载，然后在其文件夹内打开终端运行：  

```
sudo ./install
```  

### 2  

在经过一系列小点点之后，进入安装页面，首先需要你填写激活码，激活码在crack文件夹内的readme文件里，当然，这里你需要的是选取单机版而不是服务器版本，在填好激活码之后，一路next，安装路径根据自己需要填写路径，其实默认就挺好的。  

### 3  

之后便是可爱的进度条，ubuntu下安装速度其实挺快的，在安装好之后，重新运行matlab，在你的matlab安装路径下找到`bin`文件夹，运行其下`matlab`脚本，如果需要权限，加上`sudo`，这次运行需要添加license文件，在crack文件中找到对应的linense文件就好，然后激活成功。  

### 4  
此时，运行matlab依旧会报错，错误为hostID的问题，这是需要将`crack/R2015b/bin/glnxn/`下的文件copy到matlab安装目录bin文件下的glnxn中，这样matlab最起码可以启动了。  

### 5  
此时运行matlab依旧会报错，错误为一大堆不支持，强行逼迫推出，这是说明ubuntu系统对matlab不支持，你需要安装matlab-support, 在软件中心或者直接运行如下命令即可：  

```
sudo apt-get install matlab-support
```  

命令运行期间，需要你填写matlab安装路径，在填写完毕之后，一路确定就好，此时matlab可以完美的运行了。
