---
title: "Mxnet学习笔记(2)--自定义DataIter"
layout: post
date: 2017-01-12
tag: mxnet
blog: true
author: karl
category: mxnet
description: 自定义DataIter
---   

## 前言　　

之前在GPU集群上配置的caffe因为一系列人为因素崩溃，搭建的Tensorflow由于cuda版本太低有些实验不能跑，
而恰逢管理员不在，只好找一款不受这些因素影响的框架，之前阅读过mxnet源码，源码不多，很容易懂。于是配置
了一把，成功只好用了。而我的实验基于多源数据，即包含两种输入，这里是face images和audio数据，如果单纯使用
官方提供DataIter不能够完成任务，只好自己写(由于数据较大，直接使用NDArrayIter不现实，不如直接自己重新设计
一种DataIter)，当然本文章着重讲解如何自定义DataIter,细节还需参看源码。
话不多说，上干货。

## 目录　　

* Mxnet中的DataIter
    * DataIter  

    * NDArrayIter  

    * MXDataIter  


* 自定义DataIter  

## Mxnet中的DataIter  

DataIter类对象都在模块`io.py`中，而所有的DataIter都继承于基类`DataIter`,其中`DataIter`源码如下：　　

```python
class DataIter(object):

    def __init__(self):
        self.batch_size = 0

    def __iter__(self):
        return self

    def reset(self):
        pass

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        pass

    def getdata(self):
        pass

    def getlabel(self):
        pass

    def getindex(self):
        return None

    def getpad(self):
        pass
```  

---  

由以上代码可以看出，DataIter是一个迭代器，核心部分在方法`next()`,而其中涉及方法`getdata(), getlabel(), getpad(), getindex()`,但这是在不
重写方法`next()`前提下，我们需要提供这四个方法；而如果重写`next()`,我们仅需要为`DataBatch`提供batch大小的data,label即可，置于pad等方法我们可以忽略或者借鉴
其他DataIter.接下来我们就这几个方法看看官方提供DataIter干了些什么。　　

### NDArrayIter  
`NDarrayIter`由名字可以看出它是基于ndarray的数据迭代器，即数据来源是numpy数据。由[官方文档`mxnet.io.NDArrayIter`](http://mxnet.io/api/python/io.html#mxnet.io.NDArrayIter)可知，
`NDarrayIter`参数主要如下：　　

Parameters:   

* data (NDArray or numpy.ndarray, a list of them, or a dict of string to them.) – NDArrayIter supports single or multiple data and label.  
* label (NDArray or numpy.ndarray, a list of them, or a dict of them.) – Same as data, but is not fed to the model during testing.  
* batch_size (batch_size)  
* shuffle (boolean)  
* last_batch_handle ('pad', 'discard', 'roll_over')  

上述五个参数，最主要的是前三个，对于data我们可以提供ndarray或者NDArray,结构可以使列表，字典形式。这里为什么可以ist, dict.我们来看源码，在类NDArrayIter中数据初始化时调用了下面方法方法  


```python  
def _init_data(data, allow_empty, default_name):
    assert (data is not None) or allow_empty
    if data is None:
        data = []

    if isinstance(data, (np.ndarray, NDArray)):
        data = [data]
    if isinstance(data, list):
        if not allow_empty:
            assert(len(data) > 0)
        if len(data) == 1:
            data = OrderedDict([(default_name, data[0])])
        else:
            data = OrderedDict([('_%d_%s' % (i, default_name), d) for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError("Input must be NDArray, numpy.ndarray, " + \
                "a list of them or dict with them as values")
    for k, v in data.items():
        if not isinstance(v, NDArray):
            try:
                data[k] = array(v)
            except:
                raise TypeError(("Invalid type '%s' for %s, "  % (type(v), k)) + \
                    "should be NDArray or numpy.ndarray")

    return list(data.items())


class NDArrayIter(DataIter):
    def __init__(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad'):
        super(NDArrayIter, self).__init__()
        self.data = _init_data(data, allow_empty=False, default_name='data')
        self.label = _init_data(label, allow_empty=True, default_name='softmax_label')
        # shuffle data
        if shuffle:
            idx = np.arange(self.data[0][1].shape[0])
            np.random.shuffle(idx)
            self.data = [(k, array(v.asnumpy()[idx], v.context)) for k, v in self.data]
            self.label = [(k, array(v.asnumpy()[idx], v.context)) for k, v in self.label]
```  

　

即如果我们给出data和label均为list,那么在`_init_data`中处理最后会得到一个有序字典(OrderedDict),并赋予默认的name,数据为data,标签名字为softmax_label.这里注意，**提供的名字必须与搭建网络时变量中的名字对应，不然会报错。**
当然，如果这里data,label为字典类型时，那么在执行方法`_init_data`时会使用字典中key代替defaultname。这里还是要注意name,name,name(重要的事情说三遍)。　　

而我们在源码中可以看到，在后面装饰器装饰的方法`provide_data`,`provide_label`，分别为向搭建好的sym中供应name和此次迭代中提供的数据shape.即mxnet的数据供应过程是通过名字和shape来实现的。　　

```python
    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]
```  

以上便是NDArrayIter的核心部分，其余跟进阅读源码即可。　　

### MXDataIter  

`MXDataIter`是mxnet中的标准数据迭代器，在官方例子中没看到使用方法，同样继承与DataIter,并重写getXXX方法，方便在next()方法中`mx.io.DataBatch()`提供参数。这部分需要进一步补充，以后遇到使用的例子进行补充。  

## 自定义DataIter  

这里自定义DataIter,其实和上述迭代器方式相同，继承DataIter即可，因为DataIter中构成迭代的部分主要在方法`next()`中，而其中集中于`DataBatch`,这里我们看一下`DataBatch`的结构：　　

---  
```python
class DataBatch(object):
    def __init__(self, data, label, pad=None, index=None,
                 bucket_key=None, provide_data=None, provide_label=None):
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index

        # the following properties are only used when bucketing is used
        self.bucket_key = bucket_key
        self.provide_data = provide_data
        self.provide_label = provide_label
```   

可以看到，只要赋值给`DataBatch`中参数data,label就可以达到数据迭代的效果，因为每个DataBatch是mxnet中默认的mini-batch数据对象。所以这里有两条路线可以走：　  

* 自定义自己的数据生成器，可以不断地生成batch_data, batch_label供给DataBatch    

* 重写方法getXXX(),即`getdata`, `getlabel`，使用父类方法`DataIter.next()`默认配置即可。　　

其实两种方法原理皆是为了想DataBatch提供数据，事不过改写不同的地方。不过在coding时需要注意以下几点：　　

* 注意`provide_data|label()`方法提供的名字需要和sym构建过程中有关输入变量的name一致,并同时提供batch大小的数据的shape　　

* 如果data中包含多个输入，比如包含image, wordvec,在`pprovide_data|label`中需要分开写：`return [('data1', shape1), ('data2', shape2)]`,同理标签如果有多个输出，也应该如此设计。　　


---  
以上就是最近自定义是遇到的问题解决，后续会不断补充，有一句说一句，mxnet还是很优秀的，代码读起来很舒服。
