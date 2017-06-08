---
title: "Mxnet学习笔记(3)--自定义Op"
layout: post
date: 2017-01-13
tag: mxnet
blog: true
author: karl
category: mxnet
description: 自定义Op
---   

## 前言　　

今天因为要用到tile操作(类似np.tile，将数据沿axises进行数据扩充)，结果发现mxnet中没有，而且很多操作都没实现，详细完成
度可以参看[`issue`](https://github.com/dmlc/mxnet/issues/3200),还在完成中，不过这并不影响我们要用的操作，这里我们
需要实现自己的Op。当然，在官方的`example/numpy-ops`中已经给出部分例子。这里具体的记录一下。　　

## 自定义Op  

自定义op都是去继承`operator.py`中的类，其中提供如下几类：　　

* `operator.py`  
	* CustomOp(object)    
	* CustomOpProp(object)  
	* NDArrayOp(PythonOp)  
	* NumpyOp(PythonOp)  
	* PythonOp(object)    

这里很清晰的可以看出，operator分为两条路线，一条路线为`CustomOp`, 另外一条路线为继承`PythonOp`,这里我们就分为两部分分别介绍这两条路线。　　

### CustomOp类　　

这条路线是有三步组成，第一步继承`CustomOp`,重写方法`forward()`和`backward()`,然后继承`CustomOpProp`,重写成员方法，并在方法`create_operator`中
调用之前写好的Op,第三步调用`operator.register()`对操作进行注册。具体我们结合官方代码`example/numpy-ops/custom_softmax.py`来解释,代码如下：　　

```python
class Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SoftmaxProp, self).__init__(need_top_grad=False)
    
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return Softmax()
```


上述代码是对softmax的自定义，在类`Softmax`中重写`forward()`和`backward()`，这里与caffe中定义层操作类似，`forward()`中定义层的前向操作，`backward()`中
定义反向传播的梯度计算。在完成定义之后，在类`SoftmaxProp`中`create_operator()`调用并返回`Softmax()`实例。那么第三步`register`如何实现，可以看到，
在`SoftmaxProp`中带有装饰器`mx.operator.register()`,等价于操作`register("custom_op")(CustomOpProp)`,这里即在代码运行前即完成了该Op的
实例化，与`optimazer`的装饰器类似。　　

这里需要注意，这条路线中数据流是以`mx.nd.NDArray`格式传输的，如果在`forward()`与`backward()`中使用numpy函数，那么可利用`mx.nd.asnumpy()`将数据转换为numpy.ndarray进行操作。

### PythonOp类　　

这条路线，`PythonOp`类为基类，而我们大多定义Op时不会去继承它，而是使用他的subclass: `NDarrayOp`、`NumpyOp`。这条路线不会像继承`CustomOp`那样需要三步，这里我们也是只讨论如何继承并定义操作，不去探究
这两个类的实现细节。还是拿官网例子来讲。上代码：　　

```python
class NDArraySoftmax(mx.operator.NDArrayOp):
    def __init__(self):
        super(NDArraySoftmax, self).__init__(False)
        self.fwd_kernel = None
        self.bwd_kernel = None
    
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        if self.fwd_kernel is None:
            self.fwd_kernel = mx.rtc('softmax', [('x', x)], [('y', y)])
        self.fwd_kernel.push([x], [y], (1, 1, 1), (x.shape[0], 1, 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        y = out_data[0]
        dx = in_grad[0]
        if self.bwd_kernel is None:
            self.bwd_kernel = mx.rtc('softmax_grad', [('y', y), ('l', l)], [('dx', dx)])
        self.bwd_kernel.push([y,l], [dx], (y.shape[0],1,1), (y.shape[1], 1, 1))
````  

继承`NDArrayOp`其实和`NumpyOp`类似，不同之处在于`forward()`和`backward()`重写方式使用函数不同，`NDArrayOp`中需要使用`mx.nd`中的操作，而
`NumpyOp`可以使用`numpy`中的操作。总之重点在`forward()`和`backward()`。当然，如此的自定义方法在使用时需要先定义类对象才可以使用。即与`CunstomOp`
的定义时间不同。　

### 成员方法`list_arguments`，`list_outpus`，`infer_shape`

虽然继承方法不同，但是效果是一样的，`forward()`和`backward()`是对Op操作的定义，剩余三个成员方法则是对Op接口的描述。　　

#### `list_arguments`  　　

该方法主要是对该Op定义时形参的命名，如上述多为`['data', 'label']`,那么该Op在使用时形参必须为`data`和`label`。这里也可以看出mxnet是用过名字
寻找变量的，DataIter,optimazer也是如此。　　

#### `list_outputs`  

同样的，该方法定义了输出变量的名字，一般为opname+'_output'。　　

#### `infer_shape`  

该方法用于在给定输入时，获取该Op的输出shape。当然，在我们自定义时，需要自己设计Op的输入和输出shape。　　

---  

以上就是自定义Op时需要做的事情，重点还是`forward()`和`backward()`，有时候无头绪的时候可以参考caffe的写法获得灵感。接下来我用例子来说描述一下上述方法。　　


```python
import mxnet as mx
import numpy as np
class TileLayer(mx.operator.NumpyOp):
    def __init__(self, tiles, axis):
        super(TileLayer, self).__init__(False)
        # tiles可以为list或者一个数
        self.tiles = tiles
        self.axis = axis
    def list_arguments(self):
        return ['input']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0] + [self.tiles]
        return [data_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y = np.tile(x, reps=self.tiles)

    def backward(self, out_grad, in_data, out_data, in_grad):
        bottom_diff = in_grad[0]
        top_diff = np.sum(out_grad[0], axis=self.axis)
        bottom_diff = top_diff

if __name__ == '__main__':
    import logging
    from collections import namedtuple
    Batch = namedtuple('Batch', ['input'])
    logging.basicConfig(level=logging.INFO)
    a = mx.sym.Variable('data')
    custie = TileLayer(tiles=10, axis=2)
    tiles_a = custie(input=a, name='tileop')
    arg_shapes, out_shape, aux_shape = tiles_a.infer_shape(data=(2, 3))
    logging.info('arg_shape:{}\n, out_shape:{}\n, aux_shape:{}\n, output_blob:{}'.format(arg_shapes, out_shape, aux_shape, tiles_a.list_outputs()))
    exe = mx.module.Module(symbol=tiles_a, logger=logging)
    exe.bind(data_shapes=[('data', (1, 10, 10))], inputs_need_grad=True)
    # exe.init_params()
    # exe.init_optimizer()
    # data1 = [mx.nd.ones((1, 10, 10))]
    # exe.forward(Batch(data1))

    # print exe.get_outputs()[0].asnumpy().shape
    # top_grads =np.random.random(size=(1, 10, 10, 10))
    # exe.backward(out_grads=top_grads)
    # print exe.get_input_grads()[0].asnumpy()
````  

以上为定义的tile操作，这里没有做完全的tile操作，只是可以在最后的axis进行数据的tile操作。`forward`中用`numpy.tile`实现，`backward`中参考caffe
中的TileLayer实现，这里代码运行结果：　　

```
INFO:root:arg_shape:[(2L, 3L)]
out_shape:[(2L, 3L, 10L)]
aux_shape:[]
output_blob:['tileop_output']
```  

上述代码因为在`list_arguments`中定义了形参名字为`input`,因此在使用是形参必须为`input`,结果中也可以看到，`infer_shape`以及`list_output`的结果，基本细节就是上述。　　

在我们定义好Op后，我们需要通过`mx.mod.Moudle()`将Op进行整合，并通过`bind()`来申请内存，在此之后，我们可以通过以下两种方法训练它：　　

* 分别调用`init_params()`初始化参数(当然这里没有参数需要初始化)，`init_optimazer()`初始化optimazer,接下来就可以通过`forward()`和`backward()`进行前向反向传播训练模块。　　
* 或者直接调用`fit()`方法进行训练，因为`fit()`中包含初始化操作。　


关于`Moudle`可以参看[`mx.mod.Module`](http://mxnet.io/api/python/module.html)，当然参看例子也可以看这篇[博客center_loss](https://pangyupo.github.io/2016/10/16/mxnet-center-loss/)



