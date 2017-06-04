---
title: "Mxnet学习笔记(1)--源码中的装饰器理解"
layout: post
date: 2016-12-28
image: /downloads/mxnet_register/register.png
headerImage: true
tag: mxnet
category: mxnet
blog: true
author: karl
description: python装饰器
---  

今天趁着跑实验的间隙跑去阅读mxnet源码，看到了这么一段代码：　　


```python
class Optimizer(object):
    """Base class of all optimizers."""
    opt_registry = {}

    @staticmethod
    def register(klass):
        """Register optimizers to the optimizer factory"""
        assert(isinstance(klass, type))
        name = klass.__name__.lower()
        if name in Optimizer.opt_registry:
            print('WARNING: New optimizer %s.%s is overriding '
                  'existing optimizer %s.%s' % (
                      klass.__module__, klass.__name__,
                      Optimizer.opt_registry[name].__module__,
                      Optimizer.opt_registry[name].__name__))
        Optimizer.opt_registry[name] = klass
        return klass

    @staticmethod
    def create_optimizer(name, rescale_grad=1, **kwargs):
        """Create an optimizer with specified name.

        Parameters
        ----------
        name: str
            Name of required optimizer. Should be the name
            of a subclass of Optimizer. Case insensitive.

        rescale_grad : float
            Rescaling factor on gradient.

        kwargs: dict
            Parameters for optimizer

        Returns
        -------
        opt : Optimizer
            The result optimizer.
        """
        if name.lower() in Optimizer.opt_registry:
            return Optimizer.opt_registry[name.lower()](
                rescale_grad=rescale_grad,
                **kwargs)
        else:
            raise ValueError('Cannot find optimizer %s' % name)

register = Optimizer.register

@register
class SGD(Optimizer):
    """A very simple SGD optimizer with momentum and weight regularization.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
    """
    def __init__(self, momentum=0.0, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.momentum = momentum

```  

以上代码在Optimazer.py中，然后你会在model.py中看到构造Optimazer时仅仅使用`create_optimizer(*args)`便可以得到对应的Optimazer,这里我好奇它的执行顺序，入口为函数`create_optimizer()`,终点为`class SGD`,那么通过基类如何构造子类，注意到有装饰器`@register`,所以首先执行类方法`register`,然后执行构建基类，再构建SGD,于是我做了如下测试：　　

```python

class my_base(object):
    test_dict = {}

    @staticmethod
    def register(klass):
        name = klass.__name__.lower()
        print '{} is registing....'.format(name)
        if name in my_base.test_dict:
            print('WARNING: New test_item %s.%s is overriding '
                  'existing test_item %s.%s' % (
                      klass.__module__, klass.__name__,
                      my_base.test_dict[name].__module__,
                      my_base.test_dict[name].__name__))
        my_base.test_dict[name] = klass
        return klass

    @staticmethod
    def create_test_item(name, classes):
        if name.lower() in my_base.test_dict:
            return my_base.test_dict[name](classes)
        else:
            raise ValueError('Cannot find test_item %s' % name)

    def __init__(self, classes):
        self.classes = classes
        print 'base class'
register = my_base.register

@register
class sun(my_base):

    def __init__(self, classes):
        super(sun, self).__init__(classes)
        print 'sun'

if __name__ == '__main__':
    s = my_base.create_test_item('sun', 2)

```  

目的旨在看看如何执行的，运行结果如下：　　

```
sun is registing....
base class
sun
```  

这下就可以明白，调用类方法是不需要构建对象的，即调用方法`create_test_item`时基类`my_base`还未创建类对象，而且在debug时，发现`@register`在`s = my_base.create_test_item('sun', 2)`之前执行的，即在执行之前便已经加入到了`test_dict`中，这样在后续的`create_test_item`可以顺利的返回`sun(classes)`的结果，然后根据继承关系构建类对象。

总而言之一句话，装饰器运行在代码运行之前，即定义阶段就已经完成。
