---
layout: page
title: Effective Python 29
tag: python
category: python
comments: true
blog: true
data: 2016-09-19
---

### 写在前面的话　　

读这本书好久了，才读到第四章，而之前博客还没搭起来，因此从第四章开始做笔记。这本书源码可以在[此处](http://www.effectivepython.com/)找到，关于源码实例也可以在该网站找到。

#### 何为元类　　

元类(metaclass),模糊地描述了一种高于类，而又超乎类的概念。说白了，就是能够定制其他类的类，也可以称为描述类。
文章就此展开对类属性访问的一些技巧的描述：

### 第29条　用纯属性取代get和set方法　　

在其他语言中，为利于定义类的接口，多使用`setter`,`getter`方法，同时可以使得开发者更为方便的封装功能等操作。在python中，则无需此类方法，通常使用public属性直接对象调用属性即可修改。
而这是对于那些对类属性简单访问来说的，对于一些复杂操作，如在对某一属性赋值时，需要修改另一属性，此时除了繁杂的操作之外，python提供了修饰器`@property`来实现对属性的访问，同时以其对
应`setter`实现对属性的修改。如下述代码：　　


```python
class resistor(object):
    def __init__(self, ohms):
        self.ohms = ohms
        self.voltage = 0
        self.current = 0

class voltageres(resistor):
    def __init__(self, ohms):
        super(voltageres, self).__init__(ohms)
        self._voltages = 0

    @property
    def voltage(self):
        return self._voltages

    @voltage.setter
    def voltage(self, voltage):
        self._voltages = voltage
        self.current = self._voltages / self.ohms

if __name__ == '__main__':
    vol = voltageres(20)
    print vol.voltage
    vol.voltage = 200
    print vol.voltage, vol.current
```

针对上述代码，我们可以看到，利用`@property`可以实现对属性`voltage`的访问和修改，同时在修改`voltage`时，对`current`进行了修改。`@property`的优点在于简洁，可以实现对类属性的复杂访问操作，在
访问的同时可以进行一些额外的简单操作。值得注意的是，**在使用时，应将修改属性值的操作置于`setter`中，而不是`getter`中**。　　

#### 要点　　

* 编写新类时，应该使用简单的`public`属性来定义接口，而不要手工实现`set`和`get`方法。
* 如果访问对象的某个属性时，需要表现出特殊行为，则使用`@property`来定义行为。
* `@property`方法应该遵循最小惊讶原则，即应该符合广为人知的编程习惯。
* `@property`方法需要执行迅速，缓慢或复杂的工作，应该放在普通方法里边。　　

-------
