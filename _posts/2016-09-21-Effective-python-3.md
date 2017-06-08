---
layout: page
title: Effective Python metaclass
tag: python
category: python
comments: true
blog: true
data: 2016-09-21
---

### 写在前面的话　　

最近精神不大好，晚上总是睡不着，白天也不困，可闲下来就觉得很累。因此想着找个时间偷一下懒，休息一下，刚好老大让去北京风流几天，趁机解放下自己^_^。这篇文章我们正式讲元类。　　

### 元类　　

关于元类，在上篇博客我们说了，它是用来构造类的类。期间我看过一篇[博客](http://blog.jobbole.com/21351/)，里边详细讲解了元类的来由。python中，皆为对象。
而对于每个类来说，也是对象(实例)，他们是元类的实例。而元类都来源于**type**这一神奇的类，究其原因，`type`可以用来创建类，元类的概念可以从[博客](http://blog.jobbole.com/21351/)看到，这
里我们主要讲一下元类的使用。

#### metaclass的功能　　

* 拦截类的创建
* 修改类
* 返回修改之后的类  

Talk cheap, show you code!  

```python
class Field(object):
    def __init__(self):
        self.name = None
        self.internal_name = None

class Meta(type):
    def __new__(meta, name, bases, class_dict):
        for key, value in class_dict.items():
            if isinstance(value, Field):
                value.name = key
                value.internal_name = '_' + key
        meta = type.__new__(meta, name, bases, class_dict)
        return meta

class DatabaseRow(object):
    __metaclass__ = Meta
    pass

class BetterCustomer(DatabaseRow):
    first_name = Field()
    last_name = Field()
    prefix = Field()
    suffix = Field()

if __name__ == '__main__':
    foo = BetterCustomer()
    print('before:', repr(foo.first_name), foo.__dict__)
    foo.first_name = 'Euler'
    print('after:', repr(foo.first_name), foo.__dict__)

```  

就这么简单的一个程序，我们来分析一下元类到底干了些什么：在创建类`BetterCustomer`时，python会在类中寻找属性`__metaclass__`，若未找到，则从其父类`DatabaseRow`
寻找`__metaclass__`属性，并以此元类来创建该类，若在父类中仍未找到，则从内置模块中寻找可以创建此类的类。　　

依据此思路，我们可以看到，在类`BetterCustomer`创建时，python会去寻找元类，并在其父类`DatabaseRow`中找到了元类`Meta`,并利用此类来创建类`BetterCustomer`，有`Meta`
中可以看出，在创建类的过程中，提前将各个属性变为`protected`，而不是在类创建之后完成。而对于描述符，在访问属性时发生转译，遵从描述符协议。具体还是参看博客。　　

### 写在后面的话　　

最近心里烦，一度认为自己心态变了，又何尝不是渐渐发现身边事物变成了自己讨厌的东西，很可能连自己都变成了那个自己讨厌的人，调整下，如此下去，心会乱糟糟的。
Don't worry, be happy!!!
