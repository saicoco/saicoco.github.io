---
layout: page
title: Effective Python 31
tag: python
category: python
comments: true
blog: true
data: 2016-09-19
---

### 第31条　用描述符来改写需要复用的@property方法　　

`@property`有个缺点，就是不便于复用。受它修饰的方法，无法为同一个类中其他属性所复用，与之无关的类，也无法复用这些方法。为了避免为各种方法反复的加上`@property`,提出描述符。描述符格式如下：　　

```python

class grade(object):
    def __set__(*args, **kwargs):
        #...

    def __get__(*args, **kwargs):
        #...
```  


下面利用一个样例来解释描述符。　　

```python

from weakref import WeakKeyDictionary
class grade(object):

    def __init__(self):
        self._value = WeakKeyDictionary()

    def __set__(self, instance, value):
        if not(0 <= value <= 100):
            raise ValueError('grade must be between 0 and 100')
        self._value[instance] = value

    def __get__(self, instance, owner):
        # add multiple instance
        if instance is None: return self
        return self._value.get(instance, 0)


class exam(object):

    math_grade = grade()
    writing_grade = grade()
    science_grade = grade()

if __name__ == '__main__':
    fi_exam = exam()
    fi_exam.math_grade = 82
    fi_exam.writing_grade = 99

    print('math', fi_exam.math_grade)
    print('wirite', fi_exam.writing_grade)
    print('science', fi_exam.science_grade)

    se_exam = exam()
    se_exam.math_grade = 100
    print('fi_math', fi_exam.math_grade)
    print('se_math', se_exam.math_grade)
```  

上述代码时改进版本的描述符，这里需要明白的是，当程序访问到`exam`实例的描述符属性时，python会对这种访问操作进行转译：如果`exam`实例没有名为`writing_grade`的属性，那么python就会
转向`exam`类，在该类中查找同名类属性。这个类属性，如果实现了`__get__`和`__set__`方法的对象，那么python就认为此对象遵从描述符协议。　　

这里的`grade`被`fi_exam`和`se_exam`两个类实例共享，因此需要对每个实例进行字典式的存储。此处`dict.get(key, 0)`表示获取键值`key`对应的值，如果不存在键`key`,则返回0。而且初始化字典
使用`WeakKeyDictionary()`，该方法主要返回一个键弱引用的字典，即在运行期间发现这种字典持有的引用，是整个系统里面指向`exam`的最后一份引用，那么，系统会自动将该实例从字典的键中移除。
即当程序不再使用`exam`任何实例时，`_value`字典会是空的。　　　

#### 要点　　

* 如果想复用`@property`方法以及验证机制，那么可以自己定义描述符类。
* `WeakKeyDictionary`可以保证描述符类不会发生泄露内存。
* 通过描述符协议实现属性的获取和设置操作时，不要纠结于`__getattribute__`的方法具体运作细节。
