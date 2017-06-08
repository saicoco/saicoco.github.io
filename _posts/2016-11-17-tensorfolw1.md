---
layout: page
title: tensorflow使用笔记(1)--全览
image: /downloads/tf/image_header.png
headerImage: true
category: tensorflow
tag: tensorflow
comments: true
blog: true
data: 2016-11-17
---  

今天开始计划开启一大块，主要关于tensorflow使用的，我想以blog的记录下来。因为实验需要，估计会用tensorflow实现LSTM等复杂操作，以此来熟悉tensorflow。  

还是从一个tutorial开始。一个完整的tensorflow程序可以分为一下几部分：  

* Inputs and Placeholders  
* Build the Graph  
    * Inference  
    * Loss  
* Traing  
* Train the Model  
* Visualize the Status  
* Save a Checkpoint  
* Evaluate the Model  
* Build the Eval Graph  
* Eval Output  

### Inputs and Placeholders  

对于一个完整的网络来说，必定有输入还有输出，而`Placeholders`就是针对网络输入来的，相当与预先给输入变量占个坑，拿mnist来说，占坑代码可以如下面的例子：  

```python
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,mnist.IMAGE_PIXELS))

labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

```  

上述代码相当于未mnist图片和标签分别占坑，而`tf.placeholder`参数可以如下面所示：  

```python
tf.placeholder(dtype, shape=None, name=None)
```  
即需要提供占坑数据类型`dtype`,占坑数据`shape`,当然也可以给它提供一个唯一的`name`。  

### Build the Graph  

因为tf是通过构建图模型来进行网络搭建的，因此搭建网络也就是'Build the Graph'。  

#### Inference  

首先就是构建图，利用一系列符号将要表达的操作表达清楚，以用于后续模型的训练。如下面代码：  

```python
with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],\
        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')

    biases = tf.Variable(tf.zeros([hidden1_units]),\
    name='biases')

```  
如上述代码，对于一个图的搭建，需要一些变量来支持我们的运算，比如矩阵相乘等，需要通过`tf.Variable`来声明变量，其参数格式如下：  

```python  

tf.Variable(self, initial_value=None, trainable=True, collections=None, validate_shape=True,\
    caching_device=None, name=None, variable_def=None, dtype=None)

```  
需要提供变量初始值`initial_value`, 是否接受训练`trainable`,对于`validate_shape`表示该变脸是否可以改变，如果形状可以改变，那么应该为`False`。对于每个变量，可以赋予不同的名字`tf.name_scope`。  

#### Loss  

在定义完图结构之后，我们需要有个目标函数，用作更新图结构中的各个变量。  

```python
labels = tf.to_int64(labels)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')

```  
如上，通过给定的`labels`占坑变量，完成手写数字识别的最后交叉熵函数。  

### Training  

在得到目标函数之后，我们就可以对模型进行训练，这里常用梯度下降法。在训练阶段，我们可以通过`tf.scalar_summary`来实现变脸的记录，用作后续的tensorboard的可视化，如：  

```python
tf.scalar_summary(loss.op.name, loss)
```  
然后通过`tf.SummaryWriter()`来得到对应的提交值。而对于模型的最优化，这里tf提供了很多optimazer,通常在`tf.train`里面，这里常用的是`GradientDenscentOptimizer(lr)`,然后通过调用：  

```python
train_op = optimizer.minimize(loss, global_step=global_step)
```  

### Train the Model  

在模型训练是，我们需要打开一个默认的图环境，用作训练，如：  

```python
with tf.Graph().as_default():
```  
以此来打开一个图结构，然后我们需要声明一个回话在所有操作都定义完毕之后，这样我们就可以利用这个session来运行Graph.可以通过如下方法声明：  

```python
with tf.Session() as sess:
    init = tf.initialize_all_variables()

    sess.run(init)

```
每次我们可以通过`sess.run`来运行一些操作，进而获取其的输出值，  

```python
sess.run(fetches, feed_dict=None, options=None, run_metadata=None)
```  
可以看到，run需要`fetches`，即操作，`feed_dict`为`fetches`的输入，即占坑变量与其对应值构成的字典。  

### Visualize the Status  

当然，在运行过程，我们可以通过可视化的操作来看网络运行情况。  
在之前的`tf.scalar_summary`,我们可以通过：  

```python  

summary = tf.merge_all_summaries()
```   
将在图构建阶段的变量收集起来，然后在session创建之后运行如下命令生成可视化的值。  

```python  

summary_str = sess.run(summary, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)
```   
其中`summary_writer`由如下得到：  

```python  

summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
```  
然后用tensorboard打开对应文件即可。  

### Save a Chenckpoint  

对于模型的保存，可以通过如下代码实现：  

```python  
saver = tf.train.Saver()  
saver.save(sess, FLAGS.train_dir, global_step=step)
```  
而载入模型可以通过简单的模型来实现魔心搞得载入：   

```python

saver.restore(sess, FLAGS.train_dir)
```  
当然了，模型的估计就类似上述了。  

这样简单的模型搭建到运行就完成了。本文主要用到这些函数：  

* `tf.placeholder`  
* `tf.Variable`  
* `tf.train`  
    * `tf.train.GradientDenscentOptimizer`  
    * `tf.train.SummaryWriter`  
    * `tf.train.Saver`  
* `tf.session`  
* `tf.Graph`  
* `tf.add_summary`  
* `tf.merge_all_summaries`  
其实构建一个模型基本就用这些函数，然后就是一些数理计算方法。详情参看[tensorflow](https://www.tensorflow.org/)  
今天下午发邮件问LSTM speaker naming的作者，结果意外加了微信，然后通了电话，问清楚了一些自己一直以来困惑的问题，大牛其实是没有架子的。还得多看论文。
