---
layout: post
title: 利用lmdb制作自己的数据库
tag: caffe
category: caffe
comments: true
blog: true
data: 2016-10-09
---  

### 写在前面的话　　

今天在处理语音的时候，一直out of memory，于是想着模仿caffe制作自己的lmdb，然后从中读取语音frame，于是这里记录一下制作过程。顿时发现，caffe真的很nice。过几天再把昨天挖的坑
填起来。　　

### lmdb, Protocol_Buffers  

为生成lmdb,我们需要编写.proto文件，就像caffe那样，那个我的栗子来说:  

>audio_frame.proto  

```
message Audio{
    optional int32 length = 1;
    optional int32 channels = 2;
    optional bytes data = 3;
    optional int32 label = 4;
    repeated float float_data = 6;
}
```  

由上述可以看到，proto是由一系列`message`组成的，`message`表示一个结构，其中参数类型分`required`，`optional`，`repeated`，分别表示初始化时参数
初始化要求，`required`表示必须有，没有则会抛出异常；`optional`表示选择性的初始化，`repeated`则表示该参数可以赋值多次，在数据库中排列会按序排列。
而值的类型`bool`，`int32`，`float`，`string`，当然也有枚举类型。具体可以参考[protocol buffer guide](https://developers.google.com/protocol-buffers/docs/proto)。　　

如上，我们构造我们的protocol buffer,接下来我们编译其为我们需要的文件，编译器为`protoc`，由`--help`可以看到，输出可以为`python`,`java`,`c++`,根据需要选择即可。　　

`protoc audio_frame.proto --python_out=./audio_pro`  

运行如上命令，我们在文件夹`audio_pro`中生成了我们的`audio_frame_pb2.py`,内容则主要是`Audio`类，是一个`readable buffer object`,可以利用其来保存每个
audio frame，然后将其作为`key-value`中的value保存至lmdb中。　　

通过上述`message`类的生成，我们可以利用其将对应格式的audio保存至lmdb中。直接上核心部分:  

>to_lmdb  

```python
 # operate lmdb
    env = lmdb.open(train_lmdb, 50000*1000*5)
    txn = env.begin(write=True)
    count = 0

    for i in xrange(data.shape[0]):
        audio = array2audio(data[i], labels[i])
        str_id = '{:08}'.format(count)
        txn.put(str_id, audio.SerializeToString())
        count += 1

        if count%1000 == 0:
            logging.info('already handled with {} frames'.format(count))
            txn.commit()
            txn=env.begin(write=True)
    txn.commit()
    env.close()
```  

打开数据库，获取游标位置，然后将序列化好的数据写入数据库然后移动游标，实现数据的保存。这里值得注意的是，`Protocol Buffer`类有两个方法可以
将对象进行序列化和解析：　　

* SerializeToString():序列化这个message并以字符串返回，注意这是一个二进制字节;
* ParseFromString(data):从给定的data中解析出一个message;  

详细可以参看[Message API](https://developers.google.com/protocol-buffers/docs/reference/python/google.protobuf.message.Message-class)  

以上就是DIY一个自己的lmdb数据库的建议过程，详细还需参看文档实现复杂功能。代码可以在[github]()找到　　

### 写在后面的话　　

《解忧杂货店》看完了，最近又在看东野圭吾的《白夜行》，看了不到30%，剧情扑朔迷离，亮司和雪穗肯定不是表面那样，如果说亮司一开始不断做坏事让人讨厌，倒不如说雪穗
的笑里藏刀让人胆寒，还没看到两人底细，如果有谁看到了我这篇blog,可以推荐你看看这篇小说，人心可以坏成什么样子……
