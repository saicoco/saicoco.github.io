---
title: "Mxnet学习笔记(4)--operator源码之基类Operator"
layout: page
date: 2017-02-16
tag: mxnet
blog: true
author: karl
category: mxnet
description: 自定义Op
---  

过完年了，新的开始，计划给自己分配点任务:计划将mxnet中的c++源码读一遍，全当学习一遍c++。大致分为以下几个模块
进行阅读：　　

1. operator  
2. optimazer  
3. ndarray  
4. kvstore  
5. executor  

因为在使用python API时感觉没底，最近又遇到自定义op时的打击，于是计划开启学习，目的旨在学习c++和
mxnet中的源码设计，以此来督促自己。其中operator是重点，因为我们自定义层时主要集中在这里。希望可以顺利
进行下去。今天先讲基类Operator.  

## operator.h  

在头文件operator.h中，主要定义了operator的一些方法还有基类operator.这里我们主要学习基类operator,其余方法参看源码文件
operator.h。先贴源码框架　　


```c++
class Operator{
    public:
    virtual ~Operator(){}
    virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) = 0;
    virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    LOG(FATAL) << "Backward is not implemented";
    }
}

class OperatorProperty{
    public:
    virtual ~OperatorProperty() {}
    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) = 0;
    virtual std::map<std::string, std::string> GetParams() const = 0;
    virtual std::map<std::string, std::string> GetParams() const = 0;

    virtual std::vector<std::string> ListArguments() const {
        return {"data"};
    }

    virtual std::vector<std::string> ListOutputs() const {
        return {"output"};
    }

    virtual std::vector<std::string> ListAuxiliaryStates() const {
        return {};
    }

    virtual int NumOutputs() const {
        return this->ListOutputs().size();
    }

    virtual int NumVisibleOutputs() const {
        return NumOutputs();
    }

    virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          std::vector<TShape> *aux_shape) const = 0;
    
    virtual bool InferType(std::vector<int> *in_type,
                          std::vector<int> *out_type,
                          std::vector<int> *aux_type) const {...}
    virtual OperatorProperty* Copy() const = 0;
    virtual Operator* CreateOperator(Context ctx) const = 0;
    virtual Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const{
                                         return CreateOperator(ctx);
                                     }
    
    ...

    static OperatorProperty *Create(const char* type_name);

    #define MXNET_REGISTER_OP_PROPERTY(name, OperatorPropertyType)          \
    DMLC_REGISTRY_REGISTER(::mxnet::OperatorPropertyReg, OperatorPropertyReg, name) \
    .set_body([]() { return new OperatorPropertyType(); })                \
    .set_return_type("Symbol") \
    .check_name()
                          
}
```  

由上述代码，可以发现，python operator API与此类似，在上篇文章我们已经说到，在类Operator中定义forward()与backward()方法，
在OperatorProperty类中定义参数接口，输出参数名字，如方法ListXXX(),在其中定义参数名字(如Arguments中定义输入参数，Outputs中
定义输出参数名字，AuxiliaryStates中定义辅助变量的名字，而这些在网络运行时可以利用名字或者顺序index获取数值)。同样的，利用Inferhape获取输入和输出的
shape。在最后成员方法CreateOperatorEx中可以看到调用了成员方法CreateOperator构建operator,然后利用工厂函数Create()利用名字创建
operator,然后在最后的注册函数中对operator进行注册，完成对operator的定义。