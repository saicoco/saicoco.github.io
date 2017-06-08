---
title: "Mxnet学习笔记(5)--operator源码之Activation"
layout: post
date: 2017-02-17
tag: mxnet
blog: true
author: karl
category: mxnet
description: 自定义Op
---  

上篇说道operator.h，接着我们说NN中最为常见的操作Activation，就着激活函数我们分析一个operator的
构建需要的文件结构，涉及到的一些细节，我们这里讲讲。　　

## 文件结构　　

打开mxnet下源码文件夹mxnet/src/operator,我们可以看到里面有很多我们常见的深度学习中的操作，大到
损失函数，小到padding操作，其文件结构齐刷刷的都是三个文件(原谅我是个c++菜鸟)以activation为例：　　

* activation-inl.h  
* activation.cc  
* activation.cu  

第三个文件不用说了,这是gpu版本的activation,我们这里不做分析，仅对前两个文件进行分析，在文件
activation-inl.h中，主要定义类ActivationOp以及具体实现，在activation.cc中则是根据不同激活函数
对类进行实例化。接下来我们一个一个的来说。　　

## activation-inl.h    

```c++
#ifndef MXNET_OPERATOR_ACTIVATION_INL_H_
#define MXNET_OPERATOR_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace activation {
enum ActivationOpInputs {kData};
enum ActivationOpOutputs {kOut};
enum ActivationOpType {kReLU, kSigmoid, kTanh, kSoftReLU};
}  // activation

struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .describe("Activation function to be applied.");
  }
};
```  

在文件最开始，定义了激活函数的枚举类型:输入kData, 输出kOut,激活函数类型kReLU, kSigmoid，kTanh等用于后续对
输入输出的管理。接下来是结构体`ActivationParam`,这里需要将模板编程看看。这里调用`dmlc::Parameter`,我们看看
它是什么东西：　　

```c++
template<typename PType>
struct Parameter {
 public:
  /*!
   * \brief initialize the parameter by keyword arguments.
   *  This function will initialize the parameter struct, check consistency
   *  and throw error if something wrong happens.
   *
   * \param kwargs map of keyword arguments, or vector of pairs
   * \parma option The option on initialization.
   * \tparam Container container type
   * \throw ParamError when something go wrong.
   */
  template<typename Container>
  inline void Init(const Container &kwargs,
                   parameter::ParamInitOption option = parameter::kAllowHidden) {
    PType::__MANAGER__()->RunInit(static_cast<PType*>(this),
                                  kwargs.begin(), kwargs.end(),
                                  NULL,
                                  option);
  }
```  

即调用RunInit函数对激活函数进行初始化。　　

接下来是正常的继承基类Operator，首先是类ActivationOp.  

```c++
template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
//xpu指cpu或gpu,　ForwardOp和BackwardOp在mshadow_op中，待会描述，DType是数据类型
class ActivationOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[activation::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[activation::kOut].FlatTo2D<xpu, DType>(s);
    Assign(out, req[activation::kOut], F<ForwardOp>(data));
  }
/**
* F<Op>(data): 这里Op来自类初始化时mashow::Op, 以此来完成链式求导
*/
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> m_out_grad = out_grad[activation::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_out_data = out_data[activation::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_in_grad = in_grad[activation::kData].FlatTo2D<xpu, DType>(s);
    Assign(m_in_grad, req[activation::kData], F<BackwardOp>(m_out_data) * m_out_grad);
  }
};  // class ActivationOp
```


对于Forward方法的形参，可以参看文档[operator](http://mxnet.io/doxygen/classmxnet_1_1Operator.html):  

* ctx: 表示运行与cpu或者gpu  
* out_grad: 表示反向传播是上层(从loss到当前层)传来的gradient　　
* in_data: 表示输入的数据，即我们输入的数据　　
* out_data: 表示in_data经过operator得到的数据　　
* req: 表示该operator对写入时的选择:write, inplace, null三种选择　　
* in_grad: 表示我们需要求出的导数，即我们最后求出的grad  
* aux_states: 表示辅助变量的状态，声明在list_aux...()方法中　　

知道上述参数表示什么意思后，我们对代码进行分析，对于Forward()方法，data，out分别表示输入输出数据，Assign表示将`F<ForwardOp>(data)`
的结果输出到out中，这里`F<ForwardOp>(data)`干了些什么：　　



```c++
op = new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad, DType>();
```  

以上是实例化为relu时的参数情况,可以看到ForwardOp，BackwardOp都来自mashadow_op中，而在mashadow_op中，则是具体的函数：　　

```c++
struct relu {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a > DType(0.0f) ? a : DType(0.0f));
  }
};
struct relu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(a > DType(0.0f) ? DType(1.0f) : DType(0.0f));
  }
};
```  

而函数`F<ForwardOp>(data)`如下所示：　　

```c++
template<typename OP, typename TA, typename TB, typename DType, int etype>
struct BinaryMapExp: public Exp<BinaryMapExp<OP, TA, TB, DType, etype>,
                                DType, etype> {
  /*! \brief left operand */
  const TA &lhs_;
  /*! \brief right operand */
  const TB &rhs_;
  /*! \brief constructor */
  explicit BinaryMapExp(const TA &lhs, const TB &rhs)
      :lhs_(lhs), rhs_(rhs) {}
};

/*! \brief make expression */
template<typename OP, typename TA, typename TB, typename DType, int ta, int tb>
inline BinaryMapExp<OP, TA, TB, DType, (ta|tb|type::kMapper)>
MakeExp(const Exp<TA, DType, ta> &lhs, const Exp<TB, DType, tb> &rhs) {
  return BinaryMapExp<OP, TA, TB, DType,
                      (ta|tb|type::kMapper)>(lhs.self(), rhs.self());
}


template<typename OP>
inline BinaryMapExp<OP, ScalarExp<MSHADOW_SCALAR_>, ScalarExp<MSHADOW_SCALAR_>,
                    MSHADOW_SCALAR_, (1|type::kMapper)>
F(const ScalarExp<MSHADOW_SCALAR_> &lhs, const ScalarExp<MSHADOW_SCALAR_> &rhs) {
  return MakeExp<OP>(lhs, rhs);
}
```  

可以看到，方法F中调用MakeExp方法，即实现ForwardOp(data)（这里我猜的，模板变成还需看看）。经过上述一系列过程完成
方法Forward()。同理，对于方法Backward()，我们需要利用chain rule获得in_grad,即in_grad = curr_grad * out_grad.如
`F<BackwardOp>(m_out_data) * m_out_grad`。　　

在定义ActivationOp之后，我们需要对ActivationProp进行定义，同样的继承与类OperatorProperty.  

```c++
template<typename xpu>
Operator* CreateOp(ActivationParam type, int dtype);

#if DMLC_USE_CXX11
class ActivationProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(activation::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
          (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Activation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[activation::kOut], out_data[activation::kOut], in_data[activation::kData]};
#else
    return {out_grad[activation::kOut], out_data[activation::kOut]};
#endif  // MXNET_USE_CUDNN
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[activation::kOut], in_grad[activation::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[activation::kData], out_data[activation::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ActivationParam param_;
};
```  

在进行类定义之前，首先声明工厂方法CreateOp(),用于在activation.cc中进行op的构建。接下来进行各个函数的继承，没什么要说的，重点在forward与backward方法的写法。接下来我们看看activation.cc文件中
做了些什么。　　

## activation.cc  　　

```c++
template<>
Operator *CreateOp<cpu>(ActivationParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  if (param.act_type == activation::kReLU) {
      switch (dtype) {
      case mshadow::kFloat32:
          return new MKLReluOp<cpu, float>();
      case mshadow::kFloat64:
          return new MKLReluOp<cpu, double>();
      default:
          break;
      }
  }
  if (enableMKLWarnGenerated())
    LOG(INFO) << MKLReluOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.act_type) {
      case activation::kReLU:
      // 由次传入forwardOp, backwardOp, 此处类初始化时需对模板进行初始化
        op = new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad, DType>();
        break;
      case activation::kSigmoid:
        op = new ActivationOp<cpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>();
        break;
      case activation::kTanh:
        op = new ActivationOp<cpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>();
        break;
      case activation::kSoftReLU:
        op = new ActivationOp<cpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>();
        break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  })
  return op;
}
```  

首先便是工厂方法CreateOp的实现，这里根据库的不同分为两种，一种基于IntelMKL库，这里我们说另外一种。可以看到，
上述代码中根据act_type的不同获取不同的激活函数，而实例化的过程上面已经说过了，主要是调用F方法.这里CreateOp方法会
返回一个新的op.　　

```c++
Operator *ActivationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(ActivationParam);

MXNET_REGISTER_OP_PROPERTY(Activation, ActivationProp)
.describe(R"(Elementwise activation function.

The following activation types are supported (operations are applied elementwisely to each
scalar of the input tensor):

- `relu`: Rectified Linear Unit, `y = max(x, 0)`
- `sigmoid`: `y = 1 / (1 + exp(-x))`
- `tanh`: Hyperbolic tangent, `y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
- `softrelu`: Soft ReLU, or SoftPlus, `y = log(1 + exp(x))`

See `LeakyReLU` for other activations with parameters.
)")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(ActivationParam::__FIELDS__());
```  

接下来，便是方法CreateOperatorEx的实现，该方法中将CreateOp与param_绑定，这样就可以根据不同的param_获取不同的激活函数。
最后阶段是对操作符的注册以及形参的添加。　　


### 后话　　

一开始看代码的时候一头雾水，不知道是不是VScode的原因，看代码很清晰，mxnet的代码感觉比caffe代码清晰，虽然mxnet后端有nnvm, dmlc, mshadow等库，
看起来会感觉很难，但是遇到啥看啥是个好办法，完全集中mxnet的源码上就好了。


