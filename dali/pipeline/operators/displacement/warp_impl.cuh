// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_IMPL_CUH_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_IMPL_CUH_

#include "dali/pipeline/operators/displacement/warp.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/imgproc/warp_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/alloc.h"
#include "dali/core/static_switch.h"

namespace dali {

template <typename Backend>
class OpImplInterface {
 public:
  virtual void Setup(kernels::TensorListShape<> &shape, Workspace<Backend> &ws) = 0;
  virtual void Run(Workspace<Backend> &ws) = 0;
};

template <typename ActualOp, typename Kernel>
class WarpOpImplGPU;

template <typename ActualOp, typename Kernel>
class WarpOpImplGPUBase : public OpImplInterface<GPUBackend> {
 public:
  using OutputType = typename Kernel::OutputType;
  using InputType = typename Kernel::InputType;
  using Mapping = typename Kernel::Mapping;
  using MappingParams = typename Kernel::MappingParams;
  using BorderValue = typename Kernel::BorderValue;
  static constexpr int spatial_ndim = Kernel::spatial_ndim;
  static constexpr int tensor_ndim = Kernel::tensor_ndim;

  WarpOpImplGPUBase(ActualOp *op) : operator_(op) {
  }

  kernels::KernelContext GetContext(DeviceWorkspace &ws) {
    kernels::KernelContext context;
    context.gpu.stream = ws.stream();
    return context;
  }

  void SetInterp(DeviceWorkspace &ws) {
    interp_types_.clear();
    if (Spec().HasTensorArgument("interp_type")) {
      int num_samples = ws.Input<GPUBackend>(0).shape().num_samples();
      auto &tensor = ws.ArgumentInput("interp_type");
      int n = tensor.shape()[0];
      DALI_ENFORCE(n == 1 || n == num_samples,
        "interp_type must be a single value or contain one value per sample");
      auto *data = tensor.template data<DALIInterpType>();
      interp_types_.resize(n);

      for (int i = 0; i < n; i++)
        interp_types_[i] = data[i];
    } else {
      interp_types_.resize(1, Spec().template GetArgument<DALIInterpType>("interp_type"));
    }

    for (size_t i = 0; i < interp_types_.size(); i++) {
      DALI_ENFORCE(interp_types_[i] == DALI_INTERP_NN || interp_types_[i] == DALI_INTERP_LINEAR,
        "Only nearest and linear interpolation is supported");

    }
  }

  virtual void SetParams(DeviceWorkspace &ws) {

  }

  virtual void SetBorder(DeviceWorkspace &ws) {
    border_ = {};
  }

  virtual kernels::TensorShape<spatial_ndim> GetScalarOutputSize() {
    std::vector<float> out_size_f;
    if (Spec().TryGetArgument(out_size_f, "output_size")) {
      DALI_ENFORCE(static_cast<int>(out_size_f.size()) == spatial_ndim,
        "output_size must specify same number of dimensions as the input (excluding channels)");
      kernels::TensorShape<spatial_ndim> out_size;
      for (int d = 0; d < spatial_ndim; d++) {
        float s = out_size_f[d];
        DALI_ENFORCE(s > 0, "Output size must be positive");
        out_size[d] = std::max<int>(std::roundf(s), 1);
      }
      return out_size;
    }
    else
      return {};
  }

  virtual void SetOutputSizes(DeviceWorkspace &ws) {
    const int N = input_.num_samples();
    output_sizes_.resize(N);
    auto size = GetScalarOutputSize();
    if (size == kernels::TensorShape<spatial_ndim>()) {
      for (int i = 0; i < N; i++) {
        output_sizes_[i] = input_.shape[i].template first<spatial_ndim>();
      }
    }
  }

  void Setup(kernels::TensorListShape<> &shape, DeviceWorkspace &ws) override {
    input_ = view<const InputType,  tensor_ndim>(ws.Input<GPUBackend>(0));
    kmgr_.Resize<Kernel>(1, 1);

    SetParams(ws);
    SetInterp(ws);
    SetBorder(ws);
    SetOutputSizes(ws);

    auto context = GetContext(ws);

    auto &req = kmgr_.Setup<Kernel>(
        0, context,
        input_,
        params_gpu_,
        make_span(output_sizes_),
        make_span(interp_types_),
        border_);

    shape = req.output_shapes[0];
  }

  void Run(DeviceWorkspace &ws) override {
    auto output = view<OutputType, tensor_ndim>(ws.Output<GPUBackend>(0));
    input_ = view<const InputType,  tensor_ndim>(ws.Input<GPUBackend>(0));
    auto context = GetContext(ws);
    kmgr_.Run<Kernel>(
        0, 0, context,
        output,
        input_,
        params_gpu_,
        make_span(output_sizes_),
        make_span(interp_types_),
        border_);
  }

  ActualOp &Op() const { return *operator_; }
  const OpSpec &Spec() const { return Op().Spec(); }

 protected:
  kernels::TensorListView<kernels::StorageGPU, const InputType, tensor_ndim> input_;
  kernels::TensorListView<kernels::StorageGPU, OutputType, tensor_ndim> output_;

  kernels::TensorView<kernels::StorageGPU, MappingParams, 1> params_gpu_;
  kernels::TensorView<kernels::StorageCPU, MappingParams, 1> params_cpu_;


  kernels::TensorListShape<tensor_ndim> input_shape_;
  std::vector<kernels::TensorShape<spatial_ndim>> output_sizes_;
  BorderValue border_ = {};
  std::vector<DALIInterpType> interp_types_;

  kernels::KernelManager kmgr_;
  ActualOp *operator_;
};

template <typename ActualOp, typename Kernel>
class WarpOpImplGPU : public WarpOpImplGPUBase<ActualOp, Kernel> {
 public:
  WarpOpImplGPU(ActualOp *op) : WarpOpImplGPUBase<ActualOp, Kernel>(op) {}
};

template <typename Derived>
class Warp<GPUBackend, Derived> : public Operator<GPUBackend> {
 public:
  using MyType = Derived;
  MyType &This() { return static_cast<MyType&>(*this); }
  const MyType &This() const { return static_cast<const MyType&>(*this); }
  using Backend = GPUBackend;
  using Workspace = DeviceWorkspace;

  const OpSpec &Spec() const { return spec_; }

 private:
  /// @defgroup WarpStaticType Dynamic to static type routing

  /// @addtogroup WarpStaticType
  /// @{

  template <typename F>
  void ToStaticTypeEx(std::tuple<> &&, F &&functor) {
    DALI_FAIL("Unsupported input/output types for the operator");
  }

  template <typename F, typename FirstTypePair, typename... TypePairs>
  void ToStaticTypeEx(std::tuple<FirstTypePair, TypePairs...> &&, F &&functor) {
    if (type2id<typename FirstTypePair::first_type>::value == output_type_ &&
        type2id<typename FirstTypePair::first_type>::value == input_type_)
      functor(FirstTypePair());
    else
      ToStaticTypeEx(std::tuple<TypePairs...>(), std::forward<F>(functor));
  }

  template <typename F>
  void ToStaticType(F &&functor) {
    using supported_types = typename MyType::SupportedTypes;
    ToStaticTypeEx(supported_types(), std::forward<F>(functor));
  }

  // TODO(michalz): Change value switch over SpatialDim to (2, 3) when kernel is implemented
  #define WARP_STATIC_TYPES(...) {                              \
    VALUE_SWITCH(This().SpatialDim(ws), spatial_ndim, (2), (    \
          ToStaticType(                                         \
            [&](auto &&args) {                                  \
            using OutputType = decltype(args.first);            \
            using InputType = decltype(args.second);            \
            __VA_ARGS__                                         \
          }); ), (                                              \
          DALI_FAIL("Only 2D and 3D warping is supported")));   \
  }

  /// @}
 public:
  using Operator<GPUBackend>::Operator;

  int SpatialDim(Workspace &ws) const {
    return input_shape_.sample_dim()-1;
  }

  bool InferOutputs(
      std::vector<kernels::TensorListShape<>> &shapes,
      std::vector<TypeInfo> &types, DeviceWorkspace &ws) {
    shapes.resize(1);
    types.resize(1);

    DALIDataType out_type;
    Setup(shapes[0], out_type, ws);
    types[0] = TypeTable::GetTypeInfo(out_type);
    return true;
  }

  using DefaultSupportedTypes = UnzipPairs<
    uint8_t, uint8_t,
    uint8_t, float,
    float,   uint8_t,

    int16_t, int16_t,
    int16_t, float,
    float,   int16_t,

    int32_t, int32_t,
    int32_t, float,
    float,   int32_t,

    float,   float
  >;

  /// @brief May be shadowed by Derived, if necessary
  using SupportedTypes = DefaultSupportedTypes;

  void Setup(kernels::TensorListShape<> &out_shape,
             DALIDataType &out_type,
             DeviceWorkspace &ws) {
    auto &input = ws.Input<GPUBackend>(0);
    input_shape_ = input.shape();
    DALIDataType new_input_type = input.type().id();
    DALIDataType new_output_type;
    if (!this->spec_.TryGetArgument<DALIDataType>(new_output_type, "output_type"))
      output_type_ = input_type_;
    out_type = new_output_type;

    output_type_ = new_output_type;
    input_type_ = new_input_type;

    WARP_STATIC_TYPES(
     using Kernel =
        typename MyType::template KernelType<spatial_ndim, OutputType, InputType>;

      using ImplType = WarpOpImplGPU<Derived, Kernel>;
      if (!dynamic_cast<ImplType*>(impl_.get()))
        impl_.reset(new ImplType(&This()));
    );

    impl_->Setup(out_shape, ws);
  }

  void RunImpl(DeviceWorkspace* ws) {
    std::vector<kernels::TensorListShape<>> shapes;
    std::vector<TypeInfo> types;
    InferOutputs(shapes, types, *ws);
    auto &output = ws->Output<GPUBackend>(0);
    output.Resize(shapes[0]);
    output.set_type(types[0]);

    assert(impl_);

    impl_->Run(*ws);
  }

 protected:
  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;
  kernels::TensorListShape<> input_shape_;
  std::unique_ptr<OpImplInterface<GPUBackend>> impl_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_IMPL_CUH_
