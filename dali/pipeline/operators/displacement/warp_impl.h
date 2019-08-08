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

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_IMPL_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_IMPL_H_

#include <cassert>
#include <vector>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "dali/pipeline/operators/displacement/warp.h"
#include "dali/pipeline/operators/displacement/warp_param_provider.h"
#include "dali/pipeline/data/views.h"
#include "dali/kernels/imgproc/warp_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/alloc.h"
#include "dali/core/static_switch.h"

namespace dali {

template <typename Backend>
class OpImplInterface {
 public:
  virtual void Setup(kernels::TensorListShape<> &shape, const workspace_t<Backend> &ws) = 0;
  virtual void Run(workspace_t<Backend> &ws) = 0;
};

template <typename Kernel>
class WarpOpImp;

template <typename Backend, typename Kernel>
class WarpOpImpl : public OpImplInterface<Backend> {
 public:
  using Storage = detail::storage_tag_map_t<Backend>;

  using OutputType = typename Kernel::OutputType;
  using InputType = typename Kernel::InputType;
  using Mapping = typename Kernel::Mapping;
  using MappingParams = typename Kernel::MappingParams;
  using BorderType = typename Kernel::BorderType;
  static constexpr int spatial_ndim = Kernel::spatial_ndim;
  static constexpr int tensor_ndim = Kernel::tensor_ndim;
  using ParamProvider = WarpParamProvider<Backend, spatial_ndim, MappingParams, BorderType>;
  using Workspace = workspace_t<Backend>;

  WarpOpImpl(const OpSpec &spec, std::unique_ptr<ParamProvider> &&pp)
  : spec_(spec), param_provider_(std::move(pp)) {
  }

  kernels::KernelContext GetContext(const Workspace &ws) {
    kernels::KernelContext context;
    context.gpu.stream = ws.has_stream() ? ws.stream() : 0;
    return context;
  }

  template <typename WorkspaceType>
  enable_if_t<std::is_same<DeviceWorkspace, WorkspaceType>::value, void>
  SetupBackend(kernels::TensorListShape<> &shape, const WorkspaceType &ws) {
    auto context = GetContext(ws);
    auto &req = kmgr_.Setup<Kernel>(
        0, context,
        input_,
        param_provider_->ParamsGPU(),
        param_provider_->OutputSizes(),
        param_provider_->InterpTypes(),
        param_provider_->Border());
    shape = req.output_shapes[0];
  }

  template <typename WorkspaceType>
  enable_if_t<std::is_same<HostWorkspace, WorkspaceType>::value, void>
  CallBackend(kernels::TensorListShape<> &shape, const WorkspaceType &ws) {
    auto context = GetContext(ws);
    auto &req = kmgr_.Setup<Kernel>(
        0, context,
        input_,
        param_provider_->ParamsCPU(),
        param_provider_->OutputSizes(),
        param_provider_->InterpTypes(),
        param_provider_->Border());
    shape = req.output_shapes[0];
  }

  void Setup(kernels::TensorListShape<> &shape, const Workspace &ws) override {
    param_provider_->SetContext(Spec(), ws);

    input_ = view<const InputType, tensor_ndim>(ws.template Input<Backend>(0));
    kmgr_.Resize<Kernel>(1, 1);

    param_provider_->Setup();

    SetupBackend(shape, ws);
  }

  template <typename WorkspaceType>
  std::enable_if_t<std::is_same<WorkspaceType, HostWorkspace>::value, void>
  RunBackend(WorkspaceType &_ws) override {
    DeviceWorkspace &ws = _ws;
    param_provider_->SetContext(Spec(), ws);

    auto output = view<OutputType, tensor_ndim>(ws.template Output<Backend>(0));
    input_ = view<const InputType,  tensor_ndim>(ws.template Input<Backend>(0));
    auto context = GetContext(ws);
    kmgr_.Run<Kernel>(
        0, 0, context,
        output,
        input_,
        param_provider_->ParamsGPU(),
        param_provider_->OutputSizes(),
        param_provider_->InterpTypes(),
        param_provider_->Border());
  }


  template <typename WorkspaceType>
  std::enable_if_t<std::is_same<WorkspaceType, DeviceWorkspace>::value, void>
  RunBackend(WorkspaceType &_ws) override {
    DeviceWorkspace &ws = _ws;
    param_provider_->SetContext(Spec(), ws);

    auto output = view<OutputType, tensor_ndim>(ws.template Output<Backend>(0));
    input_ = view<const InputType,  tensor_ndim>(ws.template Input<Backend>(0));
    auto context = GetContext(ws);
    kmgr_.Run<Kernel>(
        0, 0, context,
        output,
        input_,
        param_provider_->ParamsGPU(),
        param_provider_->OutputSizes(),
        param_provider_->InterpTypes(),
        param_provider_->Border());
  }

  void Run(Workspace &ws) override {
    RunBackend(ws);
  }

  const OpSpec &Spec() const { return spec_; }

 protected:
  const OpSpec &spec_;
  kernels::KernelManager kmgr_;

  kernels::TensorListView<Storage, const InputType, tensor_ndim> input_;
  kernels::TensorListView<Storage, OutputType, tensor_ndim> output_;

  std::unique_ptr<ParamProvider> param_provider_;
};


template <typename Backend, typename Derived>
class Warp : public Operator<Backend> {
 public:
  using MyType = Derived;
  MyType &This() { return static_cast<MyType&>(*this); }
  const MyType &This() const { return static_cast<const MyType&>(*this); }
  using Workspace = workspace_t<Backend>;

  const OpSpec &Spec() const { return this->spec_; }

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
  #define WARP_STATIC_TYPES(...) {                                          \
    VALUE_SWITCH(This().SpatialDim(), spatial_ndim, (2), (                  \
      VALUE_SWITCH(This().BorderClamp() ? 1 : 0, UseBorderClamp, (0, 1), (  \
          ToStaticType(                                                     \
            [&](auto &&args) {                                              \
            using OutputType = decltype(args.first);                        \
            using InputType = decltype(args.second);                        \
            using BorderType = std::conditional_t<                          \
              UseBorderClamp, kernels::BorderClamp, OutputType>;            \
            __VA_ARGS__                                                     \
          });),                                                             \
          (assert(!"impossible")))),                                        \
        (DALI_FAIL("Only 2D and 3D warping is supported")));                \
  }

  /// @}
 public:
  using Operator<Backend>::Operator;

  int SpatialDim() const {
    return input_shape_.sample_dim()-1;
  }

  bool BorderClamp() const {
    return !Spec().HasArgument("border");
  }

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &outputs, const DeviceWorkspace &ws) override {
    outputs.resize(1);

    DALIDataType out_type;
    SetupWarp(outputs[0].shape, out_type, ws);
    outputs[0].type = TypeTable::GetTypeInfo(out_type);
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

  void SetupWarp(kernels::TensorListShape<> &out_shape,
                 DALIDataType &out_type,
                 const DeviceWorkspace &ws) {
    auto &input = ws.template Input<Backend>(0);
    input_shape_ = input.shape();
    DALIDataType new_input_type = input.type().id();
    DALIDataType new_output_type;
    if (!Spec().TryGetArgument(new_output_type, "output_type"))
      new_output_type = new_input_type;

    output_type_ = new_output_type;
    input_type_ = new_input_type;

    WARP_STATIC_TYPES(
      using Kernel =
        typename MyType::template KernelType<spatial_ndim, OutputType, InputType, BorderType>;

      using ImplType = WarpOpImpl<Backend, Kernel>;
      if (!dynamic_cast<ImplType*>(impl_.get())) {
        auto param_provider = This().template CreateParamProvider<spatial_ndim, BorderType>();
        impl_.reset(new ImplType(Spec(), std::move(param_provider)));
      }
    ); // NOLINT

    impl_->Setup(out_shape, ws);
    out_type = output_type_;
  }

  void RunImpl(DeviceWorkspace* ws) {
    assert(impl_);
    impl_->Run(*ws);
  }

 protected:
  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;
  kernels::TensorListShape<> input_shape_;
  std::unique_ptr<OpImplInterface<Backend>> impl_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_WARP_IMPL_H_
