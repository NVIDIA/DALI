// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_REMAP_WARP_H_
#define DALI_OPERATORS_IMAGE_REMAP_WARP_H_

#include <cassert>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/core/tuple_helpers.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/imgproc/warp_cpu.h"
#include "dali/kernels/imgproc/warp_gpu.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/image/remap/warp_param_provider.h"
#include "dali/pipeline/data/views.h"

namespace dali {
namespace detail {

template <typename... Pairs>
struct UnzipPairsHelper;

template <typename... Pairs>
using UnzipPairs = typename detail::UnzipPairsHelper<Pairs...>::type;

template <>
struct UnzipPairsHelper<> {
  using type = std::tuple<>;
};

template <typename T1, typename T2, typename... Tail>
struct UnzipPairsHelper<T1, T2, Tail...> {
  static_assert(sizeof...(Tail) % 2 == 0, "Number of types for unzip must be even");
  using type = detail::tuple_cat_t<std::tuple<std::pair<T1, T2>>, UnzipPairs<Tail...>>;
};

}  // namespace detail
using detail::UnzipPairs;

template <typename Backend>
class OpImplInterface {
 public:
  virtual void Setup(TensorListShape<> &shape, const workspace_t<Backend> &ws) = 0;
  virtual void Run(workspace_t<Backend> &ws) = 0;
  virtual ~OpImplInterface() = default;
};

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

  WarpOpImpl(const OpSpec &spec, std::unique_ptr<ParamProvider> pp)
  : spec_(spec), param_provider_(std::move(pp)) {
  }

  void Setup(TensorListShape<> &shape, const Workspace &ws) override {
    param_provider_->SetContext(Spec(), ws);

    input_ = view<const InputType, tensor_ndim>(ws.template InputRef<Backend>(0));
    param_provider_->Setup();

    SetupBackend(shape, ws);
  }

  void Run(Workspace &ws) override {
    RunBackend(ws);
  }

  const OpSpec &Spec() const { return spec_; }

 private:
  const OpSpec &spec_;
  kernels::KernelManager kmgr_;

  TensorListView<Storage, const InputType, tensor_ndim> input_;

  std::unique_ptr<ParamProvider> param_provider_;

  kernels::KernelContext GetContext(const Workspace &ws) {
    kernels::KernelContext context;
    context.gpu.stream = ws.has_stream() ? ws.stream() : 0;
    return context;
  }

  void SetupBackend(TensorListShape<> &shape, const DeviceWorkspace &ws) {
    auto context = GetContext(ws);
    kmgr_.Resize<Kernel>(1, 1);
    auto &req = kmgr_.Setup<Kernel>(
        0, context,
        input_,
        param_provider_->ParamsGPU(),
        param_provider_->OutputSizes(),
        param_provider_->InterpTypes(),
        param_provider_->Border());
    shape = req.output_shapes[0];
  }

  void SetupBackend(TensorListShape<> &shape, const HostWorkspace &ws) {
    int threads = ws.HasThreadPool() ? ws.GetThreadPool().NumThreads() : 1;
    int N = input_.num_samples();
    kmgr_.Resize<Kernel>(threads, N);

    shape.resize(N, input_.sample_dim());
    auto interp_types = param_provider_->InterpTypes();

    auto context = GetContext(ws);
    for (int i = 0; i < N; i++) {
      DALIInterpType interp_type = interp_types.size() > 1 ? interp_types[i] : interp_types[0];
      auto &req = kmgr_.Setup<Kernel>(
          i, context,
          input_[i],
          *param_provider_->ParamsCPU()(i),
          param_provider_->OutputSizes()[i],
          interp_type,
          param_provider_->Border());
      shape.set_tensor_shape(i, req.output_shapes[0][0]);
    }
  }


  void RunBackend(HostWorkspace &ws) {
    param_provider_->SetContext(Spec(), ws);

    auto output = view<OutputType, tensor_ndim>(ws.template OutputRef<Backend>(0));
    input_ = view<const InputType,  tensor_ndim>(ws.template InputRef<Backend>(0));

    ThreadPool &pool = ws.GetThreadPool();
    auto interp_types = param_provider_->InterpTypes();

    for (int i = 0; i < input_.num_samples(); i++) {
      pool.AddWork([&, i](int tid) {
        DALIInterpType interp_type = interp_types.size() > 1 ? interp_types[i] : interp_types[0];
        auto context = GetContext(ws);
        kmgr_.Run<Kernel>(
            tid, i, context,
            output[i],
            input_[i],
            *param_provider_->ParamsCPU()(i),
            param_provider_->OutputSizes()[i],
            interp_type,
            param_provider_->Border());
      }, output.shape.tensor_size(i));
    }
    pool.RunAll();
  }

  void RunBackend(DeviceWorkspace &ws) {
    param_provider_->SetContext(Spec(), ws);

    auto output = view<OutputType, tensor_ndim>(ws.template OutputRef<Backend>(0));
    input_ = view<const InputType,  tensor_ndim>(ws.template InputRef<Backend>(0));
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
};

template <typename Backend,
          typename Mapping, int spatial_ndim,
          typename OutputType, typename InputType, typename BorderType>
struct WarpKernelSelector;

template <typename Mapping, int spatial_ndim,
          typename OutputType, typename InputType, typename BorderType>
struct WarpKernelSelector<GPUBackend, Mapping, spatial_ndim, OutputType, InputType, BorderType> {
  using type = kernels::WarpGPU<Mapping, spatial_ndim, OutputType, InputType, BorderType>;
};

template <typename Mapping, int spatial_ndim,
          typename OutputType, typename InputType, typename BorderType>
struct WarpKernelSelector<CPUBackend, Mapping, spatial_ndim, OutputType, InputType, BorderType> {
  using type = kernels::WarpCPU<Mapping, spatial_ndim, OutputType, InputType, BorderType>;
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
    auto &in_name = TypeTable::GetTypeInfo(input_type_).name();
    auto &out_name = TypeTable::GetTypeInfo(output_type_).name();
    DALI_FAIL("Unsupported input/output types for the operator: " + in_name + " -> " + out_name);
  }

  template <typename F, typename FirstTypePair, typename... TypePairs>
  void ToStaticTypeEx(std::tuple<FirstTypePair, TypePairs...> &&, F &&functor) {
    if (type2id<typename FirstTypePair::first_type>::value == output_type_ &&
        type2id<typename FirstTypePair::second_type>::value == input_type_)
      functor(FirstTypePair());
    else
      ToStaticTypeEx(std::tuple<TypePairs...>(), std::forward<F>(functor));
  }

  template <typename F>
  void ToStaticType(F &&functor) {
    using supported_types = typename MyType::SupportedTypes;
    ToStaticTypeEx(supported_types(), std::forward<F>(functor));
  }

  /// @}
 public:
  explicit Warp(const OpSpec &spec) : Operator<Backend>(spec) {
    border_clamp_ = !spec.HasArgument("fill_value");
    spec.TryGetArgument(output_type_arg_, "dtype");
  }

  int SpatialDim() const {
    return input_shape_.sample_dim()-1;
  }

  bool BorderClamp() const {
    return border_clamp_;
  }

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &outputs, const Workspace &ws) override {
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

  /** @brief May be shadowed by Derived, if necessary */
  using SupportedTypes = DefaultSupportedTypes;

  void SetupWarp(TensorListShape<> &out_shape,
                 DALIDataType &out_type,
                 const Workspace &ws) {
    auto &input = ws.template InputRef<Backend>(0);
    input_shape_ = input.shape();
    input_type_ = input.type().id();
    output_type_ = output_type_arg_ == DALI_NO_TYPE ? input_type_ : output_type_arg_;

    VALUE_SWITCH(This().SpatialDim(), spatial_ndim, (2, 3), (
      BOOL_SWITCH(This().BorderClamp(), UseBorderClamp, (
          ToStaticType(
            [&](auto &&args) {
            using OutputType = decltype(args.first);
            using InputType = decltype(args.second);
            using BorderType = std::conditional_t<
              UseBorderClamp, kernels::BorderClamp, OutputType>;

            using Mapping = typename MyType::template Mapping<spatial_ndim>;
            using Kernel = typename WarpKernelSelector<
                Backend, Mapping, spatial_ndim, OutputType, InputType, BorderType>::type;

            using ImplType = WarpOpImpl<Backend, Kernel>;
            if (!dynamic_cast<ImplType*>(impl_.get())) {
              auto param_provider = This().template CreateParamProvider<spatial_ndim, BorderType>();
              impl_.reset(new ImplType(Spec(), std::move(param_provider)));
            }
          });))),
        (DALI_FAIL("Only 2D and 3D warping is supported")));


    impl_->Setup(out_shape, ws);
    out_type = output_type_;
  }

  void RunImpl(Workspace &ws) override {
    assert(impl_);
    impl_->Run(ws);
    auto &out = ws.template OutputRef<Backend>(0);
    auto &in = ws.template InputRef<Backend>(0);
    out.SetLayout(in.GetLayout());
  }

 protected:
  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;
  DALIDataType output_type_arg_ = DALI_NO_TYPE;
  TensorListShape<> input_shape_;
  std::unique_ptr<OpImplInterface<Backend>> impl_;
  bool border_clamp_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_WARP_H_
