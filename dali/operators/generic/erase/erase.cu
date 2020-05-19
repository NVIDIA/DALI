// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/erase/erase_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/generic/erase/erase.h"
#include "dali/operators/generic/erase/erase_utils.h"
#include "dali/pipeline/data/views.h"

#define ERASE_SUPPORTED_NDIMS (1, 2, 3, 4, 5)
#define ERASE_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                               uint64_t, int64_t, float)
namespace dali {

namespace detail {

template <int ndim>
ivec<ndim> to_ivec(const TensorShape<ndim> &shape) {
  ivec<ndim> result;
  for (int d = 0; d < ndim; d++) {
    result[d] = shape[d];
  }
  return result;
}

template <int ndim>
kernels::ibox<ndim> make_box(const TensorShape<ndim> &lcorner, const TensorShape<ndim> &shape) {
  auto lc = to_ivec(lcorner);
  return Box<ndim, int>(lc, lc + to_ivec(shape));
}

}  // namespace detail

template <typename T, int Dims, int channel_dim>
class EraseImplGpu : public OpImplBase<GPUBackend> {
 public:
  using EraseKernel = kernels::EraseGpu<T, Dims, channel_dim>;

  explicit EraseImplGpu(const OpSpec &spec)
      : spec_(spec), batch_size_(spec.GetArgument<int>("batch_size")) {
        kmgr_.Resize<EraseKernel>(1, 1);
      }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<GPUBackend> &ws) override {
    const auto &input = ws.template InputRef<GPUBackend>(0);
    auto layout = input.GetLayout();
    auto type = input.type();
    auto shape = input.shape();
    AcquireArgs(ws, shape, layout);

    auto in_view = view<const T, Dims>(input);
    kmgr_.Initialize<EraseKernel>();
    ctx_.gpu.stream = ws.stream();
    kmgr_.Setup<EraseKernel>(0, ctx_, in_view, regions_, make_cspan(fill_values_));

    output_desc.resize(1);
    output_desc[0] = {shape, input.type()};
    return true;
  };

  void RunImpl(workspace_t<GPUBackend> &ws) override {
    auto input = view<const T, Dims>(ws.template InputRef<GPUBackend>(0));
    auto output = view<T, Dims>(ws.template OutputRef<GPUBackend>(0));
    kmgr_.Run<EraseKernel>(0, 0, ctx_, output, input, regions_, make_cspan(fill_values_));
  }

  void AcquireArgs(const ArgumentWorkspace &ws, TensorListShape<> in_shape,
                   TensorLayout in_layout) {
    fill_values_ = spec_.template GetRepeatedArgument<float>("fill_value");
    DALI_ENFORCE(channel_dim >= 0 || fill_values_.size() <= 1,
      "If a multi channel fill value is provided, the input layout must have a 'C' dimension");
    regions_.resize(batch_size_);
    auto args = detail::GetEraseArgs<T, Dims>(spec_, ws, in_shape, in_layout);
    for (int i = 0; i < batch_size_; ++i) {
      regions_[i].reserve(args[i].rois.size());
      const auto &rois = args[i].rois;
      for (const auto &roi : args[i].rois) {
        regions_[i].push_back(detail::make_box(roi.anchor, roi.shape));
      }
    }
  }

 private:
  OpSpec spec_;
  int batch_size_ = 0;
  std::vector<int> axes_;
  std::vector<kernels::EraseArgs<T, Dims>> args_;
  std::vector<std::vector<kernels::ibox<Dims>>> regions_;
  SmallVector<T, 3> fill_values_;
  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
};

namespace detail {

template <typename T, int Dims, int... ChDims>
struct select_impl {};

template <typename T, int Dims, int ChDim, int... ChDims>
struct select_impl<T, Dims, ChDim, ChDims...> {
  static std::unique_ptr<OpImplBase<GPUBackend>> make(int channel_dim, const OpSpec &spec) {
    if (channel_dim == ChDim) {
      return std::make_unique<EraseImplGpu<T, Dims, ChDim>>(spec);
    } else {
      return select_impl<T, Dims, ChDims...>::make(channel_dim, spec);
    }
  }
};

template <typename T, int Dims, int ChDim>
struct select_impl<T, Dims, ChDim> {
  static std::unique_ptr<OpImplBase<GPUBackend>> make(int channel_dim, const OpSpec &spec) {
    if (channel_dim == ChDim) {
      return std::make_unique<EraseImplGpu<T, Dims, ChDim>>(spec);
    } else {
      DALI_FAIL("Unsopported layout. Only channel-first and channel-last layouts are supported.");
    }
  }
};

}  // namespace detail

template <>
bool Erase<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                  const workspace_t<GPUBackend> &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  auto in_shape = input.shape();
  auto channel_dim = input.GetLayout().find('C');
  TYPE_SWITCH(input.type().id(), type2id, T, ERASE_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, ERASE_SUPPORTED_NDIMS, (
      impl_ = detail::select_impl<T, Dims, -1, 0, Dims-1>::make(channel_dim, spec_);
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type().id())));  // NOLINT

  assert(impl_ != nullptr);
  return impl_->SetupImpl(output_desc, ws);
}

template <>
void Erase<GPUBackend>::RunImpl(workspace_t<GPUBackend> &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(Erase, Erase<GPUBackend>, GPU);

}  // namespace dali
