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
#include "dali/kernels/kernel_params.h"
#include "dali/operators/generic/erase/erase.h"
#include "dali/operators/generic/erase/erase_utils.h"
#include "dali/pipeline/data/views.h"

#define ERASE_SUPPORTED_NDIMS (1, 2, 3, 4, 5)
#define ERASE_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                               uint64_t, int64_t, float)
namespace dali {

namespace detail {

template <int ndim>
kernels::ibox<ndim> make_box(const TensorShape<ndim> &lcorner, const TensorShape<ndim> &shape) {
  auto lc = kernels::to_ivec(lcorner);
  return kernels::ibox<ndim>(lc, lc + kernels::to_ivec(shape));
}

}  // namespace detail

template <typename T, int Dims, int channel_dim>
class EraseImplGpu : public OpImplBase<GPUBackend> {
 public:
  using EraseKernel = kernels::EraseGpu<T, Dims, channel_dim>;

  explicit EraseImplGpu(const OpSpec &spec) : spec_(spec) {
    kmgr_.Resize<EraseKernel>(1, 1);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<GPUBackend> &ws) override {
    const auto &input = ws.template InputRef<GPUBackend>(0);
    auto layout = input.GetLayout();
    auto type = input.type();
    auto shape = input.shape();
    AcquireArgs(ws, shape, layout);

    auto in_view = view<const T, Dims>(input);
    kmgr_.Initialize<EraseKernel>();
    ctx_.gpu.stream = ws.stream();
    auto regions_view = view<kernels::ibox<Dims>, 1>(regions_gpu_);
    kmgr_.Setup<EraseKernel>(0, ctx_, in_view, regions_view, make_cspan(fill_values_));

    output_desc.resize(1);
    output_desc[0] = {shape, input.type()};
    return true;
  };

  void RunImpl(workspace_t<GPUBackend> &ws) override {
    const auto &input_ref = ws.template InputRef<GPUBackend>(0);
    auto &output_ref = ws.template OutputRef<GPUBackend>(0);
    output_ref.SetLayout(input_ref.GetLayout());
    auto input = view<const T, Dims>(input_ref);
    auto output = view<T, Dims>(output_ref);
    auto regions_view = view<kernels::ibox<Dims>, 1>(regions_gpu_);
    kmgr_.Run<EraseKernel>(0, 0, ctx_, output, input, regions_view, make_cspan(fill_values_));
  }

  void AcquireArgs(const DeviceWorkspace &ws, TensorListShape<> in_shape,
                   TensorLayout in_layout) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    fill_values_ = spec_.template GetRepeatedArgument<float>("fill_value");
    auto args = detail::GetEraseArgs<T, Dims>(spec_, ws, in_shape, in_layout);
    auto regions_shape = TensorListShape<1>(curr_batch_size);
    for (int i = 0; i < curr_batch_size; ++i) {
      auto n_regions = static_cast<int>(args[i].rois.size());
      regions_shape.set_tensor_shape(i, {n_regions});
    }
    TensorList<CPUBackend> regions_cpu;
    regions_cpu.set_type(TypeInfo::Create<kernels::ibox<Dims>>());
    regions_cpu.Resize(regions_shape);
    auto regions_tlv = view<kernels::ibox<Dims>, 1>(regions_cpu);
    for (int i = 0; i < curr_batch_size; ++i) {
      auto regions_tv = regions_tlv[i];
      for (int j = 0; j < regions_tv.shape[0]; ++j) {
        auto box = detail::make_box(args[i].rois[j].anchor, args[i].rois[j].shape);
        *regions_tv(j) = box;
      }
    }
    regions_gpu_.Copy(regions_cpu, ws.stream());
  }

 private:
  const OpSpec &spec_;
  std::vector<int> axes_;
  std::vector<kernels::EraseArgs<T, Dims>> args_;
  TensorList<GPUBackend> regions_gpu_;
  SmallVector<T, 3> fill_values_;
  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
};

template <>
bool Erase<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                  const workspace_t<GPUBackend> &ws) {
  const auto &input = ws.InputRef<GPUBackend>(0);
  auto in_shape = input.shape();
  auto channel_dim = input.GetLayout().find('C');
  TYPE_SWITCH(input.type().id(), type2id, T, ERASE_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, ERASE_SUPPORTED_NDIMS, (
      if (channel_dim == -1)
        impl_ = std::make_unique<EraseImplGpu<T, Dims, -1>>(spec_);
      else if (channel_dim == 0)
        impl_ = std::make_unique<EraseImplGpu<T, Dims, 0>>(spec_);
      else if (channel_dim == Dims-1)
        impl_ = std::make_unique<EraseImplGpu<T, Dims, Dims-1>>(spec_);
      else
        DALI_FAIL("Unsupported layout. Only 'no channel', "
                  "'channel first' and 'channel last' layouts are supported.");
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
