// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  /**
   * @param spec  Pointer to a persistent OpSpec object,
   *              which is guaranteed to be alive for the entire lifetime of this object
   */
  explicit EraseImplGpu(const OpSpec *spec) : spec_(*spec), fill_value_arg_{"fill_value", *spec} {
    kmgr_.Resize<EraseKernel>(1);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    auto layout = input.GetLayout();
    auto type = input.type();
    auto shape = input.shape();
    AcquireArgs(ws, shape, layout);

    auto in_view = view<const T, Dims>(input);
    kmgr_.Initialize<EraseKernel>();
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    auto regions_view = view<kernels::ibox<Dims>, 1>(regions_gpu_);
    kmgr_.Setup<EraseKernel>(0, ctx, in_view);

    output_desc.resize(1);
    output_desc[0] = {shape, input.type()};
    return true;
  };

  void RunImpl(Workspace &ws) override {
    const auto &input_ref = ws.Input<GPUBackend>(0);
    auto &output_ref = ws.Output<GPUBackend>(0);
    output_ref.SetLayout(input_ref.GetLayout());
    auto input = view<const T, Dims>(input_ref);
    int nsamples = input.num_samples();
    auto output = view<T, Dims>(output_ref);
    auto regions_view = view<kernels::ibox<Dims>, 1>(regions_gpu_);
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();

    if (!fill_value_arg_) {  // default fill values
      kmgr_.Run<EraseKernel>(0, ctx, output, input, regions_view);
    } else {
      auto fill_value_gpu = view<const T, 1>(fill_value_gpu_);
      kmgr_.Run<EraseKernel>(0, ctx, output, input, regions_view, fill_value_gpu);
    }
  }

  void AcquireArgs(const Workspace &ws, TensorListShape<> in_shape,
                   TensorLayout in_layout) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    auto args = detail::GetEraseArgs<T, Dims>(spec_, ws, fill_value_arg_, in_shape, in_layout);
    auto regions_shape = TensorListShape<1>(curr_batch_size);
    for (int i = 0; i < curr_batch_size; ++i) {
      auto n_regions = static_cast<int>(args[i].rois.size());
      regions_shape.set_tensor_shape(i, {n_regions});
    }
    TensorList<CPUBackend> regions_cpu;
    regions_cpu.set_type<kernels::ibox<Dims>>();
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

    auto read_fill_value = [&]() {
      int nfill_value = fill_value_arg_[0].shape[0];  // uniform shape already enforced
      int nchannels = channel_dim >= 0 ? in_shape.tensor_shape_span(0)[channel_dim] : 1;
      if (nchannels != nfill_value && nfill_value > 1) {
        throw std::invalid_argument(make_string(
            "Expected as many fill values as the number of channels. nchannels=", nchannels,
            ", nfill_value=", nfill_value));
      }

      fill_value_cpu_.Resize(uniform_list_shape<1>(curr_batch_size, TensorShape<1>{nchannels}),
                             type2id<T>::value);

      for (int i = 0; i < curr_batch_size; ++i) {
        auto fill_value_arg_tv = fill_value_arg_[i];
        assert(nfill_value == fill_value_arg_tv.shape[0]);
        T *fill_value_ptr = fill_value_cpu_.mutable_tensor<T>(i);
        for (int c = 0; c < nchannels; c++) {
          fill_value_ptr[c] = ConvertSat<T>(fill_value_arg_tv.data[nfill_value > 1 ? c : 0]);
        }
      }
      fill_value_gpu_.Copy(fill_value_cpu_, ws.stream());
    };

    if (fill_value_arg_.HasArgumentInput()) {
      read_fill_value();
    } else if (fill_value_arg_ && !constant_fill_value_cpu_read_) {
      // If constant fill values, only read once, next time reuse
      read_fill_value();
      constant_fill_value_cpu_read_ = true;
    }
  }

 private:
  const OpSpec &spec_;
  ArgValue<float, 1> fill_value_arg_;
  std::vector<int> axes_;
  std::vector<kernels::EraseArgs<T, Dims>> args_;
  TensorList<GPUBackend> regions_gpu_;
  TensorList<CPUBackend> fill_value_cpu_;
  TensorList<GPUBackend> fill_value_gpu_;
  kernels::KernelManager kmgr_;

  bool constant_fill_value_cpu_read_ = false;
};

template <>
bool Erase<GPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                  const Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  auto in_shape = input.shape();
  auto channel_dim = input.GetLayout().find('C');
  TYPE_SWITCH(input.type(), type2id, T, ERASE_SUPPORTED_TYPES, (
    VALUE_SWITCH(in_shape.sample_dim(), Dims, ERASE_SUPPORTED_NDIMS, (
      if (channel_dim == -1)
        impl_ = std::make_unique<EraseImplGpu<T, Dims, -1>>(&spec_);
      else if (channel_dim == 0)
        impl_ = std::make_unique<EraseImplGpu<T, Dims, 0>>(&spec_);
      else if (channel_dim == Dims-1)
        impl_ = std::make_unique<EraseImplGpu<T, Dims, Dims-1>>(&spec_);
      else
        DALI_FAIL("Unsupported layout. Only 'no channel', "
                  "'channel first' and 'channel last' layouts are supported.");
    ), DALI_FAIL(make_string("Unsupported number of dimensions ", in_shape.size())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported data type: ", input.type())));  // NOLINT

  assert(impl_ != nullptr);
  return impl_->SetupImpl(output_desc, ws);
}

template <>
void Erase<GPUBackend>::RunImpl(Workspace &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

DALI_REGISTER_OPERATOR(Erase, Erase<GPUBackend>, GPU);

}  // namespace dali
