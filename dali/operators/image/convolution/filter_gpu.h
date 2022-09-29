// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_GPU_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_GPU_H_

#include <memory>
#include <string>
#include <vector>

#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/filter_gpu.cuh"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/filter.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

namespace filter {

template <typename Out, typename In, typename W, int num_seq_dims, bool has_channels_last>
class FilterOpGpu : public OpImplBase<GPUBackend> {
 public:
  static constexpr bool is_sequence = num_seq_dims > 0;
  using Kernel = kernels::Filter2dGpu<Out, In, W, has_channels_last, is_sequence>;
  static constexpr int ndim = Kernel::ndim;
  static constexpr int filter_ndim = Kernel::filter_ndim;

  /**
   * @param spec  Pointer to a persistent OpSpec object,
   *              which is guaranteed to be alive for the entire lifetime of this object
   */
  explicit FilterOpGpu(const OpSpec* spec)
      : spec_{*spec},
        anchor_arg_{"anchor", spec_},
        border_type_{parse_filter_border_type(spec_.GetArgument<std::string>("border_type"))} {
    kmgr_.Resize<Kernel>(1);
    filter_dev_.set_type(type2id<W>::value);
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const workspace_t<GPUBackend>& ws) override {
    ctx_.gpu.stream = ws.stream();
    const auto& input = ws.template Input<GPUBackend>(0);
    int num_samples = input.num_samples();
    anchor_arg_.Acquire(spec_, ws, num_samples, TensorShape<1>{filter_ndim});
    output_desc.resize(1);
    output_desc[0].type = type2id<Out>::value;
    output_desc[0].shape =
        infer_output_shape(input.shape(), ws.GetInputShape(1), border_type_, num_seq_dims);
    return true;
  }

  void RunImpl(workspace_t<GPUBackend>& ws) override {
    const auto& input = ws.template Input<GPUBackend>(0);
    auto& output = ws.template Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    auto in_shape = input.shape();
    auto out_shape = output.shape();
    if (is_sequence) {
      in_shape = collapse_dims(in_shape, {{0, num_seq_dims}});
      out_shape = collapse_dims(out_shape, {{0, num_seq_dims}});
    }
    auto in_views_dyn = view<const In>(input);
    auto out_views_dyn = view<Out>(output);
    auto in_views = reshape<ndim>(in_views_dyn, in_shape.to_static<ndim>());
    auto out_views = reshape<ndim>(out_views_dyn, out_shape.to_static<ndim>());
    auto anchor_views = anchor_arg_.get();
    auto filter_views = GetFilterViews(ws);
    auto fill_value_views = GetFillValueViews(ws, input.num_samples());
    kmgr_.Run<Kernel>(0, ctx_, out_views, in_views, filter_views, anchor_views, border_type_,
                      fill_value_views);
  }

 private:
  TensorListView<StorageGPU, const W, filter_ndim> GetFilterViews(
      const workspace_t<GPUBackend>& ws) {
    if (ws.template InputIsType<GPUBackend>(1)) {
      return view<const W, filter_ndim>(ws.template Input<GPUBackend>(1));
    } else {
      const auto& filters = ws.template Input<CPUBackend>(1);
      filter_dev_.set_order(ws.stream());
      filter_dev_.Copy(filters);
      return view<const W, filter_ndim>(filter_dev_);
    }
  }

  TensorListView<StorageGPU, const In, 0> GetFillValueViews(const workspace_t<GPUBackend>& ws,
                                                            int num_samples) {
    if (ws.NumInput() < 3) {
      return {};
    }
    if (ws.template InputIsType<GPUBackend>(2)) {
      return get_fill_values_view<In>(ws.template Input<GPUBackend>(2));
    } else {
      const auto& fill_values = ws.template Input<CPUBackend>(2);
      fill_values_dev_.set_order(ws.stream());
      fill_values_dev_.Copy(fill_values);
      return get_fill_values_view<In>(fill_values_dev_);
    }
  }

  const OpSpec& spec_;
  ArgValue<int, 1> anchor_arg_;
  BoundaryType border_type_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
  TensorList<GPUBackend> filter_dev_;
  TensorList<GPUBackend> fill_values_dev_;
};

template <typename Out, typename In, typename W, int num_seq_dims, bool has_channels_last>
constexpr int FilterOpGpu<Out, In, W, num_seq_dims, has_channels_last>::filter_ndim;

template <typename Out, typename In, typename W>
typename std::enable_if<!std::is_integral<In>::value || !std::is_integral<W>::value ||
                            std::is_unsigned<In>::value == std::is_unsigned<W>::value,
                        std::unique_ptr<OpImplBase<GPUBackend>>>::type
get_filter_gpu_op_impl(const OpSpec& spec_, const InputLayoutDesc& input_desc) {
  BOOL_SWITCH(
      input_desc.num_seq_dims > 0, IsSequence,
      (BOOL_SWITCH(input_desc.has_channels, HasChannels,
                   (return std::make_unique<FilterOpGpu<Out, In, W, IsSequence, HasChannels>>(
                               &spec_);));  // NOLINT
       ));                                  // NOLINT
}

template <typename Out, typename In, typename W>
typename std::enable_if<std::is_integral<In>::value && std::is_integral<W>::value &&
                            std::is_unsigned<In>::value != std::is_unsigned<W>::value,
                        std::unique_ptr<OpImplBase<GPUBackend>>>::type
get_filter_gpu_op_impl(const OpSpec& spec_, const InputLayoutDesc& input_desc) {
  DALI_FAIL(
      make_string("Input and filter types must be of the same signedness. Got input of type: ",
                  type2id<In>::value, " and filter of type: ", type2id<W>::value, "."));
}


}  // namespace filter
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_GPU_H_
