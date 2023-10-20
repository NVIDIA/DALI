// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/geom/geom_utils.h"
#include "dali/core/geom/vec.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/convolution/filter_gpu.cuh"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/convolution/filter.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

namespace filter {

template <typename Out, typename In, typename W, int axes, bool is_sequence, bool has_channels,
          bool enable_roi>
class FilterOpGpu : public OpImplBase<GPUBackend> {
 public:
  using Kernel = kernels::FilterGpu<Out, In, W, has_channels, is_sequence, axes, enable_roi>;
  static constexpr int ndim = Kernel::ndim;

  /**
   * @param spec  Pointer to a persistent OpSpec object,
   *              which is guaranteed to be alive for the entire lifetime of this object
   */
  explicit FilterOpGpu(const OpSpec* spec, InputDesc input_desc)
      : spec_{*spec},
        input_desc_{input_desc},
        anchor_arg_{"anchor", spec_},
        border_type_{parse_filter_border_type(spec_.GetArgument<std::string>("border"))} {
    kmgr_.Resize<Kernel>(1);
    filter_dev_.set_type(type2id<W>::value);
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace& ws) override {
    ctx_.gpu.stream = ws.stream();
    const auto& input = ws.template Input<GPUBackend>(0);
    int num_samples = input.num_samples();
    anchor_arg_.Acquire(spec_, ws, num_samples, TensorShape<1>{axes});
    output_desc.resize(1);
    output_desc[0].type = type2id<Out>::value;
    output_desc[0].shape = infer_output_shape(input.shape(), ws.GetInputShape(1), input_desc_);
    return true;
  }

  void RunImpl(Workspace& ws) override {
    const auto& input = ws.template Input<GPUBackend>(0);
    auto& output = ws.template Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    auto in_shape = input.shape();
    auto out_shape = output.shape();
    if (is_sequence) {
      in_shape = collapse_dims(in_shape, {{0, input_desc_.num_seq_dims}});
      out_shape = collapse_dims(out_shape, {{0, input_desc_.num_seq_dims}});
    }
    auto in_views_dyn = view<const In>(input);
    auto out_views_dyn = view<Out>(output);
    auto in_views = reshape<ndim>(in_views_dyn, in_shape.to_static<ndim>());
    auto out_views = reshape<ndim>(out_views_dyn, out_shape.to_static<ndim>());
    auto filter_views = GetFilterViews(ws);
    auto anchors = GetAnchors(filter_views.shape, input.num_samples());
    auto fill_value_views = GetFillValueViews(ws, input.num_samples());
    kmgr_.Run<Kernel>(0, ctx_, out_views, in_views, filter_views, anchors, border_type_,
                      fill_value_views);
  }

 private:
  TensorListView<StorageGPU, const W, axes> GetFilterViews(const Workspace& ws) {
    if (ws.template InputIsType<GPUBackend>(1)) {
      return view<const W, axes>(ws.template Input<GPUBackend>(1));
    } else {
      const auto& filters = ws.template Input<CPUBackend>(1);
      filter_dev_.set_order(ws.stream());
      filter_dev_.Copy(filters);
      return view<const W, axes>(filter_dev_);
    }
  }

  span<const ivec<axes>> GetAnchors(const TensorListShape<axes>& filter_shapes, int num_samples) {
    using kernels::shape2vec;
    if (input_desc_.is_valid_mode) {
      // in valid mode there is exactly one way to position filter so that
      // for each output point it lies fully within the input
      anchors_.clear();
      anchors_.resize(num_samples, 0);
      return make_cspan(anchors_);
    }
    anchors_.clear();
    anchors_.reserve(num_samples);
    auto anchor_views = anchor_arg_.get();
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto& anchor_view = anchor_views[sample_idx];
      assert(anchor_view.num_elements() == axes);  // relying on arg.Acquire validation here
      TensorShape<axes> anchor_shape;
      for (int dim = 0; dim < axes; dim++) {
        anchor_shape[dim] = anchor_view.data[dim];
      }
      ivec<axes> filter_extents = shape2vec(filter_shapes[sample_idx]);
      ivec<axes> anchor = shape2vec(anchor_shape);
      DALI_ENFORCE(all_coords(-1 <= anchor) && all_coords(anchor < filter_extents),
                    make_string("Anchor must lie within the filter. Got anchor ", anchor_shape,
                                " with a filter of shape ", filter_shapes[sample_idx],
                                " for sample of idx ", sample_idx, "."));
      for (int dim = 0; dim < axes; dim++) {
        anchor[dim] = anchor[dim] == -1 ? filter_extents[dim] / 2 : anchor[dim];
      }
      anchors_.push_back(anchor);
    }
    return make_cspan(anchors_);
  }

  TensorListView<StorageGPU, const In, 0> GetFillValueViews(const Workspace& ws, int num_samples) {
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
  InputDesc input_desc_;
  ArgValue<int, 1> anchor_arg_;
  BoundaryType border_type_;

  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
  TensorList<GPUBackend> filter_dev_;
  TensorList<GPUBackend> fill_values_dev_;
  std::vector<ivec<axes>> anchors_;
};

template <typename Out, typename In, typename W>
std::unique_ptr<OpImplBase<GPUBackend>> get_filter_gpu_op_impl(const OpSpec& spec_,
                                                               const InputDesc& input_desc) {
  VALUE_SWITCH(input_desc.axes, Axes, FILTER_INPUT_SUPPORTED_SPATIAL_NDIM_GPU, (
    BOOL_SWITCH(
      input_desc.num_seq_dims > 0, IsSequence, (
        BOOL_SWITCH(input_desc.has_channels, HasChannels, (
          BOOL_SWITCH(input_desc.is_valid_mode, EnableROI, (
            using OpImpl = FilterOpGpu<Out, In, W, Axes, IsSequence, HasChannels, EnableROI>;
            return std::make_unique<OpImpl>(&spec_, input_desc);
          ));  // NOLINT
        ));  // NOLINT
       ));   // NOLINT
  ), (   // NOLINT
    DALI_FAIL(make_string("Unsupported input data dimensionality. ",
              "Got input with ", input_desc.axes, "spatial dimensions. ",
              "Filter operator supports only 2 and 3 dimensional convolutions."));
  ));  // NOLINT
}

}  // namespace filter
}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_GPU_H_
