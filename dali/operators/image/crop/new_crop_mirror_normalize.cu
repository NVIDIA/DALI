// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <utility>
#include <vector>
#include "dali/core/any.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/float16.h"
#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/slice/slice_flip_normalize_gpu.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_gpu.h"
#include "dali/operators/generic/slice/out_of_bounds_policy.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/operators/image/crop/crop_mirror_normalize.h"  // TODO(janton): remove fallback
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

#define IN_TYPES (uint8_t, int16_t, uint16_t, int32_t, float, float16)
#define OUT_TYPES (float , uint8_t, int8_t, float16)
#define SPATIAL_NDIMS (2)
#define CHANNEL_DIMS (0, 2)

namespace dali {

class NewCropMirrorNormalizeGPU : public Operator<GPUBackend> {
  public:
  explicit inline NewCropMirrorNormalizeGPU(const OpSpec &spec)
      : Operator<GPUBackend>(spec),
        fallback_(spec),
        crop_attr_(spec),
        output_type_(spec.GetArgument<DALIDataType>("dtype")),
        output_layout_(spec.GetArgument<TensorLayout>("output_layout")),
        pad_output_(spec.GetArgument<bool>("pad_output")),
        out_of_bounds_policy_(GetOutOfBoundsPolicy(spec)),
        mean_arg_("mean", spec),
        std_arg_("std", spec),
        scale_(spec.GetArgument<float>("scale")),
        shift_(spec.GetArgument<float>("shift")) {
    if (out_of_bounds_policy_ == OutOfBoundsPolicy::Pad) {
      fill_values_ = spec.GetRepeatedArgument<float>("fill_values");
    }
  }

  inline ~NewCropMirrorNormalizeGPU() override = default;

 protected:
  template <typename Out, typename In, int spatial_ndim, int channel_dim>
  bool SetupImplTyped(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    static constexpr int ndim = spatial_ndim + 1;
    auto nsamples = ws.GetInputBatchSize(0);
    output_desc.resize(1);

    const auto &input = ws.Input<GPUBackend>(0);
    auto in_shape = input.shape();

    using Kernel = kernels::slice_flip_normalize::SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>;
    if (!kernel_args_.has_value())
    kernel_args_ = typename Kernel::Args{};
    auto &args = any_cast<typename Kernel::Args&>(kernel_args_);

    int h_dim = input_layout_.find('H');
    assert(h_dim >= 0);
    int w_dim = input_layout_.find('W');
    assert(w_dim >= 0);

    if (output_layout_.empty()) {
      output_layout_ = input_layout_;
      std::iota(args.perm.begin(), args.perm.end(), 0);
    } else {
      DALI_ENFORCE(output_layout_.is_permutation_of(input_layout_),
        "The requested output layout is not a permutation of input layout.");
      auto permuted_dims = GetLayoutMapping<ndim>(input_layout_, output_layout_);
      for (int d = 0; d < ndim; d++)
        args.perm[d] = permuted_dims[d];
    }

    args.sample_args.clear();
    args.sample_args.reserve(nsamples);
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      auto sample_sh = in_shape[sample_idx];
      args.sample_args.emplace_back();
      auto &a = args.sample_args.back();

      auto crop_win_gen = crop_attr_.GetCropWindowGenerator(sample_idx);
      assert(crop_win_gen);
      CropWindow crop_window = crop_win_gen(sample_sh, input_layout_);
      ApplySliceBoundsPolicy(out_of_bounds_policy_, sample_sh, crop_window.anchor,
        crop_window.shape);

      a.roi.lo.x = crop_window.anchor[w_dim];
      a.roi.hi.x = a.roi.lo.x + crop_window.shape[w_dim];
      a.roi.lo.y = crop_window.anchor[h_dim];
      a.roi.hi.y = a.roi.lo.y + crop_window.shape[h_dim];

      // Horizontal flip
      a.flip.x = this->spec_.template GetArgument<int>("mirror", &ws, sample_idx);

      span<const float> mean_arg(mean_arg_[sample_idx].data, mean_arg_[sample_idx].num_elements());
      span<const float> std_arg(std_arg_[sample_idx].data, std_arg_[sample_idx].num_elements());
      int nchannels = sample_sh[channel_dim];
      auto nargs = std::max(mean_arg.size(), std_arg.size());
      DALI_ENFORCE(
          mean_arg.size() == std_arg.size() || mean_arg.size() == 1 || std_arg.size() == 1,
          "``mean`` and ``std`` must either be of the same size, be scalars, or one of them can be a "
          "vector and the other a scalar.");
      a.mean.resize(nchannels);
      a.inv_stddev.resize(nchannels);
      for (int c = 0; c < nchannels; c++) {
        double mean_val = mean_arg[c % mean_arg.size()];
        double std_val = std_arg[c % std_arg.size()];
        a.mean[c] = std::fma(-shift_, std_val / scale_, mean_val);
        a.inv_stddev[c] = scale_ / std_val;
      }

      if (fill_values_.size() > 0) {
        a.fill_values.resize(nchannels);
        DALI_ENFORCE(fill_values_.size() == 1 || fill_values_.size() >= nchannels,
                     "Should provide either a single fill_value or at least as many as number of "
                     "channels in the input");
        for (int c = 0; c < nchannels; c++)
          a.fill_values[c] = fill_values_[c % fill_values_.size()];
      }

      if (pad_output_) {
        int out_nchannels = next_pow2(nchannels);
        int start_c = a.fill_values.size();
        for (int c = start_c; c < out_nchannels; c++) {
          a.fill_values.push_back(0.0f);
        }
      }
    }

    kmgr_.Resize<Kernel>(1);
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    auto sh = input.shape().to_static<ndim>();
    auto &req = kmgr_.Setup<Kernel>(0, ctx, sh, args);
    output_desc[0].type = output_type_;
    output_desc[0].shape = req.output_shapes[0];
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    input_type_ = input.type();
    input_layout_ = input.GetLayout();
    assert(output_type_ != DALI_NO_TYPE);

    auto nsamples = ws.GetInputBatchSize(0);
    auto in_shape = input.shape();
    int ndim = in_shape.sample_dim();

    spatial_ndim_ = ImageLayoutInfo::NumSpatialDims(input_layout_);
    DALI_ENFORCE(ImageLayoutInfo::HasChannel(input_layout_),
                 "This operator expects an explicit channel dimension, even for monochrome images");
    DALI_ENFORCE(input_layout_.ndim() == ndim);
    channel_dim_idx_ = ImageLayoutInfo::ChannelDimIndex(input_layout_);
    assert(channel_dim_idx_ >= 0);

    // TODO(janton): remove fallback
    use_fallback_ = spatial_ndim_ != 2 || ndim != 3 ||
                    (channel_dim_idx_ != 0 && channel_dim_idx_ != spatial_ndim_);// ||
                    //pad_output_;

    if (use_fallback_)
      return fallback_.Setup(output_desc, ws);

    crop_attr_.ProcessArguments(spec_, ws);

    ArgValueFlags flags = ArgValue_EnforceUniform;
    mean_arg_.Acquire(spec_, ws, nsamples, flags);
    std_arg_.Acquire(spec_, ws, nsamples, flags);

    TYPE_SWITCH(input_type_, type2id, InputType, IN_TYPES, (
      TYPE_SWITCH(output_type_, type2id, OutputType, OUT_TYPES, (
        VALUE_SWITCH(spatial_ndim_, SpatialNdim, SPATIAL_NDIMS, (
          VALUE_SWITCH(channel_dim_idx_, ChannelDimIndex, CHANNEL_DIMS, (
            return SetupImplTyped<OutputType, InputType, SpatialNdim, ChannelDimIndex>(output_desc, ws);
          ), DALI_FAIL(make_string("Not supported channel dimension:", channel_dim_idx_)););  // NOLINT
        ), DALI_FAIL(make_string("Not supported number of spatial dimensions:", spatial_ndim_)););  // NOLINT
      ), DALI_FAIL(make_string("Not supported output type:", output_type_)););  // NOLINT
    ), DALI_FAIL(make_string("Not supported input type:", input_type_)););  // NOLINT
  }

  template <typename Out, typename In, int spatial_ndim, int channel_dim>
  void RunImplTyped(const Workspace &ws) override {
    static constexpr int ndim = spatial_ndim + 1;
    using Kernel = kernels::slice_flip_normalize::SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>;
    auto &args = any_cast<typename Kernel::Args&>(kernel_args_);
    const auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout(output_layout_);
    auto in_view = view<const In, ndim>(input);
    auto out_view = view<Out, ndim>(output);
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    kmgr_.Run<Kernel>(0, ctx, out_view, in_view, args);
  }

  void RunImpl(Workspace &ws) override {
    if (use_fallback_) {
      fallback_.Run(ws);
      return;
    }

    TYPE_SWITCH(input_type_, type2id, InputType, IN_TYPES, (
      TYPE_SWITCH(output_type_, type2id, OutputType, OUT_TYPES, (
        VALUE_SWITCH(spatial_ndim_, SpatialNdim, SPATIAL_NDIMS, (
          VALUE_SWITCH(channel_dim_idx_, ChannelDimIndex, CHANNEL_DIMS, (
            RunImplTyped<OutputType, InputType, SpatialNdim, ChannelDimIndex>(ws);
          ), DALI_FAIL(make_string("Not supported channel dimension:", channel_dim_idx_)););  // NOLINT
        ), DALI_FAIL(make_string("Not supported number of spatial dimensions:", spatial_ndim_)););  // NOLINT
      ), DALI_FAIL(make_string("Not supported output type:", output_type_)););  // NOLINT
    ), DALI_FAIL(make_string("Not supported input type:", input_type_)););  // NOLINT
  }

  bool CanInferOutputs() const override {
    return true;
  }

  CropMirrorNormalize<GPUBackend> fallback_;
  CropAttr crop_attr_;

  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;
  TensorLayout input_layout_;
  TensorLayout output_layout_;
  int channel_dim_idx_;
  int spatial_ndim_;
  bool pad_output_;  // Whether to pad channel dimension to the next power of 2
  bool use_fallback_ = false;  // whether to use old implementation

  std::vector<float> fill_values_;
  OutOfBoundsPolicy out_of_bounds_policy_ = OutOfBoundsPolicy::Error;

  ArgValue<float, 1> mean_arg_;
  ArgValue<float, 1> std_arg_;
  float scale_ = 1.0f;
  float shift_ = 0.0f;

  kernels::KernelManager kmgr_;
  any kernel_args_;

  USE_OPERATOR_MEMBERS();
};


DALI_REGISTER_OPERATOR(CropMirrorNormalize, NewCropMirrorNormalizeGPU, GPU);

}  // namespace dali
