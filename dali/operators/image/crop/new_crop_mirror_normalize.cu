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

#include <any>
#include <cstdint>
#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/float16.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/util.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/slice/slice_flip_normalize_gpu.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_gpu.h"
#include "dali/kernels/slice/slice_hwc2chw_normalize_gpu.h"
#include "dali/operators/generic/slice/out_of_bounds_policy.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/operators/image/crop/crop_mirror_normalize.h"  // TODO(janton): remove fallback
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/workspace/workspace.h"

#define IN_TYPES (uint8_t, int16_t, uint16_t, int32_t, float, float16)
#define OUT_TYPES (float, uint8_t, int8_t, float16)
#define SPATIAL_NDIMS (2)
#define CHANNEL_DIMS (0, 2)

namespace dali {

class NewCropMirrorNormalizeGPU : public StatelessOperator<GPUBackend> {
 public:
  explicit inline NewCropMirrorNormalizeGPU(const OpSpec &spec)
      : StatelessOperator<GPUBackend>(spec),
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
  enum class CmnImplKind {
    SliceFlipNormalizeGpuGeneric,
    SliceHwc2HwcChwNormalize,
    FallbackGeneric
  };

  /**
   * @brief Select which implementation to use based on the supported parameters.
   */
  CmnImplKind GetImplementationKind(DALIDataType in_type, DALIDataType out_type,
                                    const TensorLayout &in_layout, const TensorLayout &out_layout,
                                    int ndim, int spatial_dim, int channel_dim,
                                    const TensorListShape<> &in_shape, OutOfBoundsPolicy oobp) {
    if (spatial_dim != 2 || ndim != 3 ||
        (channel_dim_idx_ != 0 && channel_dim_idx_ != spatial_ndim_)) {
      return CmnImplKind::FallbackGeneric;
    }
    // check for optimized version
    if (in_type == DALI_UINT8 && (out_type == DALI_FLOAT || out_type == DALI_FLOAT16) &&
        in_layout == "HWC" && (out_layout == "CHW" || out_layout == "HWC") &&
        (oobp == OutOfBoundsPolicy::Error || oobp == OutOfBoundsPolicy::TrimToShape)) {
      if (in_shape.num_samples() > 0 && in_shape.tensor_shape_span(0)[2] == 3)
        return CmnImplKind::SliceHwc2HwcChwNormalize;
    }
    return CmnImplKind::SliceFlipNormalizeGpuGeneric;
  }

  bool SetupSliceHwc2HwcChwNormalize(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
    TYPE_SWITCH(output_type_, type2id, OutputType, (float, float16), (
      return SetupSliceHwc2HwcChwNormalizeTyped<OutputType>(output_desc, ws);
    ), DALI_FAIL(make_string("Unsupported output type:", output_type_)););  // NOLINT
  }

  template <typename Out>
  bool SetupSliceHwc2HwcChwNormalizeTyped(std::vector<OutputDesc> &output_desc,
                                          const Workspace &ws) {
    using Kernel = kernels::slice_flip_normalize::SliceHwc2HwcChwNormalizeGPU<Out>;
    if (!kernel_args_.has_value())
      kernel_args_ = std::vector<typename Kernel::SampleArgs>{};
    auto &args = std::any_cast<std::vector<typename Kernel::SampleArgs> &>(kernel_args_);

    auto num_samples = ws.GetInputBatchSize(0);
    output_desc.resize(1);

    const auto &input = ws.Input<GPUBackend>(0);
    const auto &in_shape = input.shape();

    args.clear();
    args.reserve(num_samples);
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto sample_sh = in_shape[sample_idx];
      int num_channels = sample_sh[2];
      args.emplace_back();
      auto &a = args.back();

      a.roi = GetRoi(sample_idx, 0, 1, sample_sh);

      // Horizontal flip
      a.flip_x = this->spec_.template GetArgument<int>("mirror", &ws, sample_idx);

      GetNormParameters(a.mean, a.inv_stddev, sample_idx, num_channels);
      GetFillValuesParameters(a.fill_values, sample_idx, num_channels);
    }

    kmgr_.Resize<Kernel>(1);
    kernels::KernelContext ctx;
    ctx.gpu.stream = ws.stream();
    auto sh = input.shape().to_static<3>();
    // Kernel &k = kmgr_.Get<Kernel>(0);
    // const auto &req = k.Setup(ctx, sh, cargs);
    // // k.test();
    auto cargs = make_cspan(args);
    auto &req = kmgr_.Setup<Kernel>(0, ctx, sh, cargs, output_layout_);
    output_desc[0].type = output_type_;
    output_desc[0].shape = req.output_shapes[0];
    return true;
  }

  bool SetupSfnGpuGeneric(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
    TYPE_SWITCH(input_type_, type2id, InputType, IN_TYPES, (
      TYPE_SWITCH(output_type_, type2id, OutputType, OUT_TYPES, (
        VALUE_SWITCH(spatial_ndim_, SpatialNdim, SPATIAL_NDIMS, (
          VALUE_SWITCH(channel_dim_idx_, ChannelDim, CHANNEL_DIMS, (
            return SetupSfnGpuGenericTyped<OutputType, InputType,
                                           SpatialNdim, ChannelDim>(output_desc, ws);
          ), DALI_FAIL(make_string("Unsupported channel dimension:", channel_dim_idx_)););  // NOLINT
        ), DALI_FAIL(make_string("Unsupported number of spatial dimensions:", spatial_ndim_)););  // NOLINT
      ), DALI_FAIL(make_string("Unsupported output type:", output_type_)););  // NOLINT
    ), DALI_FAIL(make_string("Unsupported input type:", input_type_)););  // NOLINT
  }

  template <typename Out, typename In, int spatial_ndim, int channel_dim>
  bool SetupSfnGpuGenericTyped(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
    static constexpr int ndim = spatial_ndim + 1;
    auto num_samples = ws.GetInputBatchSize(0);
    output_desc.resize(1);

    const auto &input = ws.Input<GPUBackend>(0);
    const auto &in_shape = input.shape();

    using Kernel =
        kernels::slice_flip_normalize::SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>;
    if (!kernel_args_.has_value())
      kernel_args_ = typename Kernel::Args{};
    auto &args = std::any_cast<typename Kernel::Args &>(kernel_args_);

    int h_dim = input_layout_.find('H');
    assert(h_dim >= 0);
    int w_dim = input_layout_.find('W');
    assert(w_dim >= 0);

    auto permuted_dims = GetLayoutMapping<ndim>(input_layout_, output_layout_);
    for (int d = 0; d < ndim; d++)
      args.perm[d] = permuted_dims[d];

    args.sample_args.clear();
    args.sample_args.reserve(num_samples);
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      auto sample_sh = in_shape[sample_idx];
      int num_channels = sample_sh[channel_dim];
      args.sample_args.emplace_back();
      auto &a = args.sample_args.back();

      a.roi = GetRoi(sample_idx, h_dim, w_dim, sample_sh);

      // Horizontal flip
      a.flip.x = this->spec_.template GetArgument<int>("mirror", &ws, sample_idx);

      GetNormParameters(a.mean, a.inv_stddev, sample_idx, num_channels);
      GetFillValuesParameters(a.fill_values, sample_idx, num_channels);
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

    if (output_layout_.empty()) {
      output_layout_ = input_layout_;
    } else {
      DALI_ENFORCE(output_layout_.is_permutation_of(input_layout_),
                   "The requested output layout is not a permutation of input layout.");
    }

    auto num_samples = ws.GetInputBatchSize(0);
    auto in_shape = input.shape();
    int ndim = in_shape.sample_dim();

    spatial_ndim_ = ImageLayoutInfo::NumSpatialDims(input_layout_);
    DALI_ENFORCE(ImageLayoutInfo::HasChannel(input_layout_),
                 "This operator expects an explicit channel dimension, even for monochrome images");
    DALI_ENFORCE(input_layout_.ndim() == ndim);
    channel_dim_idx_ = ImageLayoutInfo::ChannelDimIndex(input_layout_);
    assert(channel_dim_idx_ >= 0);

    auto impl_kind =
        GetImplementationKind(input_type_, output_type_, input_layout_, output_layout_, ndim,
                              spatial_ndim_, channel_dim_idx_, in_shape, out_of_bounds_policy_);
    if (impl_kind_ != impl_kind) {
      kernel_args_.reset();
      kmgr_.Reset();
      impl_kind_ = impl_kind;
    }


    if (impl_kind_ == CmnImplKind::FallbackGeneric) {
      DALI_WARN_ONCE("using CropMirrorNormalize legacy implementation");
      return fallback_.Setup(output_desc, ws);
    }
    crop_attr_.ProcessArguments(spec_, ws);

    ArgValueFlags flags = ArgValue_EnforceUniform;
    mean_arg_.Acquire(spec_, ws, num_samples, flags);
    std_arg_.Acquire(spec_, ws, num_samples, flags);

    if (impl_kind_ == CmnImplKind::SliceFlipNormalizeGpuGeneric) {
      return SetupSfnGpuGeneric(output_desc, ws);
    }

    return SetupSliceHwc2HwcChwNormalize(output_desc, ws);
  }

  void RunSliceHwc2HwcChwNormalize(Workspace &ws) {
    if (output_type_ == DALI_FLOAT) {
      using Kernel = kernels::slice_flip_normalize::SliceHwc2HwcChwNormalizeGPU<float>;

      auto &args = std::any_cast<std::vector<typename Kernel::SampleArgs> &>(kernel_args_);
      auto cargs = make_cspan(args);
      RunSfnKernel<Kernel, float, uint8_t, 3>(ws, cargs);
      return;
    } else if (output_type_ == DALI_FLOAT16) {
      using Kernel = kernels::slice_flip_normalize::SliceHwc2HwcChwNormalizeGPU<float16>;

      auto &args = std::any_cast<std::vector<typename Kernel::SampleArgs> &>(kernel_args_);
      auto cargs = make_cspan(args);
      RunSfnKernel<Kernel, float16, uint8_t, 3>(ws, cargs);
      return;
    }
    DALI_FAIL(make_string("Unsupported output type:", output_type_));
  }

  void RunSfnGpuGeneric(Workspace &ws) {
    TYPE_SWITCH(input_type_, type2id, InputType, IN_TYPES, (
      TYPE_SWITCH(output_type_, type2id, OutputType, OUT_TYPES, (
        VALUE_SWITCH(spatial_ndim_, SpatialNdim, SPATIAL_NDIMS, (
          VALUE_SWITCH(channel_dim_idx_, ChannelDim, CHANNEL_DIMS, (
            using Kernel =
                kernels::slice_flip_normalize::SliceFlipNormalizeGPU<OutputType, InputType,
                                                                     SpatialNdim, ChannelDim>;
            constexpr int ndim = SpatialNdim + 1;

            auto &args = std::any_cast<typename Kernel::Args &>(kernel_args_);
            RunSfnKernel<Kernel, OutputType, InputType, ndim>(ws, args);
          ), DALI_FAIL(make_string("Unsupported channel dimension:", channel_dim_idx_)););  // NOLINT
        ), DALI_FAIL(make_string("Unsupported number of spatial dimensions:", spatial_ndim_)););  // NOLINT
      ), DALI_FAIL(make_string("Unsupported output type:", output_type_)););  // NOLINT
    ), DALI_FAIL(make_string("Unsupported input type:", input_type_)););  // NOLINT
  }

  template <typename Kernel, typename Out, typename In, int ndim, typename Args>
  void RunSfnKernel(const Workspace &ws, const Args &args) {
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
    if (impl_kind_ == CmnImplKind::FallbackGeneric) {
      fallback_.Run(ws);
      return;
    }

    if (impl_kind_ == CmnImplKind::SliceFlipNormalizeGpuGeneric) {
      RunSfnGpuGeneric(ws);
      return;
    }

    RunSliceHwc2HwcChwNormalize(ws);
  }


  /**
   * @brief Compute the 2D ROI for given sample_idx, crop_attr_ must be update first.
   *
   * @param sample_idx Index of the sample
   * @param h_dim Index of "H" dimension
   * @param w_dim Index of "W" dimension
   */
  kernels::Roi<2> GetRoi(int sample_idx, int h_dim, int w_dim,
                         const TensorShape<-1> &sample_shape) {
    auto crop_win_gen = crop_attr_.GetCropWindowGenerator(sample_idx);
    assert(crop_win_gen);
    CropWindow crop_window = crop_win_gen(sample_shape, input_layout_);
    ApplySliceBoundsPolicy(out_of_bounds_policy_, sample_shape, crop_window.anchor,
                           crop_window.shape);

    kernels::Roi<2> roi;

    roi.lo.x = crop_window.anchor[w_dim];
    roi.hi.x = roi.lo.x + crop_window.shape[w_dim];
    roi.lo.y = crop_window.anchor[h_dim];
    roi.hi.y = roi.lo.y + crop_window.shape[h_dim];
    return roi;
  }

  void GetNormParameters(SmallVector<float, 4> &mean, SmallVector<float, 4> &inv_stddev,
                         int sample_idx, int num_channels) {
    span<const float> mean_arg(mean_arg_[sample_idx].data, mean_arg_[sample_idx].num_elements());
    span<const float> std_arg(std_arg_[sample_idx].data, std_arg_[sample_idx].num_elements());
    auto nargs = std::max(mean_arg.size(), std_arg.size());
    DALI_ENFORCE(mean_arg.size() == std_arg.size() || mean_arg.size() == 1 || std_arg.size() == 1,
                 "``mean`` and ``std`` must either be of the same size, be scalars, or one of "
                 "them can be a "
                 "vector and the other a scalar.");
    mean.resize(num_channels);
    inv_stddev.resize(num_channels);
    for (int c = 0; c < num_channels; c++) {
      double mean_val = mean_arg[c % mean_arg.size()];
      double std_val = std_arg[c % std_arg.size()];
      mean[c] = std::fma(-shift_, std_val / scale_, mean_val);
      inv_stddev[c] = scale_ / std_val;
    }
  }

  void GetFillValuesParameters(SmallVector<float, 4> &fill_values, int sample_idx,
                               int num_channels) {
    if (fill_values_.size() > 0) {
      fill_values.resize(num_channels);
      DALI_ENFORCE(
          fill_values_.size() == 1 || static_cast<int>(fill_values_.size()) >= num_channels,
          "Should provide either a single fill_value or at least as many as number of "
          "channels in the input");
      for (int c = 0; c < num_channels; c++)
        fill_values[c] = fill_values_[c % fill_values_.size()];
    }

    if (pad_output_) {
      int out_num_channels = next_pow2(num_channels);
      int start_c = fill_values.size();
      for (int c = start_c; c < out_num_channels; c++) {
        fill_values.push_back(0.0f);
      }
    }
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
  CmnImplKind impl_kind_ = CmnImplKind::FallbackGeneric;

  std::vector<float> fill_values_;
  OutOfBoundsPolicy out_of_bounds_policy_ = OutOfBoundsPolicy::Error;

  ArgValue<float, 1> mean_arg_;
  ArgValue<float, 1> std_arg_;
  float scale_ = 1.0f;
  float shift_ = 0.0f;

  kernels::KernelManager kmgr_;
  std::any kernel_args_;

  USE_OPERATOR_MEMBERS();
};


DALI_REGISTER_OPERATOR(CropMirrorNormalize, NewCropMirrorNormalizeGPU, GPU);

}  // namespace dali
