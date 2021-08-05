// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CROP_CROP_MIRROR_NORMALIZE_H_
#define DALI_OPERATORS_IMAGE_CROP_CROP_MIRROR_NORMALIZE_H_

#include <cstring>
#include <utility>
#include <vector>
#include "dali/core/any.h"
#include "dali/core/common.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/core/error_handling.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_common.h"
#include "dali/operators/generic/slice/out_of_bounds_policy.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"

#define CMN_IN_TYPES (uint8_t, int16_t, uint16_t, int32_t, float, float16)
#define CMN_OUT_TYPES (float, float16, uint8_t, int8_t)
#define CMN_NDIMS (3, 4, 5)

namespace dali {

namespace detail {

template <int Dims>
kernels::SliceFlipNormalizePermutePadArgs<Dims> ToSliceFlipNormalizePermutePadArgs(
    TensorShape<> input_shape, TensorLayout input_layout, TensorLayout output_layout,
    const CropWindow &win, bool horizontal_flip, bool pad_channels, span<const float> mean,
    span<const float> inv_stddev, span<const float> fill_values) {
  kernels::SliceFlipNormalizePermutePadArgs<Dims> args(win.shape.to_static<Dims>(), input_shape);
  args.anchor = win.anchor.to_static<Dims>();
  args.channel_dim = -1;

  int channel_dim_idx = ImageLayoutInfo::ChannelDimIndex(input_layout);
  assert(channel_dim_idx >= 0);
  auto &nchannels = args.shape[channel_dim_idx];

  auto norm_arg_size = mean.size();
  DALI_ENFORCE(norm_arg_size == inv_stddev.size());
  auto fill_values_size = fill_values.size();

  args.fill_values = fill_values;
  args.mean = mean;
  args.inv_stddev = inv_stddev;

  auto arg_per_ch_size = std::max(norm_arg_size, fill_values_size);
  if (arg_per_ch_size > 1) {
    args.channel_dim = channel_dim_idx;
  }

  if (pad_channels) {
    nchannels = next_pow2(nchannels);  // modifies args.shape
    if (norm_arg_size > 1) {
      for (int c = norm_arg_size; c < nchannels; c++) {
        args.mean.push_back(0.0f);
        args.inv_stddev.push_back(0.0f);
      }
    }
  }

  if (horizontal_flip) {
    int horizontal_dim_idx = input_layout.find('W');
    DALI_ENFORCE(horizontal_dim_idx >= 0,
                 make_string("[H]orizontal dimension not found in the input layout. Got: ",
                             input_layout.str()));
    args.flip[horizontal_dim_idx] = true;
  }

  // Calculate permutation, if needed
  args.permuted_dims = GetLayoutMapping<Dims>(input_layout, output_layout);
  return args;
}

}  // namespace detail


template <typename Backend>
class CropMirrorNormalize : public Operator<Backend> {
 public:
  explicit inline CropMirrorNormalize(const OpSpec &spec)
      : Operator<Backend>(spec),
        crop_attr_(spec),
        output_type_(spec.GetArgument<DALIDataType>("dtype")),
        output_layout_(spec.GetArgument<TensorLayout>("output_layout")),
        pad_output_(spec.GetArgument<bool>("pad_output")),
        out_of_bounds_policy_(GetOutOfBoundsPolicy(spec)) {
    float scale = spec.GetArgument<float>("scale");
    float shift = spec.GetArgument<float>("shift");

    if (!spec.TryGetRepeatedArgument(inv_std_vec_, "std")) {
      inv_std_vec_ = { spec.GetArgument<float>("std") };
    }
    if (!spec.TryGetRepeatedArgument(mean_vec_, "mean")) {
      mean_vec_ = { spec.GetArgument<float>("mean") };
    }


    DALI_ENFORCE(!mean_vec_.empty() && !inv_std_vec_.empty(),
      "mean and standard deviation can't be empty");

    DALI_ENFORCE(
      mean_vec_.size() == inv_std_vec_.size() || mean_vec_.size() == 1 || inv_std_vec_.size() == 1,
      "`mean` and `stddev` must either be of the same size, be scalars, or one of them can be a "
      "vector and the other a scalar.");

    // Handle irregular mean/std argument lengths
    auto args_size = std::max(mean_vec_.size(), inv_std_vec_.size());
    if (mean_vec_.size() != inv_std_vec_.size()) {
      if (mean_vec_.size() == 1)
        mean_vec_.resize(args_size, mean_vec_[0]);

      if (inv_std_vec_.size() == 1)
        inv_std_vec_.resize(args_size, inv_std_vec_[0]);
    }

    for (int i = 0, d = args_size; i < d; i++) {
      mean_vec_[i] -= shift * inv_std_vec_[i] / scale;
      inv_std_vec_[i] = scale / inv_std_vec_[i];
    }

    should_normalize_ =
        !std::all_of(mean_vec_.begin(), mean_vec_.end(), [](float x) { return x == 0.0f; }) ||
        !std::all_of(inv_std_vec_.begin(), inv_std_vec_.end(), [](float x) { return x == 1.0f; });
    if (!should_normalize_) {
      mean_vec_.clear();
      inv_std_vec_.clear();
    }

    if (out_of_bounds_policy_ == OutOfBoundsPolicy::Pad) {
      fill_values_ = spec.GetRepeatedArgument<float>("fill_values");
    }
  }

  inline ~CropMirrorNormalize() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  void RunImpl(workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

  void SetupCommonImpl(const workspace_t<Backend> &ws) {
    const auto &input = ws.template InputRef<Backend>(0);
    input_type_ = input.type().id();
    if (output_type_ == DALI_NO_TYPE)
      output_type_ = input_type_;

    auto in_shape = input.shape();
    input_layout_ = input.GetLayout();
    DALI_ENFORCE(ImageLayoutInfo::IsImage(input_layout_),
      ("Unsupported layout: '" + input_layout_.str() + "' for input 0 '" +
      this->spec_.InputName(0) + "'"));
    DALI_ENFORCE(input_layout_.ndim() == in_shape.sample_dim(),
      "Number of dimension in layout description does not match the number"
      " of dimensions in the input.");
    if (output_layout_.empty())
      output_layout_ = input_layout_;
    else
      DALI_ENFORCE(output_layout_.is_permutation_of(input_layout_),
        "The requested output layout is not a permutation of input layout.");

    int ndim = in_shape.sample_dim();
    int nsamples = in_shape.size();
    DALI_ENFORCE(ndim >= 3 && ndim <= 5,
      make_string("Unexpected number of dimensions: ", ndim));
    DALI_ENFORCE(input_layout_.ndim() == ndim);
    int spatial_ndim = ImageLayoutInfo::NumSpatialDims(input_layout_);
    DALI_ENFORCE(spatial_ndim == 2 || spatial_ndim == 3,
      "Only 2D or 3D images and sequences of images are supported");
    DALI_ENFORCE(ImageLayoutInfo::HasChannel(input_layout_),
      "This operator expects an explicit channel dimension, even for monochrome images");

    crop_attr_.ProcessArguments(ws);

    VALUE_SWITCH(ndim, Dims, CMN_NDIMS,
    (
      using Args = kernels::SliceFlipNormalizePermutePadArgs<Dims>;
      // We won't change the underlying type after the first allocation
      if (!kernel_sample_args_.has_value())
        kernel_sample_args_ = std::vector<Args>(nsamples);
      auto &kernel_sample_args = any_cast<std::vector<Args>&>(kernel_sample_args_);
      kernel_sample_args.clear();
      kernel_sample_args.reserve(nsamples);
      // Set internal info for each sample based on CropAttr
      for (int data_idx = 0; data_idx < nsamples; data_idx++) {
        auto crop_win_gen = crop_attr_.GetCropWindowGenerator(data_idx);
        assert(crop_win_gen);
        CropWindow crop_window = crop_win_gen(in_shape[data_idx], input_layout_);
        bool horizontal_flip = this->spec_.template GetArgument<int>("mirror", &ws, data_idx);
        ApplySliceBoundsPolicy(out_of_bounds_policy_, in_shape[data_idx], crop_window.anchor,
                               crop_window.shape);
        kernel_sample_args.emplace_back(
          detail::ToSliceFlipNormalizePermutePadArgs<Dims>(
            in_shape[data_idx], input_layout_, output_layout_, crop_window, horizontal_flip,
            pad_output_, make_cspan(mean_vec_), make_cspan(inv_std_vec_),
            make_cspan(fill_values_)));
      }
    ), DALI_FAIL(make_string("Not supported number of dimensions: ", ndim)););  // NOLINT
  }

  CropAttr crop_attr_;

  DALIDataType input_type_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;

  TensorLayout input_layout_;
  TensorLayout output_layout_;

  // Whether to pad output to 4 channels
  bool pad_output_;
  bool should_normalize_;
  std::vector<float> mean_vec_, inv_std_vec_;

  std::vector<float> fill_values_;
  OutOfBoundsPolicy out_of_bounds_policy_ = OutOfBoundsPolicy::Error;

  kernels::KernelManager kmgr_;
  any kernel_sample_args_;

  USE_OPERATOR_MEMBERS();
};



}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_CROP_MIRROR_NORMALIZE_H_
