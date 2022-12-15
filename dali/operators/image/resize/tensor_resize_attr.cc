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

#include "dali/operators/image/resize/tensor_resize_attr.h"
#include <limits>
#include <string>
#include <vector>
#include "dali/core/math_util.h"
#include "dali/operators/image/resize/resize_attr.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

DALI_SCHEMA(TensorResizeAttr)
  .AddOptionalArg<vector<float>>("sizes", R"code(Output sizes.

When ``axes`` is provided, the size values refer to the axes specified.
Note: Arguments ``sizes`` and ``scales`` are mutually exclusive.)code", {}, true)
  .AddOptionalArg<vector<float>>("scales", R"code(Scale factors.

Output size is calculated as ``out_size = size_rounding(scale_factor * original_size)``.
See ``size_rounding`` for a list of supported rounding policies.

When ``axes`` is provided, the scale factor values refer to the axes specified.
Note: Arguments ``sizes`` and ``scales`` are mutually exclusive.)code", {}, true)
  .AddOptionalArg("axes", R"code(Indices of dimensions that `sizes`, `scales`, `max_size`, `roi_start`, `roi_end` refer to.

By default, all dimensions are assumed.)code", std::vector<int>{})
  .AddOptionalArg<std::string>("size_rounding", R"code(Determines the rounding policy when using scales.

Possible values are:
* | ``"round"`` - Rounds the resulting size to the nearest integer value, with halfway cases rounded away from zero.
* | ``"truncate"`` - Discards the fractional part of the resulting size.
* | ``"ceil"`` - Rounds up the resulting size to the next integer value.)code", "truncate")
  .AddParent("ResizeAttrBase");


void TensorResizeAttr::SetFlagsAndMode(const OpSpec &spec) {
  has_sizes_ = spec.ArgumentDefined("sizes");
  has_scales_ = spec.ArgumentDefined("scales");
  has_max_size_ = spec.ArgumentDefined("max_size");
  has_mode_ = spec.ArgumentDefined("mode");
  has_axes_ = spec.ArgumentDefined("axes");
  subpixel_scale_ = spec.GetArgument<bool>("subpixel_scale");

  DALI_ENFORCE(has_scales_ + has_sizes_ == 1, "Need one of ``scales`` or ``sizes``, but not both");

  bool has_roi_start = spec.ArgumentDefined("roi_start");
  bool has_roi_end = spec.ArgumentDefined("roi_end");
  DALI_ENFORCE(has_roi_start == has_roi_end,
               "``roi_start`` and ``roi_end`` must be specified together");
  has_roi_ = has_roi_start && has_roi_end;
  roi_relative_ = spec.GetArgument<bool>("roi_relative");

  if (has_mode_) {
    mode_ = ParseResizeMode(spec.GetArgument<std::string>("mode"));
  } else {
    mode_ = ResizeMode::Default;
  }

  if (has_axes_)
    axes_ = spec.GetRepeatedArgument<int>("axes");

  DALI_ENFORCE((mode_ != ResizeMode::NotSmaller && mode_ != ResizeMode::NotLarger) || !has_scales_,
               "Providing ``scales`` is incompatible with not-smaller or not-larger modes");
}

void TensorResizeAttr::PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                                           const TensorListShape<> &input_shape) {
  SetFlagsAndMode(spec);

  ndim_ = input_shape.sample_dim();
  if (ndim_ < 1)
    throw std::invalid_argument("Expected at least 1D inputs");
  if (!has_axes_) {
    axes_.resize(ndim_);
    std::iota(axes_.begin(), axes_.end(), 0);
  }
  int nargs = axes_.size();

  int nsamples = input_shape.num_samples();
  params_.resize(nsamples);

  auto get_shape_like_with_axes = [&](std::vector<float> &full, std::vector<float> &arg,
                                      const std::string &arg_name) {
    GetShapeLikeArgument<float>(arg, spec, arg_name, ws, nsamples, nargs);
    full.resize(nsamples * ndim_);
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
      auto orig_arg = make_span(arg.data() + sample_idx * nargs, nargs);
      auto full_arg = make_span(full.data() + sample_idx * ndim_, ndim_);
      for (int i = 0; i < nargs; i++) {
        int d = axes_[i];
        assert(d < ndim_);
        full_arg[d] = orig_arg[i];
      }
    }
  };

  if (has_sizes_) {
    get_shape_like_with_axes(sizes_, sizes_arg_, "sizes");
  } else if (has_scales_) {
    get_shape_like_with_axes(scales_, scales_arg_, "scales");
  } else {
    assert(false);  // should not happen
  }

  if (has_roi_) {
    get_shape_like_with_axes(roi_start_, roi_start_arg_, "roi_start");
    get_shape_like_with_axes(roi_end_, roi_end_arg_, "roi_end");
  }

  spatial_ndim_ = ndim_;
  first_spatial_dim_ = 0;

  max_size_.resize(ndim_,
                   std::nextafter(static_cast<float>(std::numeric_limits<int>::max()), 0.0f));
  if (has_max_size_) {
    GetSingleOrRepeatedArg(spec, max_size_arg_, "max_size", nargs);
    for (int i = 0; i < nargs; i++) {
      int d = axes_[i];
      max_size_[d] = max_size_arg_[i];
    }
  }

  SmallVector<float, 3> requested_size, in_lo, in_hi;
  requested_size.resize(ndim_);
  in_lo.resize(ndim_);
  in_hi.resize(ndim_);

  assert(has_sizes_ + has_scales_ == 1);
  const float *max_size = has_max_size_ ? max_size_.data() : nullptr;
  for (int i = 0; i < nsamples; i++) {
    auto in_sample_shape = input_shape.tensor_shape_span(i);
    if (has_sizes_) {
      for (int d = 0; d < ndim_; d++) {
        requested_size[d] = sizes_[i * ndim_ + d];
      }
    } else {
      assert(has_scales_);
      for (int d = 0; d < ndim_; d++) {
        requested_size[d] =
            scale_round_fn_(static_cast<double>(scales_[i * ndim_ + d]) * in_sample_shape[d]);
      }
    }

    bool empty_input = volume(input_shape.tensor_shape_span(i)) == 0;
    CalculateInputRoI(in_lo, in_hi, has_roi_, roi_relative_, roi_start_.data(), roi_end_.data(),
                      input_shape, i, ndim_, 0);
    CalculateSampleParams(params_[i], requested_size, in_lo, in_hi, subpixel_scale_, empty_input,
                          ndim_, mode_, max_size);
  }
}

}  // namespace dali
