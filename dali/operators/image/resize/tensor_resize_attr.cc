// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/util/axes_utils.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

DALI_SCHEMA(TensorResizeAttr)
    .AddOptionalArg<vector<float>>("sizes", R"code(Output sizes.

When `axes` is provided, the size values refer to the axes specified.
Note: Arguments `sizes` and `scales` are mutually exclusive.)code",
                                   {}, true)
    .AddOptionalArg<vector<float>>("scales", R"code(Scale factors.

The resulting output size is calculated as
``out_size = size_rounding(scale_factor * original_size)``.
See `size_rounding` for a list of supported rounding policies.

When `axes` is provided, the scale factor values refer to the axes specified.
Note: Arguments `sizes` and `scales` are mutually exclusive.)code",
                                   {}, true)
    .AddOptionalArg<float>("alignment", R"code(Determines the position of the ROI
when using scales (provided or calculated).

The real output size must be integral and may differ from "ideal" output size calculated as input
(or ROI) size multiplied by the scale factor. In that case, the output size is rounded (according
to `size_rounding` policy) and the input ROI needs to be adjusted to maintain the scale factor.
This parameter defines which relative point of the ROI should retain its position in the output.

This point is calculated as ``center = (1 - alignment) * roi_start + alignment * roi_end``.
Alignment 0.0 denotes alignment with the start of the ROI, 0.5 with the center of the region, and 1.0 with the end.
Note that when ROI is not specified, roi_start=0 and roi_end=input_size is assumed.

When using 0.5 (default), the resize operation has flip invariant properties (flipping after resizing is
mathematically equivalent to resizing after flipping).

The value of this argument contains as many elements as dimensions provided for
sizes/scales. If only one value is provided, it is applied to all dimensions.)code",
                                   std::vector<float>{0.5}, true)
    .AddOptionalArg<std::vector<int>>(
        "axes",
        R"code(Indices of dimensions that `sizes`, `scales`, `max_size`, `roi_start`, `roi_end` refer to.

Accepted range is [-ndim, ndim-1]. Negative indices are counted from the back.

By default, all dimensions are assumed. The `axis_names` and `axes` arguments are mutually exclusive.)code",
        nullptr)
    .AddOptionalArg<TensorLayout>(
        "axis_names",
        R"code(Names of the axes that `sizes`, `scales`, `max_size`, `roi_start`, `roi_end` refer to.

By default, all dimensions are assumed. The `axis_names` and `axes` arguments are mutually exclusive.)code",
        nullptr)
    .AddOptionalArg<std::string>("size_rounding",
                                 R"code(Determines the rounding policy when using scales.

Possible values are:
* | ``"round"`` - Rounds the resulting size to the nearest integer value, with halfway cases rounded away from zero.
* | ``"truncate"`` - Discards the fractional part of the resulting size.
* | ``"ceil"`` - Rounds up the resulting size to the next integer value.)code",
                                 "round")
    .AddParent("ResizeAttrBase");


TensorResizeAttr::TensorResizeAttr(const OpSpec &spec) : axes_helper_(spec) {
  has_sizes_ = spec.ArgumentDefined("sizes");
  has_scales_ = spec.ArgumentDefined("scales");
  has_max_size_ = spec.ArgumentDefined("max_size");
  has_mode_ = spec.ArgumentDefined("mode");
  has_alignment_ = spec.ArgumentDefined("alignment");
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

  DALI_ENFORCE((mode_ != ResizeMode::NotSmaller && mode_ != ResizeMode::NotLarger) || !has_scales_,
               "Providing ``scales`` is incompatible with not-smaller or not-larger modes");

  auto rounding = spec.GetArgument<std::string>("size_rounding");
  if (rounding == "round") {
    scale_round_fn_ = [](float x) {
      return round_int(x);
    };
  } else if (rounding == "truncate") {
    scale_round_fn_ = [](float x) {
      return static_cast<int>(x);
    };
  } else if (rounding == "ceil") {
    scale_round_fn_ = [](float x) {
      return static_cast<int>(std::ceil(x));
    };
  } else {
    DALI_FAIL(make_string(
        "``rounding`` value ", rounding,
        " is not supported. Supported values are \"round\", \"truncate\", or \"ceil\"."));
  }
}

void TensorResizeAttr::PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                                           const TensorListShape<> &input_shape,
                                           const TensorLayout &layout) {
  first_spatial_dim_ = 0;
  spatial_ndim_ = input_shape.sample_dim();
  if (spatial_ndim_ < 1)
    throw std::runtime_error("Input must be at least 1D");
  axes_helper_.PrepareAxes(layout, spatial_ndim_);
  span<int> axes = axes_helper_.Axes();
  int nsamples = input_shape.num_samples();
  int nargs = axes.size();

  auto set_values_axes = [&](std::vector<float>& values, const std::vector<float>& arg) {
    for (int i = 0; i < nsamples; i++) {
      for (int a = 0; a < nargs; a++) {
        int d = axes[a];
        values[d] = arg[i * nargs + a];
      }
    }
  };

  auto set_values_axes_default_const = [&](std::vector<float> &values,
                                           const std::vector<float> &arg, float constant = 0.0f) {
    values.resize(nsamples * spatial_ndim_, constant);
    set_values_axes(values, arg);
  };

  auto set_values_axes_default_input_size = [&](std::vector<float> &values,
                                                const std::vector<float> &arg) {
    values.resize(nsamples * spatial_ndim_);
    for (int i = 0; i < nsamples; i++) {
      span<const int64_t> in_sample_shape = input_shape.tensor_shape_span(i);
      for (int d = 0; d < spatial_ndim_; d++) {
        values[d] = in_sample_shape[d];
      }
    }
    set_values_axes(values, arg);
  };

  if (has_alignment_) {
    GetShapeLikeArgument<float>(alignment_arg_, spec, "alignment", ws, nsamples, nargs);
    set_values_axes_default_const(alignment_, alignment_arg_, 0.0f);
  }

  assert(has_sizes_ + has_scales_ == 1);
  if (has_sizes_) {
    GetShapeLikeArgument<float>(sizes_arg_, spec, "sizes", ws, nsamples, nargs);
    set_values_axes_default_input_size(sizes_, sizes_arg_);
  }
  if (has_scales_) {
    GetShapeLikeArgument<float>(scales_arg_, spec, "scales", ws, nsamples, nargs);
    set_values_axes_default_const(scales_, scales_arg_, 1.0f);
  }

  if (has_roi_) {
    GetShapeLikeArgument<float>(roi_start_arg_, spec, "roi_start", ws, nsamples, nargs);
    set_values_axes_default_const(roi_start_, roi_start_arg_, 0.0f);

    GetShapeLikeArgument<float>(roi_end_arg_, spec, "roi_end", ws, nsamples, nargs);
    if (roi_relative_)
      set_values_axes_default_const(roi_end_, roi_end_arg_, 1.0f);
    else
      set_values_axes_default_input_size(roi_end_, roi_end_arg_);
  }


  const float *max_size = nullptr;
  if (has_max_size_) {
    max_size_.resize(spatial_ndim_,
                     std::nextafter(static_cast<float>(std::numeric_limits<int>::max()), 0.0f));
    GetSingleOrRepeatedArg(spec, max_size_arg_, "max_size", nargs);
    for (int a = 0; a < nargs; a++) {
      int d = axes[a];
      max_size_[d] = max_size_arg_[a];
    }
    max_size = max_size_.data();
  }

  params_.resize(nsamples);
  auto roi_start = make_cspan(roi_start_);
  auto roi_end = make_cspan(roi_end_);

  SmallVector<float, 3> requested_size, in_lo, in_hi;
  requested_size.resize(spatial_ndim_);
  in_lo.resize(spatial_ndim_);
  in_hi.resize(spatial_ndim_);

  for (int i = 0; i < nsamples; i++) {
    auto in_sample_shape = input_shape.tensor_shape_span(i);

    if (has_sizes_) {
      for (int d = 0; d < spatial_ndim_; d++) {
        requested_size[d] = sizes_[i * spatial_ndim_ + d];
      }
    } else {
      assert(has_scales_);
      for (int d = 0; d < spatial_ndim_; d++) {
        requested_size[d] = scales_[i * spatial_ndim_ + d] * in_sample_shape[d];
      }
    }

    bool empty_input = volume(input_shape.tensor_shape_span(i)) == 0;
    CalculateInputRoI(in_lo, in_hi, has_roi_, roi_relative_, roi_start, roi_end,
                      input_shape, i, spatial_ndim_, first_spatial_dim_);

    span<const float> alignment;
    if (has_alignment_)
      alignment = {alignment_arg_.data() + i * spatial_ndim_, spatial_ndim_};
    assert(subpixel_scale_ || !has_alignment_);
    CalculateSampleParams(params_[i], requested_size, in_lo, in_hi, subpixel_scale_, empty_input,
                          spatial_ndim_, mode_, max_size, alignment, scale_round_fn_);
  }
}

}  // namespace dali
