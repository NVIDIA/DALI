// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/reduce/reduce_setup_utils.h"   // TODO(janton): #include "dali/kernels/common/utils.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

DALI_SCHEMA(TensorResizeAttr)
    .AddOptionalArg<vector<float>>("sizes", R"code(Output sizes.

When ``axes`` is provided, the size values refer to the axes specified.
Note: Arguments ``sizes`` and ``scales`` are mutually exclusive.)code",
                                   {}, true)
    .AddOptionalArg<vector<float>>("scales", R"code(Scale factors.

The resulting output size is calculated as
``out_size = size_rounding(scale_factor * original_size)``.
See ``size_rounding`` for a list of supported rounding policies.

When ``axes`` is provided, the scale factor values refer to the axes specified.
Note: Arguments ``sizes`` and ``scales`` are mutually exclusive.)code",
                                   {}, true)
    .AddOptionalArg<float>("alignment", R"code(Determines the position of the ROI
when using scales (provided or calculated).

The real output size must be integer and may differ from "ideal" output size calculated as input
(or ROI) size multiplied by the scale factor. In that case, the output size is rounded (according
to `size_rounding` policy) and the input ROI needs to be adjusted to maintain the scale factor.
This parameter defines which relative point of the ROI should retain its position in the output.

This point is calculated as ``center = (1 - alignment) * roi_start + alignment * roi_end``.
Alignment 0.0 denotes alignment with the start of the ROI, 0.5 with the center of the region, and 1.0 with the end.
Note that when ROI is not specified, roi_start=0 and roi_end=size is assumed.

When using 0.5 (default), the resize operation has flip invariant properties (flipping after resizing is
mathematically equivalent to resizing after flipping).

Contains as many elements as dimensions provided for sizes/scales. If only one value is provided, it is
applied to all dimensions.)code",
                                   std::vector<float>{0.5}, true)
    .AddOptionalArg(
        "axes",
        R"code(Indices of dimensions that `sizes`, `scales`, `max_size`, `roi_start`, `roi_end` refer to.

Accepted range is [-ndim, ndim-1]. Negative indices are counted from the back.

By default, all dimensions are assumed.)code",
        std::vector<int>{})
    .AddOptionalArg<std::string>("size_rounding",
                                 R"code(Determines the rounding policy when using scales.

Possible values are:
* | ``"round"`` - Rounds the resulting size to the nearest integer value, with halfway cases rounded away from zero.
* | ``"truncate"`` - Discards the fractional part of the resulting size.
* | ``"ceil"`` - Rounds up the resulting size to the next integer value.)code",
                                 "round")
    .AddParent("ResizeAttrBase");


TensorResizeAttr::TensorResizeAttr(const OpSpec &spec) {
  has_sizes_ = spec.ArgumentDefined("sizes");
  has_scales_ = spec.ArgumentDefined("scales");
  has_max_size_ = spec.ArgumentDefined("max_size");
  has_mode_ = spec.ArgumentDefined("mode");
  has_axes_ = spec.ArgumentDefined("axes");
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

  if (has_axes_) {
    axes_ = spec.GetRepeatedArgument<int>("axes");
  }

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

template <typename T>
void GetShapeLikeWithAxes(std::vector<T> &full, std::vector<T> &arg,
                          const OpSpec& spec, const ArgumentWorkspace &ws,
                          const std::string &arg_name, span<const int> axes,
                          int nsamples, int spatial_ndim, int nargs,
                          int first_spatial_dim) {
  GetShapeLikeArgument<T>(arg, spec, arg_name, ws, nsamples, nargs);
  full.resize(nsamples * spatial_ndim);
  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    auto orig_arg = make_span(arg.data() + sample_idx * nargs, nargs);
    auto full_arg = make_span(full.data() + sample_idx * spatial_ndim, spatial_ndim);
    for (int i = 0; i < nargs; i++) {
      int d = axes[i] - first_spatial_dim;
      assert(d >= 0);
      assert(d < spatial_ndim);
      full_arg[d] = orig_arg[i];
    }
  }
}

void TensorResizeAttr::PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                                           const TensorListShape<> &input_shape) {
  ndim_ = input_shape.sample_dim();
  if (ndim_ < 1)
    throw std::invalid_argument("Expected at least 1D inputs");
  if (!has_axes_) {
    axes_.resize(ndim_);
    std::iota(axes_.begin(), axes_.end(), 0);
  }
  // TODO(janton): use utils
  kernels::reduce_impl::CheckAxes(make_cspan(axes_), ndim_);
  kernels::reduce_impl::AdjustAxes(make_span(axes_), ndim_);

  int nargs = axes_.size();
  int first_axis = ndim_;
  int last_axis = -1;
  for (int axis : axes_) {
    if (axis < first_axis)
      first_axis = axis;
    if (axis > last_axis)
      last_axis = axis;
  }
  spatial_ndim_ = last_axis - first_axis + 1;
  first_spatial_dim_ = first_axis;

  int nsamples = input_shape.num_samples();
  params_.resize(nsamples);

  auto axes = make_cspan(axes_);

  if (has_alignment_) {
    GetShapeLikeWithAxes<float>(alignment_, alignment_arg_, spec, ws, "alignment", axes, nsamples,
                                spatial_ndim_, nargs, first_spatial_dim_);
  }

  if (has_sizes_) {
    GetShapeLikeWithAxes<float>(sizes_, sizes_arg_, spec, ws, "sizes", axes, nsamples,
                                spatial_ndim_, nargs, first_spatial_dim_);
  } else if (has_scales_) {
    GetShapeLikeWithAxes<float>(scales_, scales_arg_, spec, ws, "scales", axes, nsamples,
                                spatial_ndim_, nargs, first_spatial_dim_);
  } else {
    assert(false);  // should not happen
  }

  if (has_roi_) {
    GetShapeLikeWithAxes<float>(roi_start_, roi_start_arg_, spec, ws, "roi_start", axes, nsamples,
                                spatial_ndim_, nargs, first_spatial_dim_);
    GetShapeLikeWithAxes<float>(roi_end_, roi_end_arg_, spec, ws, "roi_end", axes, nsamples,
                                spatial_ndim_, nargs, first_spatial_dim_);
  }

  max_size_.resize(spatial_ndim_,
                   std::nextafter(static_cast<float>(std::numeric_limits<int>::max()), 0.0f));
  if (has_max_size_) {
    GetSingleOrRepeatedArg(spec, max_size_arg_, "max_size", nargs);
    for (int i = 0; i < nargs; i++) {
      int d = axes_[i] - first_spatial_dim_;
      assert(d >= 0);
      assert(d < spatial_ndim_);
      max_size_[d] = max_size_arg_[i];
    }
  }

  SmallVector<float, 3> requested_size, in_lo, in_hi;
  requested_size.resize(spatial_ndim_);
  in_lo.resize(spatial_ndim_);
  in_hi.resize(spatial_ndim_);

  assert(has_sizes_ + has_scales_ == 1);
  const float *max_size = has_max_size_ ? max_size_.data() : nullptr;
  for (int i = 0; i < nsamples; i++) {
    auto in_sample_shape = input_shape.tensor_shape_span(i);
    if (has_sizes_) {
      for (int d = 0; d < spatial_ndim_; d++) {
        requested_size[d] = sizes_[i * ndim_ + first_spatial_dim_ + d];
      }
    } else {
      assert(has_scales_);
      for (int d = 0; d < ndim_; d++) {
        requested_size[d] = scales_[i * ndim_ + first_spatial_dim_ + d] * in_sample_shape[d];
      }
    }

    bool empty_input = volume(input_shape.tensor_shape_span(i)) == 0;
    CalculateInputRoI(in_lo, in_hi, has_roi_, roi_relative_, make_cspan(roi_start_),
                      make_cspan(roi_end_), input_shape, i, spatial_ndim_, first_spatial_dim_);

    span<const float> alignment;
    if (has_alignment_)
      alignment = {alignment_.data() + i * spatial_ndim_, spatial_ndim_};
    assert(subpixel_scale_ || !has_alignment_);
    CalculateSampleParams(params_[i], requested_size, in_lo, in_hi, subpixel_scale_, empty_input,
                          spatial_ndim_, mode_, max_size, alignment, scale_round_fn_);
  }

  auto unchanged_dim = [&](int d) {
    for (int i = 0; i < nsamples; i++) {
      int64_t extent = input_shape.tensor_shape_span(i)[d];
      if (static_cast<int64_t>(params_[i].dst_size[d]) != extent ||
          static_cast<int64_t>(params_[i].src_lo[d]) != 0 ||
          static_cast<int64_t>(params_[i].src_hi[d]) != extent) {
        return false;
      }
    }
    return true;
  };

  int trim_start_ndim = 0;
  int trim_end_ndim = 0;
  for (int d = 0; d < spatial_ndim_; d++) {
    if (!unchanged_dim(d))
      break;
    trim_start_ndim++;
  }
  for (int d = spatial_ndim_ - 1; d >= 0; d--) {
    if (!unchanged_dim(d))
      break;
    trim_end_ndim++;
  }

  int new_spatial_ndim = spatial_ndim_ - (trim_start_ndim + trim_end_ndim);
  for (int s = 0; s < nsamples; s++) {
    auto &p = params_[s];
    for (int d = 0; d < new_spatial_ndim ; d++) {
      int orig_d = trim_start_ndim + d;
      p.dst_size[d] = p.dst_size[orig_d];
      p.src_hi[d] = p.src_hi[orig_d];
      p.src_lo[d] = p.src_lo[orig_d];
    }
    p.dst_size.resize(new_spatial_ndim);
    p.src_hi.resize(new_spatial_ndim);
    p.src_lo.resize(new_spatial_ndim);
  }
  first_spatial_dim_ += trim_start_ndim;
  spatial_ndim_ = new_spatial_ndim;
}

}  // namespace dali
