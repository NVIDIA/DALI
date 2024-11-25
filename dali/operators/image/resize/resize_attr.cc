// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/resize/resize_attr.h"
#include <limits>
#include <string>
#include <vector>
#include "dali/pipeline/operator/common.h"
#include "dali/core/math_util.h"

namespace dali {

DALI_SCHEMA(ResizeAttr)
  .AddOptionalArg("resize_x", R"code(The length of the X dimension of the resized image.

This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_y` is unspecified or 0, the operator keeps the aspect ratio of the original image.
A negative value flips the image.)code", 0.f, true)
  .AddOptionalArg("resize_y", R"code(The length of the Y dimension of the resized image.

This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_x` is unspecified or 0, the operator keeps the aspect ratio of the original image.
A negative value flips the image.)code", 0.f, true)
  .AddOptionalArg("resize_z", R"code(The length of the Z dimension of the resized volume.

This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_x` and `resize_y` are left unspecified or 0, then the op will keep
the aspect ratio of the original volume. Negative value flips the volume.)code", 0.f, true)
  .AddOptionalArg<vector<float>>("size", R"code(The desired output size.

Must be a list/tuple with one entry per spatial dimension, excluding video frames and channels.
Dimensions with a 0 extent are treated as absent, and the output size will be calculated based on
other extents and `mode` argument.)code", {}, true)
  .AddOptionalArg("resize_shorter", R"code(The length of the shorter dimension of the resized image.

This option is mutually exclusive with `resize_longer` and explicit size arguments, and
the operator keeps the aspect ratio of the original image.
This option is equivalent to specifying the same size for all dimensions and ``mode="not_smaller"``.
The longer dimension can be bounded by setting the `max_size` argument.
See `max_size` argument doc for more info.)code", 0.f, true)
  .AddOptionalArg("resize_longer", R"code(The length of the longer dimension of the resized image.

This option is mutually exclusive with `resize_shorter` and explicit size arguments, and
the operator keeps the aspect ratio of the original image.
This option is equivalent to specifying the same size for all dimensions and ``mode="not_larger"``.
)code", 0.f, true)
  .AddParent("ResizeAttrBase");


void ResizeAttr::SetFlagsAndMode(const OpSpec &spec) {
  has_resize_shorter_ = spec.ArgumentDefined("resize_shorter");
  has_resize_longer_ = spec.ArgumentDefined("resize_longer");
  has_resize_x_ = spec.ArgumentDefined("resize_x");
  has_resize_y_ = spec.ArgumentDefined("resize_y");
  has_resize_z_ = spec.ArgumentDefined("resize_z");
  has_size_ = spec.ArgumentDefined("size");
  has_max_size_ = spec.ArgumentDefined("max_size");
  has_mode_ = spec.ArgumentDefined("mode");

  subpixel_scale_ = spec.GetArgument<bool>("subpixel_scale");

  DALI_ENFORCE(HasSeparateSizeArgs() + has_size_ + has_resize_shorter_ + has_resize_longer_ == 1,
    R"(Exactly one method of specifying size must be used. The available methods:
    - separate resize_x, resize_y, resize_z arguments
    - size argument
    - resize_longer
    - resize_shorter)");

  DALI_ENFORCE(has_resize_shorter_ + has_resize_longer_ + has_mode_ <= 1,
    "`resize_shorter`, ``resize_longer`` and ``mode`` arguments are mutually exclusive");

  bool has_roi_start = spec.ArgumentDefined("roi_start");
  bool has_roi_end = spec.ArgumentDefined("roi_end");
  DALI_ENFORCE(has_roi_start == has_roi_end,
               "``roi_start`` and ``roi_end`` must be specified together");
  has_roi_ = has_roi_start && has_roi_end;
  roi_relative_ = spec.GetArgument<bool>("roi_relative");

  if (has_resize_shorter_) {
    mode_ = ResizeMode::NotSmaller;
  } else if (has_resize_longer_) {
    mode_ = ResizeMode::NotLarger;
  } else if (has_mode_) {
    mode_ = ParseResizeMode(spec.GetArgument<std::string>("mode"));
  } else {
    mode_ = ResizeMode::Default;
  }
}

void ResizeAttr::ParseLayout(
      int &spatial_ndim, int &first_spatial_dim, const TensorLayout &layout) {
  spatial_ndim = ImageLayoutInfo::NumSpatialDims(layout);

  int i = 0;
  for (; i < layout.ndim(); i++) {
    if (ImageLayoutInfo::IsSpatialDim(layout[i]))
      break;
  }
  int spatial_dims_begin = i;

  for (; i < layout.ndim(); i++) {
    if (!ImageLayoutInfo::IsSpatialDim(layout[i]))
      break;
  }

  int spatial_dims_end = i;
  DALI_ENFORCE(spatial_dims_end - spatial_dims_begin == spatial_ndim, make_string(
    "Spatial dimensions must be adjacent (as in HWC layout). Got: ", layout));

  first_spatial_dim = spatial_dims_begin;
}

void CalculateInputRoI(SmallVector<float, 3> &in_lo, SmallVector<float, 3> &in_hi, bool has_roi,
                       bool roi_relative, span<const float> roi_start, span<const float> roi_end,
                       const TensorListShape<> &input_shape, int sample_idx, int spatial_ndim,
                       int first_spatial_dim) {
  in_lo.resize(spatial_ndim);
  in_hi.resize(spatial_ndim);
  assert(roi_start.size() == roi_end.size());
  assert(!has_roi || roi_start.size() >= spatial_ndim * (sample_idx + 1));
  static constexpr float min_size = 1e-3f;  // minimum size, in pixels
  auto *in_size = &input_shape.tensor_shape_span(sample_idx)[first_spatial_dim];
  for (int d = 0; d < spatial_ndim; d++) {
    if (has_roi && in_size[d] > 0) {
      double lo = roi_start[spatial_ndim * sample_idx + d];
      double hi = roi_end[spatial_ndim * sample_idx + d];
      if (roi_relative) {
        lo *= in_size[d];
        hi *= in_size[d];
      }
      // if input ROI is too small (e.g. due to numerical error), but the input is not empty,
      // we can stretch the ROI a bit to avoid division by 0 - this will possibly stretch just
      // a single pixel to the entire output, but it's better than throwing
      if (std::fabs(hi - lo) < min_size) {
        float offset = lo <= hi ? 0.5f * min_size : -0.5f * min_size;
        lo -= offset;
        hi += offset;
      }
      in_lo[d] = lo;
      in_hi[d] = hi;
    } else {
      in_lo[d] = 0;
      in_hi[d] = in_size[d];
    }
  }
}

void ResizeAttr::GetMaxSize(const OpSpec &spec, const ArgumentWorkspace &ws) {
  max_size_.resize(spatial_ndim_,
                   std::nextafter(static_cast<float>(std::numeric_limits<int>::max()), 0.0f));
  if (has_max_size_) {
    GetSingleOrRepeatedArg(spec, max_size_, "max_size", spatial_ndim_);
  }
}

void ResizeAttr::GetRoI(const OpSpec &spec,
                        const ArgumentWorkspace &ws,
                        const TensorListShape<> input_shape) {
  if (has_roi_) {
    const int N = batch_size_;
    GetShapeLikeArgument<float>(roi_start_, spec, "roi_start", ws, N, spatial_ndim_);
    GetShapeLikeArgument<float>(roi_end_, spec, "roi_end", ws, N, spatial_ndim_);
  }
}

void ResizeAttr::PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                                     const TensorListShape<> &input_shape) {
  SetFlagsAndMode(spec);
  batch_size_ = input_shape.num_samples();
  const int N = batch_size_;
  params_.resize(N);

  GetRoI(spec, ws, input_shape);
  GetRequestedOutputSize(spec, ws);
  GetMaxSize(spec, ws);

  SmallVector<float, 3> requested_size, in_lo, in_hi;
  requested_size.resize(spatial_ndim_);
  in_lo.resize(spatial_ndim_);
  in_hi.resize(spatial_ndim_);

  const float *max_size = has_max_size_ ? max_size_.data() : nullptr;

  for (int i = 0; i < N; i++) {
    auto in_sample_shape = input_shape.tensor_shape_span(i);
    for (int d = 0; d < spatial_ndim_; d++) {
      requested_size[d] = requested_output_size_[i * spatial_ndim_ + d];
    }

    bool empty_input = volume(input_shape.tensor_shape_span(i)) == 0;
    CalculateInputRoI(in_lo, in_hi, has_roi_, roi_relative_, make_cspan(roi_start_),
                      make_cspan(roi_end_), input_shape, i, spatial_ndim_, first_spatial_dim_);
    CalculateSampleParams(params_[i], requested_size, in_lo, in_hi, subpixel_scale_, empty_input,
                          spatial_ndim_, mode_, max_size);
  }
}

void ResizeAttr::GetRequestedOutputSize(const OpSpec &spec, const ArgumentWorkspace &ws) {
  const int N = batch_size_;
  requested_output_size_.resize(N * spatial_ndim_);

  if (HasSeparateSizeArgs()) {
    res_x_.resize(N);
    res_y_.resize(N);
    res_z_.resize(N);

    float *size_vecs[3] = { nullptr, nullptr, nullptr };
    {
      // outer-first shape order :(
      int d = 0;
      if (spatial_ndim_ >= 3)
        size_vecs[d++] = res_z_.data();
      if (spatial_ndim_ >= 2)
        size_vecs[d++] = res_y_.data();
      size_vecs[d++] = res_x_.data();
    }

    if (has_resize_x_) {
      GetPerSampleArgument(res_x_, "resize_x", spec, ws, N);
    }

    if (has_resize_y_) {
      GetPerSampleArgument(res_y_, "resize_y", spec, ws, N);
    }

    if (has_resize_z_) {
      GetPerSampleArgument(res_z_, "resize_z", spec, ws, N);
    }

    for (int i = 0; i < N; i++) {
      for (int d = 0; d < spatial_ndim_; d++)
        requested_output_size_[spatial_ndim_ * i + d] = size_vecs[d][i];
    }
  } else if (has_resize_shorter_ || has_resize_longer_) {
    const char *arg_name = has_resize_shorter_ ? "resize_shorter" : "resize_longer";
    GetPerSampleArgument(res_x_, arg_name, spec, ws, N);

    for (int i = 0; i < N; i++)
      for (int d = 0; d < spatial_ndim_; d++)
        requested_output_size_[spatial_ndim_ * i + d] = res_x_[i];
  } else {
    assert(has_size_);
    GetShapeLikeArgument<float>(requested_output_size_, spec, "size", ws, N, spatial_ndim_);
  }
}


}  // namespace dali
