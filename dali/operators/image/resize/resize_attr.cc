// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

This option is mutually exclusive with ``resize_shorter``, ``resize_longer`` and ``size``.
If the ``resize_y`` is unspecified or 0, the operator keeps the aspect ratio of the original image.
A negative value flips the image.)code", 0.f, true)
  .AddOptionalArg("resize_y", R"code(The length of the Y dimension of the resized image.

This option is mutually exclusive with ``resize_shorter``, ``resize_longer`` and ``size``.
If the ``resize_x`` is unspecified or 0, the operator keeps the aspect ratio of the original image.
A negative value flips the image.)code", 0.f, true)
  .AddOptionalArg("resize_z", R"code(The length of the Z dimension of the resized volume.

This option is mutually exclusive with ``resize_shorter``, ``resize_longer`` and ``size``.
If the ``resize_x`` and ``resize_y`` are left unspecified or 0, then the op will keep
the aspect ratio of the original volume. Negative value flips the volume.)code", 0.f, true)
  .AddOptionalArg<vector<float>>("size", R"code(The desired output size.

Must be a list/tuple with one entry per spatial dimension, excluding video frames and channels.
Dimensions with a 0 extent are treated as absent, and the output size will be calculated based on
other extents and ``mode`` argument.)code", {}, true)
  .AddOptionalArg("mode", R"code(Resize mode.

Here is a list of supported modes:

* | ``"default"`` - image is resized to the specified size.
  | Missing extents are scaled with the average scale of the provided ones.
* | ``"stretch"`` - image is resized to the specified size.
  | Missing extents are not scaled at all.
* | ``"not_larger"`` - image is resized, keeping the aspect ratio, so that no extent of the
    output image exceeds the specified size.
  | For example, a 1280x720, with a desired output size of 640x480, actually produces
    a 640x360 output.
* | ``"not_smaller"`` - image is resized, keeping the aspect ratio, so that no extent of the
    output image is smaller than specified.
  | For example, a 640x480 image with a desired output size of 1920x1080, actually produces
    a 1920x1440 output.

  This argument is mutually exclusive with ``resize_longer`` and ``resize_shorter``)code", "default")
  .AddOptionalArg("resize_shorter", R"code(The length of the shorter dimension of the resized image.

This option is mutually exclusive with ``resize_longer`` and explicit size arguments, and
the operator keeps the aspect ratio of the original image.
This option is equivalent to specifying the same size for all dimensions and ``mode="not_smaller"``.
The longer dimension can be bounded by setting the ``max_size`` argument.
See ``max_size`` argument doc for more info.)code", 0.f, true)
  .AddOptionalArg("resize_longer", R"code(The length of the longer dimension of the resized image.

This option is mutually exclusive with ``resize_shorter`` and explicit size arguments, and
the operator keeps the aspect ratio of the original image.
This option is equivalent to specifying the same size for all dimensions and ``mode="not_larger"``.
)code", 0.f, true)
  .AddOptionalArg<vector<float>>("max_size", R"code(Limit of the output size.

When the operator is configured to keep aspect ratio and only the smaller dimension is specified,
the other(s) can grow very large. This can happen when using ``resize_shorter`` argument
or "not_smaller" mode or when some extents are left unspecified.

This parameter puts a limit to how big the output can become. This value can be specified per-axis
or uniformly for all axes.

.. note::
  When used with "not_smaller" mode or ``resize_shorter`` argument, ``max_size`` takes
  precedence and the aspect ratio is kept - for example, resizing with
  ``mode="not_smaller", size=800, max_size=1400`` an image of size 1200x600 would be resized to
  1400x700.
)code", {}, false)
  .AddOptionalArg("subpixel_scale", R"code(If True, fractional sizes, directly specified or
calculated, will cause the input ROI to be adjusted to keep the scale factor.

Otherwise, the scale factor will be adjusted so that the source image maps to
the rounded output size.)code", true)
  .AddOptionalArg<vector<float>>("roi_start", R"code(Origin of the input region of interest (ROI).

Must be specified together with ``roi_end``. The coordinates follow the tensor shape order, which is
the same as ``size``. The coordinates can be either absolute (in pixels, which is the default) or
relative (0..1), depending on the value of ``relative_roi`` argument. If the ROI origin is greater
than the ROI end in any dimension, the region is flipped in that dimension.)code", nullptr, true)
  .AddOptionalArg<vector<float>>("roi_end", R"code(End of the input region of interest (ROI).

Must be specified together with ``roi_start``. The coordinates follow the tensor shape order, which is
the same as ``size``. The coordinates can be either absolute (in pixels, which is the default) or
relative (0..1), depending on the value of ``relative_roi`` argument. If the ROI origin is greater
than the ROI end in any dimension, the region is flipped in that dimension.)code", nullptr, true)
  .AddOptionalArg("roi_relative", R"code(If true, ROI coordinates are relative to the input size,
where 0 denotes top/left and 1 denotes bottom/right)code", false);


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

void ResizeAttr::CalculateInputRoI(SmallVector<float, 3> &in_lo,
                                   SmallVector<float, 3> &in_hi,
                                   const TensorListShape<> &input_shape,
                                   int sample_idx) const {
  in_lo.resize(spatial_ndim_);
  in_hi.resize(spatial_ndim_);
  static constexpr float min_size = 1e-3f;  // minimum size, in pixels
  auto *in_size = &input_shape.tensor_shape_span(sample_idx)[first_spatial_dim_];
  for (int d = 0; d < spatial_ndim_; d++) {
    if (has_roi_ && in_size[d] > 0) {
      double lo = roi_start_[spatial_ndim_ * sample_idx + d];
      double hi = roi_end_[spatial_ndim_ * sample_idx + d];
      if (roi_relative_) {
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


void ResizeAttr::PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                                     const TensorListShape<> &input_shape) {
  SetFlagsAndMode(spec);
  int N = input_shape.num_samples();
  params_.resize(N);

  if (has_size_) {
    GetShapeLikeArgument<float>(size_arg_, spec, "size", ws, N, spatial_ndim_);
  }

  max_size_.resize(spatial_ndim_,
                   std::nextafter(static_cast<float>(std::numeric_limits<int>::max()), 0.0f));
  if (has_max_size_) {
    GetSingleOrRepeatedArg(spec, max_size_, "max_size", spatial_ndim_);
  }

  if (has_roi_) {
    GetShapeLikeArgument<float>(roi_start_, spec, "roi_start", ws, N, spatial_ndim_);
    GetShapeLikeArgument<float>(roi_end_, spec, "roi_end", ws, N, spatial_ndim_);
  }

  SmallVector<float, 3> requested_size, in_lo, in_hi;
  requested_size.resize(spatial_ndim_);
  in_lo.resize(spatial_ndim_);
  in_hi.resize(spatial_ndim_);

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
      for (int d = 0; d < spatial_ndim_; d++) {
        requested_size[d] = size_vecs[d][i];
      }

      bool empty_input = volume(input_shape.tensor_shape_span(i)) == 0;
      CalculateInputRoI(in_lo, in_hi, input_shape, i);
      CalculateSampleParams(params_[i], requested_size, in_lo, in_hi, subpixel_scale_, empty_input);
    }
  } else if (has_resize_shorter_ || has_resize_longer_) {
    const char *arg_name = has_resize_shorter_ ? "resize_shorter" : "resize_longer";
    GetPerSampleArgument(res_x_, arg_name, spec, ws, N);

    for (int i = 0; i < N; i++) {
      for (int d = 0; d < spatial_ndim_; d++) {
        requested_size[d] = res_x_[i];
      }

      bool empty_input = volume(input_shape.tensor_shape_span(i)) == 0;
      CalculateInputRoI(in_lo, in_hi, input_shape, i);
      CalculateSampleParams(params_[i], requested_size, in_lo, in_hi, subpixel_scale_, empty_input);
    }
  } else if (has_size_) {
    for (int i = 0; i < N; i++) {
      auto in_sample_shape = input_shape.tensor_shape_span(i);
      for (int d = 0; d < spatial_ndim_; d++) {
        requested_size[d] = size_arg_[i * spatial_ndim_ + d];
      }

      bool empty_input = volume(input_shape.tensor_shape_span(i)) == 0;
      CalculateInputRoI(in_lo, in_hi, input_shape, i);
      CalculateSampleParams(params_[i], requested_size, in_lo, in_hi, subpixel_scale_, empty_input);
    }
  }
}

void ResizeAttr::AdjustOutputSize(float *out_size, const float *in_size) {
  SmallVector<double, 3> scale;
  SmallVector<bool, 3> mask;
  scale.resize(spatial_ndim_, 1);
  mask.resize(spatial_ndim_);

  int sizes_provided = 0;
  for (int d = 0; d < spatial_ndim_; d++) {
    mask[d] = (out_size[d] != 0 && in_size[d] != 0);
    scale[d] = in_size[d] ? out_size[d] / in_size[d] : 1;
    sizes_provided += mask[d];
  }

  if (sizes_provided == 0) {  // no clue how to resize - keep original size then
    for (int d = 0; d < spatial_ndim_; d++)
      out_size[d] = in_size[d];
  } else {
    if (mode_ == ResizeMode::Default) {
      if (sizes_provided < spatial_ndim_) {
        // if only some extents are provided - calculate average scale
        // and use it for the missing dimensions;
        double avg_scale = 1;
        for (int d = 0; d < spatial_ndim_; d++) {
          if (mask[d])
            avg_scale *= std::abs(scale[d]);  // abs because of possible flipping
        }
        if (sizes_provided > 1)
          avg_scale = std::pow(avg_scale, 1.0 / sizes_provided);
        for (int d = 0; d < spatial_ndim_; d++) {
          if (!mask[d]) {
            scale[d] = avg_scale;
            out_size[d] = in_size[d] * scale[d];
          }
        }
      }
      if (has_max_size_) {
        for (int d = 0; d < spatial_ndim_; d++) {
          if (max_size_[d] > 0 && std::abs(out_size[d]) > max_size_[d]) {
            out_size[d] = std::copysignf(max_size_[d], out_size[d]);
            scale[d] = out_size[d] / in_size[d];
          }
        }
      }
    } else if (mode_ == ResizeMode::Stretch) {
      if (sizes_provided < spatial_ndim_) {
        for (int d = 0; d < spatial_ndim_; d++) {
          if (!mask[d]) {
            scale[d] = 1;
            out_size[d] = in_size[d];
          }
        }
      }
      if (has_max_size_) {
        for (int d = 0; d < spatial_ndim_; d++) {
          if (max_size_[d] > 0 && std::abs(out_size[d]) > max_size_[d]) {
            out_size[d] = std::copysignf(max_size_[d], out_size[d]);
            scale[d] = out_size[d] / in_size[d];
          }
        }
      }
    } else {
      // NotLarger or NotSmaller mode

      // First, go through all the dimensions that have their scale defined and find
      // the min (or max) scale
      double final_scale = 0;
      bool first = true;
      for (int d = 0; d < spatial_ndim_; d++) {
        if (mask[d]) {
          float s = std::abs(scale[d]);
          if (first ||
              (mode_ == ResizeMode::NotSmaller && s > final_scale) ||
              (mode_ == ResizeMode::NotLarger && s < final_scale))
            final_scale = s;
          first = false;
        }
      }

      // If there's a size limit, apply it and possibly reduce the final scale
      if (has_max_size_) {
        for (int d = 0; d < spatial_ndim_; d++) {
          if (max_size_[d] > 0) {
            double s = static_cast<double>(max_size_[d]) / in_size[d];
            if (s < final_scale)
              final_scale = s;
          }
        }
      }

      // Now, let's apply the final scale to all dimensions - if final_scale is different (in
      // absolute value) than one defined by the caller, adjust it; also, if no scale was defined
      // for a dimension, then just use the final scale.
      for (int d = 0; d < spatial_ndim_; d++) {
        if (!mask[d] || std::abs(scale[d]) != final_scale) {
          scale[d] = std::copysign(final_scale, scale[d]);
          out_size[d] = in_size[d] * scale[d];
        }
      }
    }
  }
}

void ResizeAttr::CalculateSampleParams(ResizeParams &params,
                                       SmallVector<float, 3> requested_size,
                                       SmallVector<float, 3> in_lo,
                                       SmallVector<float, 3> in_hi,
                                       bool adjust_roi,
                                       bool empty_input) {
  assert(static_cast<int>(requested_size.size()) == spatial_ndim_);
  assert(static_cast<int>(in_lo.size()) == spatial_ndim_);
  assert(static_cast<int>(in_hi.size()) == spatial_ndim_);

  SmallVector<float, 3> in_size;
  in_size.resize(spatial_ndim_);
  for (int d = 0; d < spatial_ndim_; d++) {
    float sz = in_hi[d] - in_lo[d];
    if (sz < 0) {
      std::swap(in_hi[d], in_lo[d]);
      requested_size[d] = -requested_size[d];
      sz = -sz;
    }
    in_size[d] = sz;
  }

  AdjustOutputSize(requested_size.data(), in_size.data());

  for (int d = 0; d < spatial_ndim_; d++) {
    DALI_ENFORCE(in_lo[d] != in_hi[d] || requested_size[d] == 0,
                "Cannot produce non-empty output from empty input");
  }

  params.resize(spatial_ndim_);
  params.src_lo = in_lo;
  params.src_hi = in_hi;

  // If the input sample is empty, we simply can't produce _any_ non-empty output.
  // If ROI is degenerate but there's some input, we can sample it at the degenerate location.
  // To prevent empty outputs when we have some means of producing non-empty output, we bump
  // up the size of the output to at least 1 in each axis.
  int min_size = empty_input ? 0 : 1;

  for (int d = 0; d < spatial_ndim_; d++) {
    float out_sz = requested_size[d];
    bool flip = out_sz < 0;
    params.dst_size[d] = std::max(min_size, round_int(std::fabs(out_sz)));
    if (flip)
      std::swap(params.src_lo[d], params.src_hi[d]);

    // if rounded size differs from the requested fractional size, adjust input ROI
    if (adjust_roi && params.dst_size[d] != std::fabs(out_sz)) {
      double real_size = params.dst_size[d];
      double adjustment = real_size / std::fabs(out_sz);

      // This means that our output is 0.1 pixels - we might get inaccurate results
      // with 1x1 real output and small ROI, but it means that the user should use a proper ROI
      // and real output size instead.
      adjustment = clamp(adjustment, -10.0, 10.0);

      // keep center of the ROI - adjust the edges
      double center = (params.src_lo[d] + params.src_hi[d]) * 0.5;

      // clamp to more-or-less sane interval to avoid arithmetic problems downstream
      params.src_lo[d] = clamp(center + (params.src_lo[d] - center) * adjustment, -1e+9, 1e+9);
      params.src_hi[d] = clamp(center + (params.src_hi[d] - center) * adjustment, -1e+9, 1e+9);
    }
  }
}

}  // namespace dali
