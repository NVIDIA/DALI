// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
  .AddOptionalArg("resize_x", R"(The length of the X dimension of the resized image.
This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_y` is left unspecified or 0, then the op will keep
the aspect ratio of the original image. Negative value flips the image.)", 0.f, true)
  .AddOptionalArg("resize_y", R"(The length of the Y dimension of the resized image.
This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_x` is left unspecified or 0, then the op will keep
the aspect ratio of the original image. Negative value flips the image.)", 0.f, true)
  .AddOptionalArg("resize_z", R"(The length of the Z dimension of the resized volume.
This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_x` and `resize_y` are left unspecified or 0, then the op will keep
the aspect ratio of the original volume. Negative value flips the volume.)", 0.f, true)
  .AddOptionalArg<vector<float>>("size", R"(The desired output size. Must be a list/tuple with the
one entry per spatial dimension (i.e. excluding video frames and channels). Dimensions with
0 extent are treated as absent and the output size will be calculated based on other extents
and ``mode`` argument.)", {}, true)
  .AddOptionalArg("mode", R"(Resize mode - one of:
  * "stretch"     - image is resized to the specified size; aspect ratio is not kept
  * "not_larger"  - image is resized, keeping the aspect ratio, so that no extent of the
                    output image exceeds the specified size - e.g. a 1280x720 with desired output
                    size of 640x480 will actually produce 640x360 output.
  * "not_smaller" - image is resized, keeping the aspect ratio, so that no extent of the
                    output image is smaller than specified - e.g. 640x480 image with desired output
                    size of 1920x1080 will actually produce 1920x1440 output.

  This argument is mutually exclusive with `resize_longer` and `resize_shorter`)", "stretch")
  .AddOptionalArg("resize_shorter", R"(The length of the shorter dimension of the resized image.
This option is mutually exclusive with `resize_longer` and explicit size arguments
The op will keep the aspect ratio of the original image.
This option is equivalent to specifying the same size for all dimensions and ``mode="not_smaller"``.
The longer dimension can be bounded by setting the `max_size` argument.
See `max_size` argument doc for more info.)", 0.f, true)
  .AddOptionalArg("resize_longer", R"(The length of the longer dimension of the resized image.
This option is mutually exclusive with `resize_shorter` and explicit size arguments
The op will keep the aspect ratio of the original image.
This option is equivalent to specifying the same size for all dimensions and ``mode="not_larger"``.
)", 0.f, true)
  .AddOptionalArg<vector<float>>("max_size", R"(Limit of the output size - when resizing using
`resize_shorter`, "not_smaller" mode or otherwise leaving some extents unspecified, some images
with high aspect ratios might produce extremely large outputs. This parameter puts a limit to how
big the output can become. This value can be specified per-axis of uniformly for all axes.)",
  {}, false);

void ResizeAttr::SetFlagsAndMode(const OpSpec &spec) {
  has_resize_shorter_ = spec.ArgumentDefined("resize_shorter");
  has_resize_longer_ = spec.ArgumentDefined("resize_longer");
  has_resize_x_ = spec.ArgumentDefined("resize_x");
  has_resize_y_ = spec.ArgumentDefined("resize_y");
  has_resize_z_ = spec.ArgumentDefined("resize_z");
  has_size_ = spec.ArgumentDefined("size");
  has_max_size_ = spec.ArgumentDefined("max_size");
  has_mode_ = spec.ArgumentDefined("mode");

  DALI_ENFORCE(HasSeparateSizeArgs() + has_size_ + has_resize_shorter_ + has_resize_longer_ == 1,
    R"(Exactly one method of specifying size must be used. The available methods:
    - separate resize_x, resize_y, resize_z arguments
    - size argument
    - resize_longer
    - resize_shorter)");

  DALI_ENFORCE(has_resize_shorter_ + has_resize_longer_ + has_mode_ <= 1,
    "`resize_shorter`, `resize_longer` and `mode` arguments are mutually exclusive");

  if (has_resize_shorter_) {
    mode_ = ResizeMode::NotSmaller;
  } else if (has_resize_longer_) {
    mode_ = ResizeMode::NotLarger;
  } else if (has_mode_) {
    mode_ = ParseResizeMode(spec.GetArgument<std::string>("mode"));
  } else {
    mode_ = ResizeMode::Stretch;
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

void ResizeAttr::PrepareResizeParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                                     const TensorListShape<> &input_shape) {
  SetFlagsAndMode(spec);
  int N = input_shape.num_samples();
  params_.resize(N);

  if (has_size_) {
    GetShapeArgument<float>(size_arg_, spec, "size", ws, spatial_ndim_, N);
  }

  max_size_.resize(spatial_ndim_, std::numeric_limits<int>::max());
  if (has_max_size_) {
    GetSingleOrRepeatedArg(spec, max_size_, "max_size", spatial_ndim_);
  }

  SmallVector<float, 3> requested_size, in_size;
  requested_size.resize(spatial_ndim_);
  in_size.resize(spatial_ndim_);

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
        in_size[d] = input_shape.tensor_shape_span(i)[d + first_spatial_dim_];
        requested_size[d] = size_vecs[d][i];
      }

      CalculateSampleParams(params_[i], requested_size, in_size);
    }
  } else if (has_resize_shorter_ || has_resize_longer_) {
    const char *arg_name = has_resize_shorter_ ? "resize_shorter" : "resize_longer";
    GetPerSampleArgument(res_x_, arg_name, spec, ws, N);

    for (int i = 0; i < N; i++) {
      for (int d = 0; d < spatial_ndim_; d++) {
        in_size[d] = input_shape.tensor_shape_span(i)[d + first_spatial_dim_];
        requested_size[d] = res_x_[i];
      }

      CalculateSampleParams(params_[i], requested_size, in_size);
    }
  } else if (has_size_) {
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < spatial_ndim_; d++) {
        in_size[d] = input_shape.tensor_shape_span(i)[d + first_spatial_dim_];
        requested_size[d] = size_arg_.tensor_shape_span(i)[d];
      }

      CalculateSampleParams(params_[i], requested_size, in_size);
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
    mask[d] = (out_size[d] != 0);
    scale[d] = in_size[d] ? out_size[d] / in_size[d] : 1;
    sizes_provided += mask[d];
  }

  if (sizes_provided == 0) {  // no clue how to resize - keep original size then
    for (int d = 0; d < spatial_ndim_; d++)
      out_size[d] = in_size[d];
  } else {
    if (mode_ == ResizeMode::Stretch) {
      if (sizes_provided < spatial_ndim_) {
        // if only some extents are provided - calculate average scale
        // and use it for the missing dimensions;
        double avg_scale = 1;
        for (int d = 0; d < spatial_ndim_; d++) {
          scale[d] = out_size[d] / in_size[d];
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

      // Now, let's aply the final scale to all dimensions - if final_scale is different (in
      // absolute value) than one defined by the caller, adjust it; also, if no scale was defined
      // for a dimension, then just use the final scale.
      for (int d = 0; d < spatial_ndim_; d++) {
        if (!mask[d]) {
          out_size[d] = in_size[d] * final_scale;
          scale[d] = final_scale;
        } else if (std::abs(scale[d]) != final_scale) {
          scale[d] = std::copysign(final_scale, scale[d]);
          out_size[d] = in_size[d] * scale[d];
        }
      }
    }
  }
}

void ResizeAttr::CalculateSampleParams(ResizeParams &params,
                                       SmallVector<float, 3> requested_size,
                                       SmallVector<float, 3> input_size) {
  assert(static_cast<int>(requested_size.size()) == spatial_ndim_);
  assert(static_cast<int>(input_size.size()) == spatial_ndim_);


  for (int d = 0; d < spatial_ndim_; d++) {
    DALI_ENFORCE(input_size[d] != 0 || requested_size[d] == 0,
                "Cannot produce non-empty output from empty input");
  }

  AdjustOutputSize(requested_size.data(), input_size.data());

  params.resize(spatial_ndim_);

  for (int d = 0; d < spatial_ndim_; d++) {
    float out_sz = requested_size[d];
    bool flip = out_sz < 0;
    params.dst_size[d] = std::max(1, round_int(std::fabs(out_sz)));
    params.src_lo[d] = 0;
    params.src_hi[d] = input_size[d];
    if (flip)
      std::swap(params.src_lo[d], params.src_hi[d]);
  }
}

}  // namespace dali
