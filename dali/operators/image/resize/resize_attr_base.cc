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

#include "dali/operators/image/resize/resize_attr_base.h"
#include <string>
#include <vector>
#include "dali/core/math_util.h"
#include "dali/pipeline/operator/common.h"


namespace dali {

DALI_SCHEMA(ResizeAttrBase)
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

    This argument is mutually exclusive with `resize_longer` and `resize_shorter`)code",
                    "default")
    .AddOptionalArg("subpixel_scale", R"code(If True, fractional sizes, directly specified or
calculated, will cause the input ROI to be adjusted to keep the scale factor.

Otherwise, the scale factor will be adjusted so that the source image maps to
the rounded output size.)code",
                    true)
    .AddOptionalArg<vector<float>>("roi_start", R"code(Origin of the input region of interest (ROI).

Must be specified together with `roi_end`. The coordinates follow the tensor shape order, which is
the same as `size`. The coordinates can be either absolute (in pixels, which is the default) or
relative (0..1), depending on the value of ``relative_roi`` argument. If the ROI origin is greater
than the ROI end in any dimension, the region is flipped in that dimension.)code",
                                   nullptr, true)
    .AddOptionalArg<vector<float>>("roi_end", R"code(End of the input region of interest (ROI).

Must be specified together with `roi_start`. The coordinates follow the tensor shape order, which is
the same as `size`. The coordinates can be either absolute (in pixels, which is the default) or
relative (0..1), depending on the value of ``relative_roi`` argument. If the ROI origin is greater
than the ROI end in any dimension, the region is flipped in that dimension.)code",
                                   nullptr, true)
    .AddOptionalArg("roi_relative", R"code(If true, ROI coordinates are relative to the input size,
where 0 denotes top/left and 1 denotes bottom/right)code",
                    false)
    .AddOptionalArg<vector<float>>("max_size", R"code(Limit of the output size.

When the operator is configured to keep aspect ratio and only the smaller dimension is specified,
the other(s) can grow very large. This can happen when using `resize_shorter` argument
or "not_smaller" mode or when some extents are left unspecified.

This parameter puts a limit to how big the output can become. This value can be specified per-axis
or uniformly for all axes.

.. note::
  When used with "not_smaller" mode or `resize_shorter` argument, `max_size` takes
  precedence and the aspect ratio is kept - for example, resizing with
  ``mode="not_smaller", size=800, max_size=1400`` an image of size 1200x600 would be resized to
  1400x700.
)code",
                                   {}, false);


void AdjustOutputSize(float *out_size, const float *in_size, int ndim, ResizeMode mode,
                      const float *max_size) {
  SmallVector<double, 3> scale;
  SmallVector<bool, 3> mask;
  scale.resize(ndim, 1);
  mask.resize(ndim);

  int sizes_provided = 0;
  for (int d = 0; d < ndim; d++) {
    mask[d] = (out_size[d] != 0 && in_size[d] != 0);
    scale[d] = in_size[d] ? out_size[d] / in_size[d] : 1;
    sizes_provided += mask[d];
  }

  if (sizes_provided == 0) {  // no clue how to resize - keep original size then
    for (int d = 0; d < ndim; d++)
      out_size[d] = in_size[d];
  } else {
    if (mode == ResizeMode::Default) {
      if (sizes_provided < ndim) {
        // if only some extents are provided - calculate average scale
        // and use it for the missing dimensions;
        double avg_scale = 1;
        for (int d = 0; d < ndim; d++) {
          if (mask[d])
            avg_scale *= std::abs(scale[d]);  // abs because of possible flipping
        }
        if (sizes_provided > 1)
          avg_scale = std::pow(avg_scale, 1.0 / sizes_provided);
        for (int d = 0; d < ndim; d++) {
          if (!mask[d]) {
            scale[d] = avg_scale;
            out_size[d] = in_size[d] * scale[d];
          }
        }
      }
      if (max_size) {
        for (int d = 0; d < ndim; d++) {
          if (max_size[d] > 0 && std::abs(out_size[d]) > max_size[d]) {
            out_size[d] = std::copysignf(max_size[d], out_size[d]);
            scale[d] = out_size[d] / in_size[d];
          }
        }
      }
    } else if (mode == ResizeMode::Stretch) {
      if (sizes_provided < ndim) {
        for (int d = 0; d < ndim; d++) {
          if (!mask[d]) {
            scale[d] = 1;
            out_size[d] = in_size[d];
          }
        }
      }
      if (max_size) {
        for (int d = 0; d < ndim; d++) {
          if (max_size[d] > 0 && std::abs(out_size[d]) > max_size[d]) {
            out_size[d] = std::copysignf(max_size[d], out_size[d]);
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
      for (int d = 0; d < ndim; d++) {
        if (mask[d]) {
          float s = std::abs(scale[d]);
          if (first ||
              (mode == ResizeMode::NotSmaller && s > final_scale) ||
              (mode == ResizeMode::NotLarger && s < final_scale))
            final_scale = s;
          first = false;
        }
      }

      // If there's a size limit, apply it and possibly reduce the final scale
      if (max_size) {
        for (int d = 0; d < ndim; d++) {
          if (max_size[d] > 0) {
            double s = static_cast<double>(max_size[d]) / in_size[d];
            if (s < final_scale)
              final_scale = s;
          }
        }
      }

      // Now, let's apply the final scale to all dimensions - if final_scale is different (in
      // absolute value) than one defined by the caller, adjust it; also, if no scale was defined
      // for a dimension, then just use the final scale.
      for (int d = 0; d < ndim; d++) {
        if (!mask[d] || std::abs(scale[d]) != final_scale) {
          scale[d] = std::copysign(final_scale, scale[d]);
          out_size[d] = in_size[d] * scale[d];
        }
      }
    }
  }
}

}  // namespace dali
