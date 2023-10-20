// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_BASE_H_
#define DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_BASE_H_

#include <functional>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/math_util.h"
#include "dali/core/small_vector.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape.h"
#include "dali/operators/image/resize/resize_mode.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

struct ResizeParams {
  void resize(int ndim) {
    dst_size.resize(ndim);
    src_lo.resize(ndim);
    src_hi.resize(ndim);
  }
  int size() const { return dst_size.size(); }
  SmallVector<int, 6> dst_size;
  SmallVector<float, 6> src_lo, src_hi;
};

void CalculateInputRoI(SmallVector<float, 3> &in_lo, SmallVector<float, 3> &in_hi, bool has_roi,
                       bool roi_relative, span<const float> roi_start, span<const float> roi_end,
                       const TensorListShape<> &input_shape, int sample_idx, int spatial_ndim,
                       int first_spatial_dim);

void AdjustOutputSize(float *out_size, const float *in_size, int ndim,
                      ResizeMode mode = ResizeMode::Stretch, const float *max_size = nullptr);

// pass sizes by value - the function will modify them internally
template <typename RoundFn = int (*)(float)>
void CalculateSampleParams(ResizeParams &params, SmallVector<float, 3> requested_size,
                           SmallVector<float, 3> in_lo, SmallVector<float, 3> in_hi,
                           bool adjust_roi, bool empty_input, int ndim,
                           ResizeMode mode = ResizeMode::Stretch, const float *max_size = nullptr,
                           span<const float> alignment = {}, RoundFn size_round_fn = round_int) {
  assert(static_cast<int>(requested_size.size()) == ndim);
  assert(static_cast<int>(in_lo.size()) == ndim);
  assert(static_cast<int>(in_hi.size()) == ndim);

  SmallVector<float, 3> in_size;
  in_size.resize(ndim);
  for (int d = 0; d < ndim; d++) {
    float sz = in_hi[d] - in_lo[d];
    if (sz < 0) {
      std::swap(in_hi[d], in_lo[d]);
      requested_size[d] = -requested_size[d];
      sz = -sz;
    }
    in_size[d] = sz;
  }

  AdjustOutputSize(requested_size.data(), in_size.data(), ndim, mode, max_size);

  for (int d = 0; d < ndim; d++) {
    DALI_ENFORCE(in_lo[d] != in_hi[d] || requested_size[d] == 0,
                 "Cannot produce non-empty output from empty input");
  }

  params.resize(ndim);
  params.src_lo = in_lo;
  params.src_hi = in_hi;

  // If the input sample is empty, we simply can't produce _any_ non-empty output.
  // If ROI is degenerate but there's some input, we can sample it at the degenerate location.
  // To prevent empty outputs when we have some means of producing non-empty output, we bump
  // up the size of the output to at least 1 in each axis.
  int min_size = empty_input ? 0 : 1;

  for (int d = 0; d < ndim; d++) {
    float out_sz = requested_size[d];
    bool flip = out_sz < 0;
    params.dst_size[d] = std::max(min_size, size_round_fn(std::fabs(out_sz)));
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

      // Alignment 0 aligns the ROI start, alignment 0.5 aligns the center of the ROI, alignment 1.0
      // aligns the end of the ROI
      double alignment_val = alignment.empty() ? 0.5f : alignment[d];
      double center = (1.0 - alignment_val) * params.src_lo[d] + alignment_val * params.src_hi[d];

      // clamp to more-or-less sane interval to avoid arithmetic problems downstream
      params.src_lo[d] = clamp(center + (params.src_lo[d] - center) * adjustment, -1e+9, 1e+9);
      params.src_hi[d] = clamp(center + (params.src_hi[d] - center) * adjustment, -1e+9, 1e+9);
    }
  }
}

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_BASE_H_
