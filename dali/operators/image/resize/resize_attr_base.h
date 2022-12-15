// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                       bool roi_relative, const float *roi_start, const float *roi_end,
                       const TensorListShape<> &input_shape, int sample_idx, int spatial_ndim,
                       int first_spatial_dim);

void AdjustOutputSize(float *out_size, const float *in_size, int ndim,
                      ResizeMode mode = ResizeMode::Stretch, const float *max_size = nullptr);

// pass sizes by value - the function will modify them internally
void CalculateSampleParams(
    ResizeParams &params, SmallVector<float, 3> requested_size, SmallVector<float, 3> in_lo,
    SmallVector<float, 3> in_hi, bool adjust_roi, bool empty_input, int ndim,
    ResizeMode mode = ResizeMode::Stretch, const float *max_size = nullptr,
    std::function<int(float)> size_round_fn = [](float x) { return round_int(x); });

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_RESIZE_RESIZE_ATTR_BASE_H_
