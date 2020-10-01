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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_BASELINE_CONVOLUTION_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_BASELINE_CONVOLUTION_H_

#include "dali/core/boundary.h"
#include "dali/core/convert.h"
#include "dali/core/format.h"
#include "dali/core/tensor_view.h"

namespace dali {
namespace kernels {
namespace testing {

template <typename T>
void InitTriangleWindow(const TensorView<StorageCPU, T, 1> &window) {
  int radius = window.num_elements() / 2;
  for (int i = 0; i < radius; i++) {
    *window(i) = i + 1;
    *window(window.num_elements() - i - 1) = i + 1;
  }
  *window(radius) = radius + 1;
}

template <typename Out, typename In, typename W>
void BaselineConvolveAxis(Out *out, const In *in, const W *window, int len, int r, int channel_num,
                          int64_t stride) {
  for (int i = 0; i < len; i++) {
    for (int c = 0; c < channel_num; c++) {
      W accum = {};
      for (int d = -r; d <= r; d++) {
        accum += in[boundary::idx_reflect_101(i + d, len) * stride + c] * window[d + r];
      }
      out[i * stride + c] = ConvertSat<Out>(accum);
    }
  }
}

/**
 * @brief Convolve input with window of radius r, along specified axis. Used for testing
 *
 * Uses border reflect 101.
 */
template <typename Out, typename In, typename W, int ndim>
void BaselineConvolve(const TensorView<StorageCPU, Out, ndim> &out,
                      const TensorView<StorageCPU, In, ndim> &in,
                      const TensorView<StorageCPU, W, 1> &window, int axis, int r,
                      int current_axis = 0, int64_t offset = 0) {
  if (current_axis == ndim - 1) {
    auto stride = GetStrides(out.shape)[axis];
    BaselineConvolveAxis(out.data + offset, in.data + offset, window.data, out.shape[axis], r,
                         in.shape[ndim - 1], stride);
  } else if (current_axis == axis) {
    BaselineConvolve(out, in, window, axis, r, current_axis + 1, offset);
  } else {
    for (int i = 0; i < out.shape[current_axis]; i++) {
      auto stride = GetStrides(out.shape)[current_axis];
      BaselineConvolve(out, in, window, axis, r, current_axis + 1, offset + i * stride);
    }
  }
}

}  // namespace testing
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_BASELINE_CONVOLUTION_H_
