// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_SLICE_SLICE_KERNEL_UTILS_H_
#define DALI_KERNELS_SLICE_SLICE_KERNEL_UTILS_H_

#include <vector>
#include <tuple>
#include <utility>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {

template <typename T, int Dims>
struct SliceArgs {
  TensorShape<Dims> anchor;
  TensorShape<Dims> shape;
  SmallVector<T, 8> fill_values = {0, };
  int channel_dim = -1;
};

template <typename T, typename Container>
const T *GetPtr(const Container &c) {
  return c.empty() ? nullptr : c.data();
}

template <int Dims, typename Args>
void CheckValidOutputShape(const TensorShape<Dims>& in_sample_shape,
                           const TensorShape<Dims>& out_sample_shape,
                           const Args& args) {
  for (int d = 0; d < Dims; d++) {
    DALI_ENFORCE(args.shape[d] <= out_sample_shape[d],
      "Output shape dimension " + std::to_string(d) + " is too small");
  }
}

template <int Dims, typename Args>
TensorShape<Dims> GetOutputShape(const TensorShape<Dims>& in_sample_shape,
                                 const Args& args) {
  TensorShape<Dims> out_sample_shape(args.shape);
  return out_sample_shape;
}

template <int Dims, typename Args>
TensorListShape<Dims> GetOutputShapes(const TensorListShape<Dims>& in_shapes,
                                      const std::vector<Args> &args) {
    DALI_ENFORCE(args.size() == static_cast<size_t>(in_shapes.size()),
      "Number of samples and size of slice arguments should match");

    TensorListShape<Dims> output_shapes(in_shapes.size(), Dims);
    for (int i = 0; i < in_shapes.size(); i++) {
      auto out_sample_shape = GetOutputShape(in_shapes[i], args[i]);
      output_shapes.set_tensor_shape(i, out_sample_shape);
    }
    return output_shapes;
}

template <typename Anchor, typename InShape, typename OutShape>
inline bool NeedPad(int ndim,
                    const Anchor &anchor,
                    const InShape &in_shape,
                    const OutShape &out_shape) {
  bool need_pad = false;
  for (int d = 0; d < ndim && !need_pad; d++)
    need_pad = (anchor[d] < 0) || ((anchor[d] + out_shape[d]) > in_shape[d]);
  return need_pad;
}

/**
 * @brief Fills output with nchannel values repeatedly
 */
template <typename T>
void PadFill(T *output, const T *fill_values, int64_t npixels, int64_t nchannels) {
  int64_t n = npixels * nchannels;
  int64_t i = 0;
  for (; i < nchannels; i++)
    output[i] = fill_values[i];
  for (; i < n; i++)
    output[i] = output[i - nchannels];
}

inline std::tuple<int64_t, int64_t, int64_t> CalcPadCopyExtents(int64_t anchor,
                                                                int64_t in_extent,
                                                                int64_t out_extent) {
  int64_t pad_before = std::min(out_extent, std::max<int64_t>(0, -anchor));
  int64_t to_copy = std::max<int64_t>(
      0, std::min(in_extent - std::max<int64_t>(0, anchor), out_extent - pad_before));
  int64_t pad_after = out_extent - pad_before - to_copy;
  return std::tuple<int64_t, int64_t, int64_t>{pad_before, to_copy, pad_after};
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_KERNEL_UTILS_H_
