// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/signal/window/extract_windows_cpu.h"
#include <algorithm>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/boundary.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/common/for_axis.h"
#include "dali/kernels/common/utils.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename OutputType, typename InputType, int Dims, bool vertical>
constexpr int ExtractWindowsCpu<OutputType, InputType, Dims, vertical>::InputDims;

template <typename OutputType, typename InputType, int Dims, bool vertical>
constexpr int ExtractWindowsCpu<OutputType, InputType, Dims, vertical>::OutputDims;

template <typename OutputType, typename InputType, int Dims, bool vertical>
ExtractWindowsCpu<OutputType, InputType, Dims, vertical>::~ExtractWindowsCpu() = default;

template <typename OutputType, typename InputType, int Dims, bool vertical>
KernelRequirements ExtractWindowsCpu<OutputType, InputType, Dims, vertical>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, InputDims> &in,
    const InTensorCPU<float, 1> &window_fn,
    const ExtractWindowsArgs &args) {
  KernelRequirements req;

  window_length_ = args.window_length > 0 ? args.window_length : 1;
  window_step_ = args.window_step > 0 ? args.window_step : 1;
  padding_ = args.padding;
  if (padding_ != Padding::None)
    window_center_offset_ = args.window_center < 0 ? window_length_ / 2 : args.window_center;
  else
    window_center_offset_ = 0;

  DALI_ENFORCE(window_center_offset_ >= 0 && window_center_offset_ <= window_length_,
    make_string("Window center offset must be in the range [0, ", window_length_, "]"));

  window_fn_length_ = volume(window_fn.shape);
  DALI_ENFORCE(window_fn_length_ > 0, "Window function should not be empty");
  DALI_ENFORCE(window_fn_length_ <= window_length_,
    "Window function size should be equal or less than the specified window length");

  // input data temporal axis (last in input shape by default)
  axis_ = args.axis >= 0 ? args.axis : InputDims - 1;
  DALI_ENFORCE(axis_ >= 0 && axis_ < InputDims,
    make_string("Input temporal axis (", axis_, ") is out of range [0, ", InputDims, ")"));

  const auto n = in.shape[axis_];

  nwindows_ = num_windows(n, window_length_, window_step_, padding_ != Padding::None);
  assert(nwindows_ > 0);

  TensorShape<DynamicDimensions> out_shape;
  out_shape.resize(OutputDims);

  for (int out_idx = 0, in_idx = 0; in_idx < InputDims; in_idx++) {
    if (in_idx == axis_) {
      if (vertical) {
        out_shape[out_idx++] = window_length_;
        out_shape[out_idx++] = nwindows_;
      } else {
        out_shape[out_idx++] = nwindows_;
        out_shape[out_idx++] = window_length_;
      }
    } else {
      out_shape[out_idx++] = in.shape[in_idx];
    }
  }

  std::vector<TensorShape<DynamicDimensions>> tmp = {out_shape};  // workaround for clang-6 bug
  req.output_shapes = {TensorListShape<DynamicDimensions>(tmp)};
  return req;
}

template <typename OutputType, typename InputType, int Dims, bool vertical>
void ExtractWindowsCpu<OutputType, InputType, Dims, vertical>::Run(
    KernelContext &context,
    const OutTensorCPU<OutputType, OutputDims> &out,
    const InTensorCPU<InputType, InputDims> &in,
    const InTensorCPU<float, 1> &window_fn,
    const ExtractWindowsArgs &args) {

  auto in_shape = in.shape;
  auto in_strides = GetStrides(in_shape);

  // flat_out_shape is the output shape with both window index and time dimensions combined into
  // one dimension
  auto flat_out_shape = in_shape;
  flat_out_shape[axis_] = nwindows_ * window_length_;
  auto out_strides = GetStrides(flat_out_shape);

  ForAxis(
    out.data, in.data, flat_out_shape.data(), out_strides.data(),
    in_shape.data(), in_strides.data(), axis_, InputDims,
    [this, &window_fn](
      OutputType *out_data, const InputType *in_data,
      int64_t out_size, int64_t out_stride, int64_t in_size, int64_t in_stride) {
        for (int64_t w = 0; w < nwindows_; w++) {
          int64_t window_start = w * window_step_ - window_center_offset_;
          // Window needs special treatment (falls outside of the signal)
          if (window_start < 0 || window_start + window_length_ > in_size) {
            for (int t = 0; t < window_length_; t++) {
              int64_t out_idx = vertical ? t * nwindows_ + w : w * window_length_ + t;
              int64_t in_idx = window_start + t;
              if (padding_ == Padding::Reflect) {
                // find the mirrored position if the index is out of bounds
                in_idx = boundary::idx_reflect_101(in_idx, in_size);
                // at this point we know that in_idx is in valid range
                out_data[out_idx * out_stride] = window_fn.data[t] * in_data[in_idx * in_stride];
              } else {
                // force out-of-range values to 0 (and avoid multiplication)
                out_data[out_idx * out_stride] = (in_idx >= 0 && in_idx < in_size) ?
                  window_fn.data[t] * in_data[in_idx * in_stride] : 0;
              }
            }
          } else {  // no special treatment for this window (just copy)
            for (int t = 0; t < window_length_; t++) {
              int64_t out_idx = vertical ? t * nwindows_ + w : w * window_length_ + t;
              int64_t in_idx = window_start + t;
              out_data[out_idx * out_stride] = window_fn.data[t] * in_data[in_idx * in_stride];
            }
          }
        }
    });
}

template class ExtractWindowsCpu<float, float, 1, false>;  // 1-channel
template class ExtractWindowsCpu<float, float, 1, true>;   // 1-channel
template class ExtractWindowsCpu<float, float, 2, false>;  // n-channel
template class ExtractWindowsCpu<float, float, 2, true>;   // n-channel

}  // namespace signal
}  // namespace kernels
}  // namespace dali
