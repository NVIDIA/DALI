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

#include "dali/kernels/signal/window/extract_frames_cpu.h"
#include <algorithm>
#include "dali/kernels/signal/signal_kernel_utils.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace signal {
namespace window {

template <typename OutputType, typename InputType, int Dims>
constexpr int ExtractFramesCpu<OutputType, InputType, Dims>::InputDims;

template <typename OutputType, typename InputType, int Dims>
constexpr int ExtractFramesCpu<OutputType, InputType, Dims>::OutputDims;

template <typename OutputType, typename InputType, int Dims>
ExtractFramesCpu<OutputType, InputType, Dims>::~ExtractFramesCpu() = default;

template <typename OutputType, typename InputType, int Dims>
KernelRequirements ExtractFramesCpu<OutputType, InputType, Dims>::Setup(
    KernelContext &context,
    const InTensorCPU<InputType, InputDims> &in,
    const ExtractFramesArgs &args) {
  KernelRequirements req;

  window_length_ = args.window_length > 0 ? args.window_length : 1;
  window_step_ = args.window_step > 0 ? args.window_step : 1;

  // input data temporal axis (last in input shape by default)
  in_time_axis_ = args.in_time_axis >= 0 ? args.in_time_axis : InputDims - 1;
  DALI_ENFORCE(in_time_axis_ >= 0 && in_time_axis_ < InputDims,
    make_string("Input temporal axis (", in_time_axis_, ") is out of range [0, ", InputDims, ")"));
  DALI_ENFORCE(in_time_axis_ == InputDims - 1,
    "Current implementation expects time dimension to be the inner-most dimension");

  // frame temporal axis (last in output shape by default)
  out_frame_axis_ = args.out_frame_axis >= 0 ? args.out_frame_axis : OutputDims - 1;
  DALI_ENFORCE(out_frame_axis_ >= 0 && out_frame_axis_ < OutputDims,
    make_string("Output frame temporal axis (", out_frame_axis_, ") is out of range [0, ",
      OutputDims, ")"));
  DALI_ENFORCE(out_frame_axis_ == OutputDims - 1,
    "Current implementation expects window time dimension to be the inner-most dimension");


  const auto n = in.shape[in_time_axis_];

  nwindows_ = (n + window_step_ - 1) / window_step_;
  assert(nwindows_ > 0);

  TensorShape<DynamicDimensions> out_shape;
  out_shape.resize(OutputDims);

  for (int d = 0, out_idx = 0, in_idx = 0; out_idx < OutputDims; d++) {
    if (d == out_frame_axis_ || d == in_time_axis_) {
      if (d == out_frame_axis_) {
        assert(out_idx < OutputDims);
        out_shape[out_idx++] = window_length_;
      }
      if (d == in_time_axis_) {
        assert(out_idx < OutputDims);
        assert(in_idx < InputDims);
        out_shape[out_idx++] = nwindows_;
        in_idx++;
      }
    } else {
      assert(out_idx < OutputDims);
      assert(in_idx < InputDims);
      out_shape[out_idx++] = in.shape[in_idx++];
    }
  }
  std::cout << out_shape[0] << "x" << out_shape[1] << "x" << out_shape[2] << "\n";
  req.output_shapes = {TensorListShape<DynamicDimensions>({out_shape})};
  return req;
}

template <typename OutputType, typename InputType, int Dims>
void ExtractFramesCpu<OutputType, InputType, Dims>::Run(
    KernelContext &context,
    const OutTensorCPU<OutputType, OutputDims> &out,
    const InTensorCPU<InputType, InputDims> &in,
    const ExtractFramesArgs &args) {

  auto in_shape = in.shape;
  auto in_strides = GetStrides(in_shape);

  // flat_out_shape is the output shape with both window index and time dimensions combined into
  // one dimension
  auto flat_out_shape = in_shape;
  flat_out_shape[in_time_axis_] = nwindows_ * window_length_;
  auto out_strides = GetStrides(flat_out_shape);

  std::vector<std::pair<OutputType*, const InputType*>> slices;
  slices.push_back(std::make_pair(out.data, in.data));
  Get1DSlices(slices,
              flat_out_shape.data(), out_strides.data(),
              in_shape.data(), in_strides.data(),
              in_time_axis_, InputDims);

  const auto out_stride = out_strides[in_time_axis_];
  const auto in_stride = in_strides[in_time_axis_];
  const auto in_t_size = in.shape[in_time_axis_];

  std::cout << "nwindows " << nwindows_
            << " window_length " << window_length_
            << " window step " << window_step_ << "\n"
            << " out shape (" << flat_out_shape.size()<<") " << flat_out_shape[0] << "x" <<flat_out_shape[1] << "\n"
            << " in shape (" << in_shape.size()<<") " << in_shape[0] << "x" << in_shape[1] << "\n";

  for (auto &slice : slices) {
    OutputType *out_data = slice.first;
    const InputType *in_data = slice.second;

    std::cout << "start out slice at " << (int) (slice.first - out.data)
              << " with stride " << out_stride << "\n";
    std::cout << "start in slice at " << (int) (slice.second - in.data)
              << " with stride " << in_stride << "\n";

    for (int window_idx = 0; window_idx < nwindows_; window_idx++) {
      for (int t = 0; t < window_length_; t++) {
        int out_idx = window_idx * window_length_ + t;
        int in_idx = window_idx * window_step_ + t;
        if (in_idx < in_t_size) {
          out_data[out_idx * out_stride] = in_data[in_idx * in_stride];
        } else {
          out_data[out_idx * out_stride] = 0;
        }
      }
    }
  }
}

template class ExtractFramesCpu<float, float, 1>;  // 1-channel
template class ExtractFramesCpu<uint8_t, uint8_t, 1>;  // 1-channel

template class ExtractFramesCpu<float, float, 2>;  // n-channel
template class ExtractFramesCpu<uint8_t, uint8_t, 2>;  // n-channel


}  // namespace window
}  // namespace signal
}  // namespace kernels
}  // namespace dali
