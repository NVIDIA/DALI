// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_BASIC_CROP_H_
#define DALI_PIPELINE_BASIC_CROP_H_

#include <tuple>

#include "dali/pipeline/basic/coords.h"
#include "dali/pipeline/basic/runner.h"
#include "dali/pipeline/basic/sequence.h"
#include "dali/pipeline/basic/tensor.h"

namespace dali {
namespace basic {

template <size_t N>
std::array<Index, N> ToStaticShape(const std::vector<Index> &shape) {
  std::array<Index, N> result;
  for (size_t i = 0; i < N; i++) {
    result[i] = shape[i];
  }
  return result;
}

template <size_t N>
std::vector<Index> ToDynamicShape(const std::array<Index, N> &shape) {
  return {shape.begin(), shape.end()};
}


// TODO(klecki) allow for Schema to be present without using concrete types
struct CropSchema {
  using AllowedInputs = std::tuple<uint8_t>;
  using AllowedOutputs = std::tuple<uint8_t, int8_t, int16_t, int32_t, int64_t, float>;
};

template <typename T, typename U, typename OutShape>
class Crop {
 public:
  Crop() = delete;
  static constexpr size_t input_dim = 3;
  static constexpr size_t output_dim = 3;
  // TODO(klecki) - should be converted to some kind of std::tuple for multiple input and outputs
  using InputType = TensorWrapper<const T, input_dim>;
  using OutputType = TensorWrapper<U, output_dim>;

  static constexpr bool can_calculate_output_size = true;

  // TODO(klecki): One of the issues is if this type can depend on T and U or if we can have
  //               the same KernelAttributes for all concrete Crop<T, U> types.
  struct KernelAttributes {
    const Index h_start;
    const Index w_start;
    const Index H_out;
    const Index W_out;
  };

  static std::array<Index, output_dim> CalcOutputSize(const std::array<Index, input_dim> &in, KernelAttributes attr) {  // NOLINT
    return permuteShape({attr.H_out, attr.W_out, in[2]}, OutShape{});
  }

  // signature due to change, for now it is iterator-like, should be collection of iterators
  static void Run(InputType in, KernelAttributes attr, OutputType out) {  // NOLINT
    const auto *in_ptr = in.ptr + getOffset<0, 1, 2>(in.GetShape(), {attr.h_start, attr.w_start, 0});
    for (Index h = 0; h < attr.H_out; h++) {
      for (Index w = 0; w < attr.W_out; w++) {
        for (Index c = 0; c < in.GetShape()[2]; c++) {
          // From HWC
          const auto in_idx = getOffset<0, 1, 2>(in.GetShape(), {h, w, c});
          // To HWC or CHW
          const auto out_idx = getOffset(out.GetShape(), {h, w, c}, OutShape{});
          out.ptr[out_idx] = static_cast<U>(in_ptr[in_idx]);
        }
      }
    }
  }
};

template <typename T, typename U, typename OutShape>
using SequenceCrop = SequenceAdapter<Crop<T, U, OutShape>>;

template <typename Out, typename OutShape>
using CropRunHelper = RunHelperAllowMIS<Crop<uint8_t, Out, OutShape>>;

template <typename Out, typename OutShape>
using CropSizeHelper = SizeHelperAllowMIS<Crop<uint8_t, Out, OutShape>>;

template <typename Out, typename OutShape>
using SequenceCropRunHelper = RunHelperAllowMIS<SequenceCrop<uint8_t, Out, OutShape>>;

template <typename Out, typename OutShape>
using SequenceCropSizeHelper = SizeHelperAllowMIS<SequenceCrop<uint8_t, Out, OutShape>>;

}  // namespace basic
}  // namespace dali

#endif  // DALI_PIPELINE_BASIC_CROP_H_
