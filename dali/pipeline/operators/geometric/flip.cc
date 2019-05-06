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

#include "dali/pipeline/operators/geometric/flip.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Flip)
    .DocStr(R"code(Converts between various image color models)code")
        .NumInput(1)
        .NumOutput(1)
        .AllowMultipleInputSets()
        .AddOptionalArg("horizontal",
                R"code(The color space of the input image)code",
                1)
        .AddOptionalArg("vertical",
                R"code(The color space of the output image)code",
                0);

template <>
Flip<CPUBackend>::Flip(const OpSpec& spec)
    : Operator<CPUBackend>(spec)
    , _horizontal(spec.GetArgument<int32>("horizontal"))
    , _vertical(spec.GetArgument<int32>("vertical")) {}


enum class FlipAxis {
  HORIZONTAL,
  VERTICAL,
  BOTH
};

template <FlipAxis D>
struct FlipImpl {};

template <>
struct FlipImpl<FlipAxis::HORIZONTAL> {
  template <typename T>
  static void run(T *output, const T *input, size_t height, size_t width, size_t channels) {
    for (size_t r = 0; r < height; ++r) {
      for (size_t c = 0; c < width; ++c) {
        std::copy(
            &input[(r * width + c) * channels],
            &input[(r * width + c + 1) * channels],
            &output[(r * width + width - c - 1) * channels]);
      }
    }
  }
};

template <>
struct FlipImpl<FlipAxis::VERTICAL> {
  template <typename T>
  static void run(T *output, const T *input, size_t height, size_t width, size_t channels) {
    for (size_t r = 0; r < height; ++r) {
      std::copy(
          &input[r * width * channels],
          &input[(r + 1) * width * channels],
          &output[(height - r - 1) * width * channels]);
    }
  }
};

template <>
struct FlipImpl<FlipAxis::BOTH> {
  template <typename T>
  static void run(T *output, const T *input, size_t height, size_t width, size_t channels) {
    for (size_t r = 0; r < height; ++r) {
      for (size_t c = 0; c < width; ++c) {
        std::copy(
            &input[(r * width + c) * channels],
            &input[(r * width + c + 1) * channels],
            &output[((height - r - 1) * width + width - c - 1) * channels]);
      }
    }
  }
};

template <FlipAxis D>
void RunFlip(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input) {
  DALI_TYPE_SWITCH(input.type().id(), DType,
    auto output_ptr = output.mutable_data<DType>();
    auto input_ptr = input.data<DType>();
    if (input.GetLayout() == DALI_NHWC) {
      ssize_t height = input.dim(0), width = input.dim(1), channels = input.dim(2);
      FlipImpl<D>::run(output_ptr, input_ptr, height, width, channels);
    } else if (input.GetLayout() == DALI_NCHW) {
      ssize_t height = input.dim(1), width = input.dim(2), channels = input.dim(0);
      for (ssize_t c = 0; c < channels; ++c) {
        auto slice_origin = c * height * width;
        FlipImpl<D>::run(output_ptr + slice_origin, input_ptr + slice_origin, height, width, 1);
      }
    }
  )
}

template <>
void Flip<CPUBackend>::RunImpl(Workspace<CPUBackend>* ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);
  output.SetLayout(input.GetLayout());
  output.set_type(input.type());
  output.ResizeLike(input);
  DALI_ENFORCE(input.ndim() == 3);
  if (_vertical && _horizontal) {
    RunFlip<FlipAxis::BOTH>(output, input);
  } else if (_horizontal) {
    RunFlip<FlipAxis::HORIZONTAL>(output, input);
  } else if (_vertical) {
    RunFlip<FlipAxis::VERTICAL>(output, input);
  } else {
    output.Copy(input, nullptr);
  }
}

DALI_REGISTER_OPERATOR(Flip, Flip<CPUBackend>, CPU);

}  // namespace dali
