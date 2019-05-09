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
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(Flip)
    .DocStr(R"code(Flip the image on the horizontal and/or vertical axes.)code")
    .NumInput(1)
    .NumOutput(1)
    .AllowMultipleInputSets()
    .AddOptionalArg("horizontal", R"code(Perform a horizontal flip.)code", 1, true)
    .AddOptionalArg("vertical", R"code(Perform a vertical flip.)code", 0, true);


template <>
Flip<CPUBackend>::Flip(const OpSpec &spec)
    : Operator<CPUBackend>(spec), spec_(spec) {}

int GetOcvType(const TypeInfo &type, size_t channels) {
  if (channels > CV_CN_MAX) {
    DALI_FAIL("Number of channels must be smaller than " + std::to_string(CV_CN_MAX+1) +
    " and the sample has " + std::to_string(channels) + " channels.");
  }
  switch (type.size()) {
    case 1:
      return CV_8UC(channels);
    case 2:
      return CV_16UC(channels);
    case 4:
      return CV_32FC(channels);
    case 8:
      return CV_64FC(channels);
    default:
      DALI_FAIL(type.name() + " is not a valid type for the flip operator.");
  }
}

template <typename T>
void FlipKernel(T *output, const T *input, size_t height, size_t width,
    size_t channels_per_layer, size_t layers, bool horizontal, bool vertical) {
  int flip_flag = -1;
  if (!vertical) flip_flag = 1;
  else if (!horizontal) flip_flag = 0;
  auto ocv_type = GetOcvType(TypeInfo::Create<T>(), channels_per_layer);
  size_t layer_size = height * width * channels_per_layer;
  for (size_t layer = 0; layer < layers; ++layer) {
    auto input_mat = CreateMatFromPtr(height, width, ocv_type, input + layer * layer_size);
    auto output_mat = CreateMatFromPtr(height, width, ocv_type, output + layer * layer_size);
    cv::flip(input_mat, output_mat, flip_flag);
  }
}

void RunFlip(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input,
    bool horizontal, bool vertical) {
  DALI_TYPE_SWITCH(
      input.type().id(), DType,
      auto output_ptr = output.mutable_data<DType>();
      auto input_ptr = input.data<DType>();
      if (input.GetLayout() == DALI_NHWC) {
        ssize_t height = input.dim(0), width = input.dim(1), channels = input.dim(2);
        FlipKernel(output_ptr, input_ptr, height, width, channels, 1, horizontal, vertical);
      } else if (input.GetLayout() == DALI_NCHW) {
        ssize_t height = input.dim(1), width = input.dim(2), channels = input.dim(0);
        FlipKernel(output_ptr, input_ptr, height, width, 1, channels, horizontal, vertical);
      }
  )
}

template <>
void Flip<CPUBackend>::RunImpl(Workspace<CPUBackend> *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto &output = ws->Output<CPUBackend>(idx);
  output.SetLayout(input.GetLayout());
  output.set_type(input.type());
  output.ResizeLike(input);
  DALI_ENFORCE(input.ndim() == 3);
  auto _horizontal = GetHorizontal(ws, ws->data_idx());
  auto _vertical = GetVertical(ws, ws->data_idx());
  if (!_horizontal && !_vertical) {
    output.Copy(input, nullptr);
  } else {
    RunFlip(output, input, _horizontal, _vertical);
  }
}

DALI_REGISTER_OPERATOR(Flip, Flip<CPUBackend>, CPU);

}  // namespace dali
