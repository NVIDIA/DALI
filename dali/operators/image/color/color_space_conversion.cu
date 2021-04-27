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

#include <cuda_runtime_api.h>
#include <utility>
#include <vector>
#include "dali/operators/image/color/color_space_conversion.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_kernel.cuh"
#include "dali/core/dev_buffer.h"
#include "dali/core/geom/vec.h"

namespace dali {


template<>
void ColorSpaceConversion<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  DALI_ENFORCE(IsType<uint8_t>(input.type()),
      "Color space conversion accept only uint8 tensors");
  auto &output = ws.Output<GPUBackend>(0);
  auto layout = input.GetLayout();
  output.SetLayout(layout);
  auto stream = ws.stream();

  int input_C = NumberOfChannels(input_type_);
  int output_C = NumberOfChannels(output_type_);
  const auto& input_shape = input.shape();
  auto output_shape = input_shape;
  if (input_C != output_C) {
    for (unsigned int i = 0; i < input.ntensor(); ++i) {
      DALI_ENFORCE(input_shape.tensor_shape_span(i)[2] == input_C,
        "Wrong number of channels for input");
      output_shape.tensor_shape_span(i)[2] = output_C;
    }
  }
  output.Resize(output_shape);
  output.set_type(input.type());

  DALI_ENFORCE(layout == "HWC" || (layout.empty() && output_shape.sample_dim() == 3),
               make_string("Unexpected layout: ", layout, " shape: ", output_shape,
                           ". Expected data in HWC layout."));
  for (unsigned int i = 0; i < input.ntensor(); ++i) {
    auto sample_sh = input_shape.tensor_shape_span(i);
    // input/output
    const uint8_t* input_data = input.tensor<uint8_t>(i);
    uint8_t* output_data = output.mutable_tensor<uint8_t>(i);
    int64_t npixels = sample_sh[0] * sample_sh[1];
    kernels::color::RunColorSpaceConversionKernel(output_data, input_data, output_type_,
                                                  input_type_, npixels, stream);
  }
}

DALI_REGISTER_OPERATOR(ColorSpaceConversion, ColorSpaceConversion<GPUBackend>, GPU);

}  // namespace dali
