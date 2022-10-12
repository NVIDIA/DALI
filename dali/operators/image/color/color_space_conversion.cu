// Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
void ColorSpaceConversion<GPUBackend>::RunImpl(Workspace &ws) {
  const auto& input = ws.Input<GPUBackend>(0);
  auto& output = ws.Output<GPUBackend>(0);
  output.SetLayout(input.GetLayout());
  auto in_view = view<const uint8_t>(input);
  auto out_view = view<uint8_t>(output);
  const auto &in_sh = in_view.shape;
  int nsamples = in_sh.num_samples();
  auto stream = ws.stream();
  for (int i = 0; i < nsamples; ++i) {
    auto sample_sh = in_sh.tensor_shape_span(i);
    int64_t npixels = volume(sample_sh.begin(), sample_sh.end() - 1);
    kernels::color::RunColorSpaceConversionKernel(out_view[i].data, in_view[i].data, output_type_,
                                                  input_type_, npixels, stream);
  }
}

DALI_REGISTER_OPERATOR(ColorSpaceConversion, ColorSpaceConversion<GPUBackend>, GPU);

}  // namespace dali
