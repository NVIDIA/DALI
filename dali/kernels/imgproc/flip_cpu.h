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


#ifndef DALI_KERNELS_IMGPROC_FLIP_CPU_H_
#define DALI_KERNELS_IMGPROC_FLIP_CPU_H_

#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace kernels {

int GetOcvType(const TypeInfo &type, size_t channels) {
  if (channels * type.size() > CV_CN_MAX) {
    DALI_FAIL("Pixel size must not be greater than " + std::to_string(CV_CN_MAX) + " bytes.");
  }
  return CV_8UC(type.size() * channels);
}

template <typename T>
void FlipKernel(T *output, const T *input, size_t layers, size_t height, size_t width,
                size_t channels, bool horizontal, bool vertical) {
  int flip_flag = -1;
  if (!vertical) flip_flag = 1;
  else if (!horizontal) flip_flag = 0;
  auto ocv_type = GetOcvType(TypeInfo::Create<T>(), channels);
  size_t layer_size = height * width * channels;
  for (size_t layer = 0; layer < layers; ++layer) {
    auto input_mat = CreateMatFromPtr(height, width, ocv_type, input + layer * layer_size);
    auto output_mat = CreateMatFromPtr(height, width, ocv_type, output + layer * layer_size);
    cv::flip(input_mat, output_mat, flip_flag);
  }
}

template <typename Type>
class DLL_PUBLIC FlipCPU {
 public:
  DLL_PUBLIC FlipCPU() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context, const InTensorCPU<Type, 4> &in) {
    KernelRequirements req;
    req.output_shapes.emplace_back(TensorListShape<DynamicDimensions>({in.shape}));
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &Context, OutTensorCPU<Type, 4> &out,
      const InTensorCPU<Type, 4> &in, bool horizontal, bool vertical) {
    auto in_data = in.data;
    auto out_data = out.data;
    if (horizontal || vertical) {
      FlipKernel(out_data, in_data, in.shape[0], in.shape[1], in.shape[2], in.shape[3], horizontal,
                 vertical);
    } else {
      std::copy(in_data, in_data + in.num_elements(), out_data);
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_FLIP_CPU_H_
