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
namespace detail {
namespace cpu {

inline int GetOcvType(const TypeInfo &type, size_t channels) {
  if (channels * type.size() > CV_CN_MAX) {
    DALI_FAIL("Pixel size must not be greater than " + std::to_string(CV_CN_MAX) + " bytes.");
  }
  return CV_8UC(type.size() * channels);
}

template <typename Type>
void OcvFlip(Type *output, const Type *input,
                    size_t layers, size_t height, size_t width, size_t channels,
                    bool flip_z, bool flip_y, bool flip_x) {
  assert(flip_x || flip_y);
  int flip_flag = -1;
  if (!flip_y)
    flip_flag = 1;
  else if (!flip_x)
    flip_flag = 0;
  auto ocv_type = GetOcvType(TypeInfo::Create<Type>(), channels);
  size_t layer_size = height * width * channels;
  for (size_t layer = 0; layer < layers; ++layer) {
    auto output_layer = flip_z ? layers - layer - 1 : layer;
    auto input_mat = CreateMatFromPtr(height, width, ocv_type, input + layer * layer_size);
    auto output_mat = CreateMatFromPtr(height, width, ocv_type,
                                       output + output_layer * layer_size);
    cv::flip(input_mat, output_mat, flip_flag);
  }
}

template <typename Type>
void FlipZAxis(Type *output, const Type *input, size_t layers, size_t height, size_t width,
                      size_t channels, bool flip_z) {
  if (!flip_z) {
    std::copy(input, input + layers * height * width * channels, output);
    return;
  }
  size_t layer_size = height * width * channels;
  for (size_t layer = 0; layer < layers; ++layer) {
    auto out_layer = layers - layer - 1;
    std::copy(input + layer * layer_size, input + (layer + 1) * layer_size,
              output + out_layer * layer_size);
  }
}

template <typename Type>
void FlipImpl(Type *output, const Type *input,
                       size_t layers, size_t height, size_t width,
                       size_t channels, bool flip_z, bool flip_y, bool flip_x) {
  if (flip_x || flip_y) {
    OcvFlip(output, input, layers, height, width, channels, flip_z, flip_y, flip_x);
  } else {
    FlipZAxis(output, input, layers, height, width, channels, flip_z);
  }
}

}  // namespace cpu
}  // namespace detail

template <typename Type>
class DLL_PUBLIC FlipCPU {
 public:
  DLL_PUBLIC FlipCPU() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context, const InTensorCPU<Type, 4> &in) {
    KernelRequirements req;
    req.output_shapes = {TensorListShape<DynamicDimensions>({in.shape})};
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &Context, OutTensorCPU<Type, 4> &out,
      const InTensorCPU<Type, 4> &in, bool flip_z, bool flip_y, bool flip_x) {
    auto in_data = in.data;
    auto out_data = out.data;
    detail::cpu::FlipImpl(out_data, in_data, in.shape[0], in.shape[1], in.shape[2], in.shape[3],
        flip_z, flip_y, flip_x);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_FLIP_CPU_H_
