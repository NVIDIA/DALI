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

namespace dali {
namespace kernels {

constexpr int sample_ndim = 5;

namespace detail {
namespace cpu {

template <typename Type>
inline int GetOcvType(size_t channels) {
  if (channels * sizeof(Type) > CV_CN_MAX) {
    DALI_FAIL("Pixel size must not be greater than " + std::to_string(CV_CN_MAX) + " bytes.");
  }
  return CV_MAKE_TYPE(cv::DataType<Type>::type, channels);
}

template <typename Type>
void OcvFlip(Type *output, const Type *input,
                    size_t depth, size_t height, size_t width, size_t channels,
                    bool flip_z, bool flip_y, bool flip_x) {
  assert(flip_x || flip_y);
  int flip_flag = -1;
  if (!flip_y)
    flip_flag = 1;
  else if (!flip_x)
    flip_flag = 0;
  // coerce data to uint8_t - flip won't look at the type anyway
  auto ocv_type = GetOcvType<uint8_t>(channels * sizeof(Type));
  size_t plane_size = height * width * channels;
  for (size_t plane = 0; plane < depth; ++plane) {
    auto output_plane = flip_z ? depth - plane - 1 : plane;
    auto input_mat = CreateMatFromPtr(height, width, ocv_type, input + plane * plane_size);
    auto output_mat = CreateMatFromPtr(height, width, ocv_type,
                                       output + output_plane * plane_size);
    cv::flip(input_mat, output_mat, flip_flag);
  }
}

template <typename Type>
void FlipZAxis(Type *output, const Type *input, size_t depth, size_t height, size_t width,
                      size_t channels, bool flip_z) {
  if (!flip_z) {
    std::copy(input, input + depth * height * width * channels, output);
    return;
  }
  size_t plane_size = height * width * channels;
  for (size_t plane = 0; plane < depth; ++plane) {
    auto out_plane = depth - plane - 1;
    std::copy(input + plane * plane_size, input + (plane + 1) * plane_size,
              output + out_plane * plane_size);
  }
}

template <typename Type>
void FlipImpl(Type *output, const Type *input,
              TensorShape<sample_ndim> shape, bool flip_z, bool flip_y, bool flip_x) {
  auto frame_size = volume(&shape[1], &shape[sample_ndim]);
  if (flip_x || flip_y) {
    for (Index f = 0; f < shape[0]; ++f) {
      OcvFlip(output, input, shape[1], shape[2], shape[3], shape[4], flip_z, flip_y, flip_x);
      output += frame_size;
      input += frame_size;
    }
  } else {
    for (Index f = 0; f < shape[0]; ++f) {
      FlipZAxis(output, input, shape[1], shape[2], shape[3], shape[4], flip_z);
      output += frame_size;
      input += frame_size;
    }
  }
}

}  // namespace cpu
}  // namespace detail

template <typename Type>
class DLL_PUBLIC FlipCPU {
 public:
  DLL_PUBLIC FlipCPU() = default;

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InTensorCPU<Type, sample_ndim> &in) {
    KernelRequirements req;
    req.output_shapes = {TensorListShape<DynamicDimensions>({in.shape})};
    return req;
  }

  DLL_PUBLIC void Run(KernelContext &Context, OutTensorCPU<Type, sample_ndim> &out,
      const InTensorCPU<Type, sample_ndim> &in, bool flip_z, bool flip_y, bool flip_x) {
    auto in_data = in.data;
    auto out_data = out.data;
    detail::cpu::FlipImpl(out_data, in_data, in.shape, flip_z, flip_y, flip_x);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_FLIP_CPU_H_
