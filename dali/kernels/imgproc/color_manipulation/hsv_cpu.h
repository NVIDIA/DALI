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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_HSV_CPU_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_HSV_CPU_H_

#include <utility>
#include "dali/core/convert.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {

namespace hsv {

constexpr size_t kNdims = 3;
constexpr size_t kNchannels = 3;

}  // namespace hsv


template <class OutputType, class InputType>
class HsvCpu {
  // TODO(mszolucha): implement float16
  static_assert(!std::is_same<OutputType, float16>::value &&
                !std::is_same<InputType, float16>::value, "float16 not implemented yet");

 public:
  using Roi = ::dali::kernels::Roi<hsv::kNdims - 1>;

  KernelRequirements
  Setup(KernelContext &context, const InTensorCPU<InputType, hsv::kNdims> &in, float hue,
        float saturation, float value, const Roi *roi = nullptr) {
    DALI_ENFORCE(!roi || all_coords(roi->hi >= roi->lo), "Region of interest is invalid");
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    KernelRequirements req;
    TensorListShape<> out_shape({ShapeFromRoi(adjusted_roi, hsv::kNchannels)});
    req.output_shapes = {std::move(out_shape)};
    return req;
  }


  void Run(KernelContext &context, const OutTensorCPU<OutputType, hsv::kNdims> &out,
           const InTensorCPU<InputType, hsv::kNdims> &in, float hue, float saturation, float value,
           const Roi *roi = nullptr) {
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    auto num_channels = in.shape[2];
    auto image_width = in.shape[1];
    auto ptr = out.data;

    ptrdiff_t row_stride = image_width * num_channels;
    auto *row = in.data + adjusted_roi.lo.y * row_stride;
    for (int y = adjusted_roi.lo.y; y < adjusted_roi.hi.y; y++) {
      for (int x = adjusted_roi.lo.x; x < adjusted_roi.hi.x; x++) {
        auto elem = row + x * num_channels;
        *ptr++ = ConvertSat<OutputType>(*(elem + 0) + hue /*hue hue*/);
        *ptr++ = ConvertSat<OutputType>(*(elem + 1) * saturation);
        *ptr++ = ConvertSat<OutputType>(*(elem + 2) * value);
      }
      row += row_stride;
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_HSV_CPU_H_
