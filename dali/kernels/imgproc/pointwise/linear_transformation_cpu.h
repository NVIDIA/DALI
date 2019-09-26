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

#ifndef DALI_KERNELS_IMGPROC_POINTWISE_LINEAR_TRANSFORMATION_CPU_H_
#define DALI_KERNELS_IMGPROC_POINTWISE_LINEAR_TRANSFORMATION_CPU_H_

#include <vector>
#include <utility>
#include "dali/core/format.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/roi.h"

namespace dali {
namespace kernels {

template <typename OutputType, typename InputType,
        int channels_out, int channels_in, int spatial_ndims>
class LinearTransformationCpu {
 private:
  static constexpr auto ndims_ = spatial_ndims + 1;
  using Mat = ::dali::mat<channels_out, channels_in, float>;
  using Vec = ::dali::vec<channels_out, float>;

 public:
    KernelRequirements Setup(KernelContext &context, const InTensorCPU<InputType, ndims_> &in,
                             Mat tmatrix = eye<channels_out, channels_in>(), Vec tvector = {},
                             const Roi<spatial_ndims> *roi = nullptr) {
    DALI_ENFORCE(in.shape.shape.back() == channels_in,
                 "Unexpected number of channels. Number of channels in InTensorCPU has to match"
                 " the number of channels, that the kernel is instantiated with");
    DALI_ENFORCE(!roi || all_coords(roi->hi >= roi->lo),
                 make_string("Invalid ROI: it doesn't follow {lo, hi} convention. ", roi));

    auto adjusted_roi = AdjustRoi(roi, in.shape);
    KernelRequirements req;
    TensorListShape<ndims_> output_shape({ShapeFromRoi(adjusted_roi, channels_out)});
    req.output_shapes = {std::move(output_shape)};
    return req;
  }


    void Run(KernelContext &context, const OutTensorCPU<OutputType, spatial_ndims + 1> &out,
             const InTensorCPU<InputType, spatial_ndims + 1> &in,
             Mat tmatrix = eye<channels_out, channels_in>(), Vec tvector = {},
             const Roi<spatial_ndims> *roi = nullptr) {
    auto adjusted_roi = AdjustRoi(roi, in.shape);
    auto ptr = out.data;
    auto in_width = in.shape[1];

    for (int y = adjusted_roi.lo.y; y < adjusted_roi.lo.y + adjusted_roi.extent().y; y++) {
      for (int x = adjusted_roi.lo.x; x < adjusted_roi.lo.x + adjusted_roi.extent().x; x++) {
        vec<channels_in> v_in;
        for (int k = 0; k < channels_in; k++) {
          v_in[k] = in.data[(y * in_width + x) * channels_in + k];
        }
        vec<channels_out> v_out = tmatrix * v_in + tvector;
        for (int k = 0; k < channels_out; k++) {
          *ptr++ = ConvertSat<OutputType>(v_out[k]);
        }
      }
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_POINTWISE_LINEAR_TRANSFORMATION_CPU_H_
