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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_GPU_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_GPU_H_

#include <cuda_runtime.h>
#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/data/types.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast.h"

namespace dali {
namespace kernels {
namespace brightness_contrast {


constexpr size_t kBlockDim = 32;

template <size_t ndims>
using Roi_ = Box<ndims, int>;
using Roi = Roi_<2>;

struct SampleDescriptor {

};


template <size_t ndims>
TensorListShape<ndims> calc_shape(const std::vector<Roi_<ndims>> &rois, int nchannels) {
  std::vector<TensorShape<ndims>> ret;
  for (auto roi : rois) {
    assert(all_coords(roi.hi >= roi.lo) && "Cannot create a tensor shape from an invalid Box");
    TensorShape<ndims> ts;
    auto e = roi.extent();
    auto ridx = ndims;
    ts[--ridx] = e[0] * nchannels;
    for (size_t idx = 1; idx < ndims; idx++) {
      ts[--ridx] = e[idx];
    }
    ret.emplace_back(ts);
  }
  return TensorListShape<ndims>{ret};
}


template <size_t ndims>
std::vector<Roi_<ndims>>
AdjustRois(const std::vector<Roi_<ndims>> rois, const TensorListShape<ndims + 1> &shapes) {
  assert(rois.size() == 0 || rois.size() == shapes.num_samples());
  std::vector<Roi_<ndims>> ret(shapes.num_samples());

  auto whole_image = [](auto shape) -> Roi_<ndims> {
      constexpr int spatial_dims = ndims;
      ivec<spatial_dims> size;
      for (size_t i = 0; i < spatial_dims; i++)
        size[i] = shape[spatial_dims - 1 - i];
      return {0, size};
  };

  if (rois.empty()) {
    for (int i = 0; i < shapes.num_samples(); i++) {
      ret[i] = whole_image(shapes[i]);
    }
  } else {
    for (int i = 0; i < rois.size(); i++) {
      ret[i] = intersection(rois[i], whole_image(shapes[i]));
    }
  }
  return ret;
}


template <typename InputType, typename OutputType, size_t ndims = 3>
class BrightnessContrastGpu {
 private:
  static constexpr size_t spatial_dims = ndims - 1;
//  using Roi = Roi_<spatial_dims>;
  using BlockDesc = kernels::BlockDesc<spatial_dims>;

 public:
  BlockSetup<spatial_dims, -1> block_setup_;


  KernelRequirements
  Setup(KernelContext &context, const InListGPU<InputType, ndims> &in,
        const std::vector<float> &brightness,
        const std::vector<float> &contrast, const std::vector<Roi> &rois = {}) {
    DALI_ENFORCE([=]() -> bool {
        for (const auto &roi:rois) {
          if (!all_coords(roi.hi >= roi.lo))
            return false;
        }
        return true;
    }(), "One or more regions of interests are invalid");
    DALI_ENFORCE(rois.size() == 0 || rois.size() == in.num_samples(),
                 "Provide ROIs either for all or none input tensors");
    auto adjusted_rois = AdjustRois(rois, in.shape);
    KernelRequirements req;
    ScratchpadEstimator se;
    TensorListShape<spatial_dims> output_shape({calc_shape(rois, 3)});
    block_setup_.SetupBlocks(output_shape);
    se.add<SampleDescriptor>(AllocType::GPU, output_shape.num_samples());
    se.add<BlockDesc>(AllocType::GPU, block_setup_.Blocks().size());
    req.output_shapes = {output_shape};
    req.scratch_sizes = se.sizes;
    return req;
  }


  void Run(KernelContext &context, const OutListGPU<OutputType, ndims> &out,
           const InListGPU<InputType, ndims> &in, const std::vector<float> &brightness,
           const std::vector<float> &contrast,
           const std::vector<Roi> &rois = {}) {

//    auto num_channels = in.shape[2];
//    auto image_width = in.shape[1];
//    auto ptr = out.data;
//
//    ptrdiff_t row_stride = image_width * num_channels;
//    auto *row = in.data + adjusted_roi.lo.y * row_stride;
//    for (int y = adjusted_roi.lo.y; y < adjusted_roi.hi.y; y++) {
//      for (int xc = adjusted_roi.lo.x * num_channels; xc < adjusted_roi.hi.x * num_channels; xc++)
//        *ptr++ = ConvertSat<OutputType>(row[xc] * contrast + brightness);
//      row += row_stride;
//    }
  }


 private:


};

}  // namespace brightness_contrast
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_H_
