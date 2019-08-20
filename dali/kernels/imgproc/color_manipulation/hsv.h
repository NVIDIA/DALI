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

#ifndef DALI_KERNELS_IMGPRASDFASFD_BRIGHTNESS_CONTRAST_H_
#define DALI_KERNELS_IMGPRASDFASFD_BRIGHTNESS_CONTRAST_H_

#include <utility>
#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace kernels {

namespace hsv {


constexpr size_t ndims = 3;
constexpr size_t nchannels = 3;


/**
 * - x poziomo y pionowo
 */
template<size_t ndims>
using Roi = Box<ndims, int>;


/**
 * Defines TensorShape corresponding to provided Roi.
 * Assumes HWC memory layout
 *
 * @tparam ndims_roi Number of dims in Roi
 * @param roi Region of interest
 * @param nchannels Number of channels in data
 * @return Corresponding TensorShape
 */
template<size_t ndims_roi>
TensorShape<ndims_roi + 1> roi_shape(Roi<ndims_roi> roi, size_t nchannels) {
    assert(all_coords(roi.hi >= roi.lo) && "Cannot create a tensor shape from an invalid Box");
    TensorShape<ndims_roi + 1> ret;
    auto e = roi.extent();
    auto ridx = ndims_roi;
    ret[ridx--] = nchannels;
    for (size_t idx = 0; idx < ndims_roi; idx++) {
        ret[ridx--] = e[idx];
    }
    return ret;
}

}  // namespace hsv


template<class OutputType, class InputType>
class HsvCpu {
private:
    static constexpr size_t spatial_dims = hsv::ndims - 1;

public:
    using Roi = Box<spatial_dims, int>;


    KernelRequirements
    Setup(KernelContext &context, const InTensorCPU<InputType, hsv::ndims> &in, const float hue,
          const float saturation, const float value, const Roi *roi = nullptr) {
        DALI_ENFORCE(!roi || all_coords(roi->hi >= roi->lo), "Region of interest is invalid");
        auto adjusted_roi = AdjustRoi(roi, in.shape);
        KernelRequirements req;
        TensorListShape<> out_shape({hsv::roi_shape(adjusted_roi, hsv::nchannels)});
        req.output_shapes = {std::move(out_shape)};
        return req;
    }


    void Run(KernelContext &context, const OutTensorCPU<OutputType, hsv::ndims> &out,
             const InTensorCPU<InputType, hsv::ndims> &in, float hue, float saturation, float value,
             const Roi *roi = nullptr) {
        auto adjusted_roi = AdjustRoi(roi, in.shape);
        auto num_channels = in.shape[2];
        auto image_width = in.shape[1];
        auto ptr = out.data;

        ptrdiff_t row_stride = image_width * num_channels;
        auto *row = in.data + adjusted_roi.lo.y * row_stride;
        for (int y = adjusted_roi.lo.y; y < adjusted_roi.hi.y; y++) {
            for (int x = adjusted_roi.lo.x; x < adjusted_roi.hi.x; x++) {
                *ptr++ = ConvertSat<OutputType>(row[x * num_channels + 0] + hue);
                *ptr++ = ConvertSat<OutputType>(row[x * num_channels + 1] * saturation);
                *ptr++ = ConvertSat<OutputType>(row[x * num_channels + 2] * value);
            }
            row += row_stride;
        }
    }


private:
    Roi AdjustRoi(const Roi *roi, const TensorShape<hsv::ndims> &shape) {
        ivec<spatial_dims> size;
        for (size_t i = 0; i < spatial_dims; i++)
            size[i] = shape[spatial_dims - 1 - i];
        Roi whole_image = {0, size};
        return roi ? intersection(*roi, whole_image) : whole_image;
    }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROadsON_BRIGHTNESS_CONTRAST_H_
