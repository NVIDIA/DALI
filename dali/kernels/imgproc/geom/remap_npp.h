// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_GEOM_REMAP_NPP_H_
#define DALI_KERNELS_IMGPROC_GEOM_REMAP_NPP_H_

#include <nppi_geometry_transforms.h>
#include "dali/core/geom/box.h"
#include "dali/core/span.h"
#include "dali/kernels/imgproc/geom/npp_remap_call.h"
#include "dali/kernels/imgproc/geom/remap.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

namespace dali::kernels::remap {

namespace detail {

/**
 * Return a default ROI. Default ROI is a whole image.
 */
template<int ndims>
Box<2, int64_t> default_roi(const TensorShape<ndims> &ts) {
  assert(ts.sample_dim() >= 2);
  return {{0,     0},
          {ts[1], ts[0]}};
}

}  // namespace detail

/**
 * RemapKernel implementation using NPP.
 *
 * @remark NPP doesn't offer Border policy. When using this implementation,
 *         the border is always BoundaryType::CONSTANT and equal to 0.
 *
 * @see RemapKernel.
 */
template<typename Backend, typename T>
struct NppRemapKernel : public RemapKernel<Backend, T> {
  using Border = typename RemapKernel<Backend, T>::Border;
  using MapType = typename RemapKernel<Backend, T>::MapType;


  void Run(KernelContext &context,
           TensorListView<Backend, T> output,
           TensorListView<Backend, const T> input,
           TensorListView<Backend, const MapType, 2> mapsx,
           TensorListView<Backend, const MapType, 2> mapsy,
           span<Box<2, int64_t>> output_rois = {},
           span<Box<2, int64_t>> input_rois = {},
           span<DALIInterpType> interpolations = {},
           span<Border> borders = {}) override {
    CUDA_CALL(nppSetStream(context.gpu.stream));
    NppStreamContext npp_ctx;
    CUDA_CALL(nppGetStreamContext(&npp_ctx));
    for (int sample_id = 0; sample_id < input.num_samples(); sample_id++) {
      const auto &inp = input[sample_id];
      const auto &out = output[sample_id];
      const auto &mx = mapsx[sample_id];
      const auto &my = mapsy[sample_id];
      auto oroi = !output_rois.empty() ? output_rois[sample_id] : detail::default_roi(inp.shape);
      auto iroi = !input_rois.empty() ? input_rois[sample_id] : detail::default_roi(inp.shape);
      auto interp = !interpolations.empty() ? interpolations[sample_id] : DALI_INTERP_LINEAR;

      invoke_remap_kernel(inp, out, mx, my, iroi, oroi, to_npp(interp), npp_ctx);
    }
  }


  void Run(KernelContext &context,
           TensorListView<Backend, T> output,
           TensorListView<Backend, const T> input,
           TensorView<Backend, const MapType, 2> mapx,
           TensorView<Backend, const MapType, 2> mapy,
           Box<2, int64_t> output_roi = {},
           Box<2, int64_t> input_roi = {},
           DALIInterpType interpolation = DALI_INTERP_LINEAR,
           Border border = {}) override {
    CUDA_CALL(nppSetStream(context.gpu.stream));
    NppStreamContext npp_ctx;
    CUDA_CALL(nppGetStreamContext(&npp_ctx));
    for (int sample_id = 0; sample_id < input.num_samples(); sample_id++) {
      const auto &inp = input[sample_id];
      const auto &out = output[sample_id];
      invoke_remap_kernel(inp, out, mapx, mapy,
                          !input_roi.empty() ? input_roi : detail::default_roi(inp.shape),
                          !output_roi.empty() ? output_roi : detail::default_roi(inp.shape),
                          to_npp(interpolation), npp_ctx);
    }
  }


 private:
  void invoke_remap_kernel(TensorView<Backend, const T> input,
                           TensorView<Backend, T> output,
                           TensorView<Backend, const MapType, 2> mapx,
                           TensorView<Backend, const MapType, 2> mapy,
                           Box<2, int64_t> input_roi,
                           Box<2, int64_t> output_roi,
                           int interpolation, NppStreamContext ctx) {
    assert(input.shape[2] == 3);
    assert(input.shape == output.shape);
    CUDA_CALL(detail::npp_remap_call(
            input.data,
            {static_cast<int>(input.shape[1]), static_cast<int>(input.shape[0])},
            static_cast<int>(input.shape[1]) * input.shape[2] * sizeof(T),
            {static_cast<int>(input_roi.lo.x), static_cast<int>(input_roi.lo.y),
             static_cast<int>(input_roi.extent().x), static_cast<int>(input_roi.extent().y)},
            mapx.data,
            static_cast<int>(mapx.shape[1]) * sizeof(float),
            mapy.data,
            static_cast<int>(mapy.shape[1]) * sizeof(float),
            output.data,
            static_cast<int>(output.shape[1]) * output.shape[2] * sizeof(T),
            {static_cast<int>(output_roi.extent().x), static_cast<int>(output_roi.extent().y)},
            interpolation,
            ctx));
  }
};

}  // namespace dali::kernels::remap

#endif  // DALI_KERNELS_IMGPROC_GEOM_REMAP_NPP_H_
