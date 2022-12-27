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
#include <tuple>
#include "dali/core/geom/box.h"
#include "dali/core/span.h"
#include "dali/kernels/imgproc/geom/npp_remap_call.h"
#include "dali/kernels/imgproc/geom/remap.h"
#include "include/dali/core/backend_tags.h"
#include "include/dali/core/tensor_view.h"

namespace dali {
namespace kernels {
namespace remap {

namespace detail {

/**
 * Return a default ROI. Default ROI is a whole image.
 */
template<int ndims>
Roi<2> default_roi(const TensorShape<ndims> &ts) {
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
 *
 * @tparam Backend Storage backend, must be GPU-accessible.
 * @tparam T Type of the input and output data.
 */
template<typename Backend, typename T>
struct NppRemapKernel : public RemapKernel<Backend, T> {
  using Border = typename RemapKernel<Backend, T>::Border;
  using MapType = typename RemapKernel<Backend, T>::MapType;
  using SupportedInputTypes = std::tuple<uint8_t, int16_t, uint16_t, float>;
  static_assert(contains_v<T, SupportedInputTypes>, "Unsupported input type.");


  explicit NppRemapKernel(int device_id) : npp_ctx_{CreateNppContext(device_id)} {}


  virtual ~NppRemapKernel() = default;


  void Run(KernelContext &context,
           TensorListView<Backend, T> output,
           TensorListView<Backend, const T> input,
           TensorListView<Backend, const MapType, 2> mapsx,
           TensorListView<Backend, const MapType, 2> mapsy,
           span<const Roi<2>> output_rois = {},
           span<const Roi<2>> input_rois = {},
           span<DALIInterpType> interpolations = {},
           span<Border> borders = {}) override {
    DALI_ENFORCE(output.num_samples() == input.num_samples(),
                 make_string("Incorrect number of output samples passed. Got ",
                             output.num_samples(), " output samples, but ", input.num_samples(),
                             " input samples."));
    DALI_ENFORCE(mapsx.shape == mapsy.shape,
                 make_string("Maps shapes do not match. mapsx: ", mapsx.shape, " ; mapsy: ",
                             mapsy.shape, "."));
    DALI_ENFORCE(output_rois.empty() || output_rois.size() == input.num_samples(),
                 make_string("Incorrect number of output_rois passed. Got ", output_rois.size(),
                             " output_rois, but ", input.num_samples(), " input samples."));
    DALI_ENFORCE(input_rois.empty() || input_rois.size() == input.num_samples(),
                 make_string("Incorrect number of input_rois passed. Got ", input_rois.size(),
                             " input_rois, but ", input.num_samples(), " input samples."));
    DALI_ENFORCE(interpolations.empty() || interpolations.size() == input.num_samples(),
                 make_string("Incorrect number of interpolations passed. Got ",
                             interpolations.size(), " interpolations, but ", input.num_samples(),
                             " input samples."));
    DALI_ENFORCE(borders.empty() || borders.size() == input.num_samples(),
                 make_string("Incorrect number of borders passed. Got ", borders.size(),
                             " borders, but ", input.num_samples(), " input samples."));
    if (output_rois.empty()) {
      for (int i = 0; i < output.num_samples(); i++) {
        for (int j = 0; j < mapsx.tensor_shape_span(i).size(); j++) {  // Omit the channel dimension
          DALI_ENFORCE(output.tensor_shape_span(i)[j] == mapsx.tensor_shape_span(i)[j],
                       make_string("Output and maps shapes do not match. Output: ",
                                   output.template tensor_shape(i), " ; mapsx: ",
                                   mapsx.template tensor_shape(i), "."));
        }
      }
    } else {
      DALI_ENFORCE(mapsx.shape == ShapeFromRoi(output_rois),
                   make_string("Shapes of maps and ROI don't match. ROI: ",
                               ShapeFromRoi(output_rois), " ; mapx: ", mapsx.shape, "."));
    }

    UpdateNppContextStream(npp_ctx_, context.gpu.stream);
    for (int sample_id = 0; sample_id < input.num_samples(); sample_id++) {
      const auto &inp = input[sample_id];
      const auto &out = output[sample_id];
      const auto &mx = mapsx[sample_id];
      const auto &my = mapsy[sample_id];
      auto oroi = !output_rois.empty() ? output_rois[sample_id] : detail::default_roi(inp.shape);
      auto iroi = !input_rois.empty() ? input_rois[sample_id] : detail::default_roi(inp.shape);
      auto interp = !interpolations.empty() ? interpolations[sample_id] : DALI_INTERP_LINEAR;

      invoke_remap_kernel(inp, out, mx, my, iroi, oroi, to_npp(interp), npp_ctx_);
    }
  }


  void Run(KernelContext &context,
           TensorListView<Backend, T> output,
           TensorListView<Backend, const T> input,
           TensorView<Backend, const MapType, 2> mapx,
           TensorView<Backend, const MapType, 2> mapy,
           Roi<2> output_roi = {},
           Roi<2> input_roi = {},
           DALIInterpType interpolation = DALI_INTERP_LINEAR,
           Border border = {}) override {
    DALI_ENFORCE(mapx.shape == mapy.shape,
                 make_string("Maps shapes do not match. mapx: ", mapx.shape, " ; mapy: ",
                             mapy.shape, "."));
    if (output_roi.empty()) {
      for (int i = 0; i < output.num_samples(); i++) {
        for (int j = 0; j < mapx.shape.sample_dim(); j++) {  // Omit the channel dimension
          DALI_ENFORCE(output.tensor_shape_span(i)[j] == mapx.shape[j],
                       make_string("Output and maps shapes do not match. Output: ",
                                   output.template tensor_shape(i), " ; mapsx: ",
                                   mapx.shape, "."));
        }
      }
    } else {
      DALI_ENFORCE(mapx.shape == ShapeFromRoi(output_roi),
                   make_string("Shapes of maps and ROI don't match. ROI: ",
                               ShapeFromRoi(output_roi), " ; mapx: ", mapx.shape, "."));
    }

    UpdateNppContextStream(npp_ctx_, context.gpu.stream);
    for (int sample_id = 0; sample_id < input.num_samples(); sample_id++) {
      const auto &inp = input[sample_id];
      const auto &out = output[sample_id];
      invoke_remap_kernel(inp, out, mapx, mapy,
                          !input_roi.empty() ? input_roi : detail::default_roi(inp.shape),
                          !output_roi.empty() ? output_roi : detail::default_roi(inp.shape),
                          to_npp(interpolation), npp_ctx_);
    }
  }


 private:
  void invoke_remap_kernel(TensorView<Backend, const T> input,
                           TensorView<Backend, T> output,
                           TensorView<Backend, const MapType, 2> mapx,
                           TensorView<Backend, const MapType, 2> mapy,
                           Roi<2> input_roi,
                           Roi<2> output_roi,
                           int interpolation, NppStreamContext ctx) {
    auto nchannels = input.shape[2];
    if (nchannels == 1) {
      CUDA_CALL(detail::npp_remap_call<1>(
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
    } else if (nchannels == 3) {
      CUDA_CALL(detail::npp_remap_call<3>(
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
    } else {
      DALI_FAIL(make_string("Incorrect number of channels: ", nchannels,
                            ". Only images with 1 and 3 channels are supported."));
    }
  }


  NppStreamContext npp_ctx_{cudaStream_t(-1), 0};
};

}  // namespace remap
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_GEOM_REMAP_NPP_H_
