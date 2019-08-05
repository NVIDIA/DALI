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

namespace dali {
namespace kernels {
namespace brightness_contrast {

namespace detail {

constexpr size_t kBlockDim = 32;


/**
 * Divides 2 numbers and calculates ceil() function for the result.
 */
template <typename T>
constexpr size_t divide_ceil(T num, T denom) {
  assert(num > 0 && denom > 0);
  return static_cast<size_t>((num + denom - 1) / denom);
}

/**
 * For given `row` calculates, what is the max index for which kernel should be ran.
 *
 * @param row Index of row (along Y axis), for which calculation is performed
 * @param startx Starting point of ROI, along X axis. [px]
 * @param inpitch Pitch of the input data. [px]
 * @param nchannels Number of channels of the image.
 * @param width Width of the output image. [px]
 */
DALI_HOST_DEV constexpr size_t
idxmax(size_t row, size_t startx, size_t inpitch, size_t nchannels, size_t width) {
  return (startx + row * inpitch + width) * nchannels;
}


/**
 * Brightness contrast CUDA kernel.
 *
 * Assumes HWC layout.
 *
 * Assumes, that `input` and `output` memory is properly initialized.
 *
 * Assumes, that both images have the same number of channels.
 *
 * @tparam InputType
 * @tparam OutputType
 * @param input
 * @param output
 * @param in_pitch Pitch of the input frame. [px]
 * @param out_pitch Pitch of the output frame. [px]
 * @param start_x Starting point of ROI, along X axis. [px]
 * @param start_y Starting point of ROI, along Y axis. [px]
 * @param out_width Width of the output image. [px]
 * @param out_height Height if the output image. [px]
 * @param nchannels Number of channels in both images.
 * @param brightness Additive brightness delta. 0 denotes no change.
 * @param contrast Multiplicative contrast delta. 1 denotes no change.
 */
template <class InputType, class OutputType>
__global__ void
BrightnessContrastKernel(const InputType *input, OutputType *output, size_t in_pitch,
                         size_t out_pitch, size_t start_x, size_t start_y, size_t out_width,
                         size_t out_height, size_t nchannels, float brightness, float contrast) {
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto idxout = (y * out_pitch + x) * nchannels;
  auto idxin = ((y + start_y) * in_pitch + (x + start_x)) * nchannels;
  if (y >= out_height || idxin >= idxmax(y, start_x, in_pitch, nchannels, out_width)) return;
  for (int channel = 0; channel < nchannels; channel++) {
    output[idxout + channel] = ConvertSat<OutputType>(
            input[idxin + channel] * contrast + brightness);
  }
//  printf("\n%lu %lu %lu %lu %lu %lu %lu %f %f %d %d %d %d %f %f", inpitch,  outpitch,  startx,  starty,
//           width,  height,  nchannels,  brightness,  contrast, x,y,idxin,idxout,input[idxin],output[idxout]);
}

}  // namespace detail

/**
 * The invocator of BrightnessContrast CUDA kernel, for no-ROI case.
 *
 * Assumes HWC layout.
 *
 * Assumes, that `input` and `output` memory is properly initialized.
 *
 * Assumes, that both images have the same number of channels.
 *
 * @tparam InputType
 * @tparam OutputType
 * @param input
 * @param output
 * @param output_width Width of the output image. [px]
 * @param output_height Height if the output image. [px]
 * @param nchannels Number of channels in both images.
 * @param brightness Additive brightness delta. 0 denotes no change.
 * @param contrast Multiplicative contrast delta. 1 denotes no change.
 */
template <class InputType, class OutputType>
void BrightnessContrastKernel(const InputType *input, OutputType *output, size_t output_width,
                              size_t output_height, size_t nchannels, float brightness,
                              float contrast) {

  dim3 blockdim(detail::kBlockDim, detail::kBlockDim);
  dim3 griddim(detail::divide_ceil(output_width, detail::kBlockDim),
               detail::divide_ceil(output_height, detail::kBlockDim));

  detail::BrightnessContrastKernel<<<griddim, blockdim>>>
      (input, output, output_width, output_width, 0, 0,
              output_width, output_height, nchannels, brightness, contrast);
}


/**
 * The invocator of BrightnessContrast CUDA kernel, for ROI case.
 *
 * Assumes HWC layout.
 *
 * Assumes, that `input` and `output` memory is properly initialized.
 *
 * Assumes, that both images have the same number of channels.
 *
 * @tparam InputType
 * @tparam OutputType
 * @param input
 * @param output
 * @param output_width Width of the output image. [px]
 * @param output_height Height if the output image. [px]
 * @param nchannels Number of channels in both images.
 * @param roi_x Starting point of ROI, along X axis. [px]
 * @param roi_y Starting point of ROI, along Y axis. [px]
 * @param roi_w Width of ROI. [px]
 * @param roi_h Height of ROI. [px]
 * @param brightness Additive brightness delta. 0 denotes no change.
 * @param contrast Multiplicative contrast delta. 1 denotes no change.
 */
template <class InputType, class OutputType>
void BrightnessContrastKernel(const InputType *input, OutputType *output, size_t output_width,
                              size_t output_height, size_t nchannels, size_t roi_x, size_t roi_y,
                              size_t roi_w, size_t roi_h, float brightness, float contrast) {
  dim3 blockdim(detail::kBlockDim, detail::kBlockDim);
  dim3 griddim(detail::divide_ceil(output_width, detail::kBlockDim),
               detail::divide_ceil(output_height, detail::kBlockDim));

  detail::BrightnessContrastKernel<<<griddim, blockdim>>>
      (input, output, output_width, roi_w,
              roi_x, roi_y, roi_w, roi_h, nchannels, brightness, contrast);
}


}  // namespace brightness_contrast
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_H_
