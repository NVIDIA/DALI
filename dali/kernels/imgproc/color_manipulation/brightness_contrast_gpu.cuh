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

using Type = float;

constexpr size_t kBlockDim = 32;


//__global__ void Bricon(Type *input, Type *output, size_t image_width, size_t image_height, size_t nchannels,
//                       float brightness, float contrast,
//                       size_t roi_x, size_t roi_y, size_t roi_width, size_t roi_height){
//  for (int i = 0; i < image_width * image_height * nchannels; i++) {
//    output[i] = input[i] * contrast + brightness;
//  }
//}

__host__ __device__ size_t
idxmax(size_t row, size_t startx, size_t inpitch, size_t nchannels, size_t width) {
  return (startx + row * inpitch + width) * nchannels;

}


__global__ void
Bricon(const Type *input, Type *output, size_t inpitch, size_t outpitch, size_t startx, size_t starty,
       size_t width, size_t height, size_t nchannels, float brightness, float contrast) {
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
//  printf("ASDASD %d %d %d\n",x,y,height);
  int idxout = (y * outpitch + x) * nchannels;
  int idxin = ((y + starty) * inpitch + (x + startx)) * nchannels;
  if (y >= height || idxin >= idxmax(y, startx, inpitch, nchannels, width)) return;
//  printf("\nDUPA %d %d %d %d %d %d %d %f %f",x,y,idxmax(y, startx, inpitch, nchannels, width),idxin, idxout,input[idxin],output[idxout]);
for (int channel=0;channel<nchannels;channel++) {
  output[idxout+channel] = input[idxin+channel] * contrast + brightness;
}
  printf("\n%lu %lu %lu %lu %lu %lu %lu %f %f %d %d %d %d %f %f", inpitch,  outpitch,  startx,  starty,
           width,  height,  nchannels,  brightness,  contrast, x,y,idxin,idxout,input[idxin],output[idxout]);
}


//void BrightnessContrastKernel(Type *input, Type *output, size_t image_width, size_t image_height, size_t nchannels,
//                              float brightness, float contrast,
//                              size_t roi_x, size_t roi_y, size_t roi_width, size_t roi_height
//                              ) {
//  Bricon<<<1,1>>>(input, output, image_width, image_height, nchannels, brightness, contrast, roi_x, roi_y, roi_width, roi_height);
//}

void BrightnessContrastKernel(const Type *input, Type *output, size_t image_width, size_t image_height,
                              size_t nchannels, float brightness, float contrast) {

//  Bricon<<<(static_cast<int>(image_width/kBlockDim),static_cast<int>(image_height/kBlockDim)),(kBlockDim,kBlockDim)>>>(input, output)
dim3 blockdim(kBlockDim, kBlockDim);
dim3 griddim(static_cast<int>(image_width/kBlockDim)+1, static_cast<int>(image_height/kBlockDim)+1);
Bricon<<<griddim,blockdim>>>
(input, output, image_width, image_width, 0,0,image_width, image_height, nchannels, brightness, contrast);
}

void BrightnessConstrastKernel(const Type*input, Type *output, size_t image_width, size_t image_height,
                               size_t nchannels, size_t x, size_t y, size_t w, size_t h, float brightness, float contrast) {
  dim3 blockdim(kBlockDim, kBlockDim);
  dim3 griddim(static_cast<int>(image_width/kBlockDim)+1, static_cast<int>(image_height/kBlockDim)+1);
  Bricon<<<griddim,blockdim>>>
(input, output, image_width, image_height, x, y, w, h, nchannels, brightness, contrast);
}


}  // namespace brightness_contrast



}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_BRIGHTNESS_CONTRAST_H_
