// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/static_switch.h"
#include "dali/core/util.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_kernel.cuh"
#include "dali/operators/decoder/nvjpeg/permute_layout.h"

namespace dali {

template <int C, typename Output, typename Input>
__global__ void planar_to_interleaved(Output *output, const Input *input, int64_t npixels) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npixels) return;
  Output *out = output + C * tid;
  for (int c = 0; c < C; ++c) {
    out[c] = ConvertSatNorm<Output>(input[c * npixels + tid]);
  }
}

template <typename Output, typename Input>
__global__ void planar_rgb_to_bgr(Output *output, const Input *input, int64_t npixels) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npixels) return;
  Output r = ConvertSatNorm<Output>(input[tid]);
  Output g = ConvertSatNorm<Output>(input[tid + npixels]);
  Output b = ConvertSatNorm<Output>(input[tid + 2 * npixels]);
  Output *out = output + 3 * tid;
  out[0] = b;
  out[1] = g;
  out[2] = r;
}

template <typename Output, typename Input>
__global__ void planar_rgb_to_ycbcr(Output *output, const Input *input, int64_t npixels) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npixels) return;
  vec<3, float> rgb = {ConvertNorm<float>(input[tid]), ConvertNorm<float>(input[tid + npixels]),
                       ConvertNorm<float>(input[tid + 2 * npixels])};
  Output *out = output + 3 * tid;
  out[0] = kernels::color::itu_r_bt_601::rgb_to_y<Output>(rgb);
  out[1] = kernels::color::itu_r_bt_601::rgb_to_cb<Output>(rgb);
  out[2] = kernels::color::itu_r_bt_601::rgb_to_cr<Output>(rgb);
}

template <typename Output, typename Input>
__global__ void planar_rgb_to_gray(Output *output, const Input *input, int64_t npixels) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npixels) return;
  vec<3, float> rgb = {ConvertNorm<float>(input[tid]), ConvertNorm<float>(input[tid + npixels]),
                       ConvertNorm<float>(input[tid + 2 * npixels])};
  output[tid] = kernels::color::rgb_to_gray<Output>(rgb);
}

template <typename Output, typename Input>
void PlanarToInterleaved(Output *output, const Input *input, int64_t npixels,
                         int64_t comp_count, DALIImageType out_img_type, DALIDataType pixel_type,
                         cudaStream_t stream) {
  if (comp_count < 2) {
    CUDA_CALL(
      cudaMemcpyAsync(output, input, npixels * comp_count, cudaMemcpyDeviceToDevice, stream));
    return;
  }
  int num_blocks = div_ceil(npixels, 1024);
  int block_size = (npixels < 1024) ? npixels : 1024;

  if (out_img_type == DALI_RGB || out_img_type == DALI_ANY_DATA) {
    VALUE_SWITCH(comp_count, c_static, (2, 3, 4), (
      planar_to_interleaved<c_static>
        <<<num_blocks, block_size, 0, stream>>>(output, input, npixels);
    ), DALI_FAIL(make_string("Unsupported number of components: ", comp_count)););  // NOLINT
  } else if (out_img_type == DALI_BGR) {
    planar_rgb_to_bgr<<<num_blocks, block_size, 0, stream>>>(output, input, npixels);
  } else if (out_img_type == DALI_YCbCr) {
    planar_rgb_to_ycbcr<<<num_blocks, block_size, 0, stream>>>(output, input, npixels);
  } else {
    assert(false);
  }
  CUDA_CALL(cudaGetLastError());
}

template <typename Output, typename Input>
void PlanarRGBToGray(Output *output, const Input *input, int64_t npixels,
                     DALIDataType pixel_type, cudaStream_t stream) {
  int num_blocks = div_ceil(npixels, 1024);
  int block_size = (npixels < 1024) ? npixels : 1024;
  planar_rgb_to_gray<<<num_blocks, block_size, 0, stream>>>(output, input, npixels);
  CUDA_CALL(cudaGetLastError());
}

template <typename Output, typename Input>
void Convert_RGB_to_YCbCr(Output *out_data, const Input *in_data, int64_t npixels,
                          cudaStream_t stream) {
  kernels::color::RunColorSpaceConversionKernel(out_data, in_data, DALI_YCbCr, DALI_RGB, npixels,
                                                stream);
}


template void PlanarToInterleaved<uint8_t, uint16_t>(uint8_t *, const uint16_t *, int64_t, int64_t,
                                                     DALIImageType, DALIDataType, cudaStream_t);
template void PlanarToInterleaved<uint8_t, uint8_t>(uint8_t *, const uint8_t *, int64_t, int64_t,
                                                    DALIImageType, DALIDataType, cudaStream_t);
template void PlanarRGBToGray<uint8_t, uint16_t>(uint8_t *, const uint16_t *, int64_t, DALIDataType,
                                                 cudaStream_t);
template void PlanarRGBToGray<uint8_t, uint8_t>(uint8_t *, const uint8_t *, int64_t, DALIDataType,
                                                cudaStream_t);
template void Convert_RGB_to_YCbCr<uint8_t, uint8_t>(uint8_t *, const uint8_t *, int64_t,
                                                     cudaStream_t);

}  // namespace dali
