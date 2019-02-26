// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <utility>
#include <vector>
#include "dali/pipeline/operators/color_space/color_space_conversion.h"
#include "dali/util/npp.h"

namespace dali {

namespace detail {

template<typename T = uint8_t>
__global__ void ConvertRGBToBGRKernel(const T *input, T *output, unsigned int total_pixels) {
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const unsigned int C = 3;
  const T* pixel_in = &input[idx * C];
  T* pixel_out = &output[idx * C];

  T tmp[3] = { pixel_in[0], pixel_in[1], pixel_in[2] };
  pixel_out[0] = tmp[2];
  pixel_out[1] = tmp[1];
  pixel_out[2] = tmp[0];
}

auto ConvertBGRToRGB8uKernel = ConvertRGBToBGRKernel<uint8_t>;
auto ConvertRGBToBGR8uKernel = ConvertRGBToBGRKernel<uint8_t>;


template<typename T = uint8_t>
__global__ void ConvertGrayToRGBKernel(const T *input, T *output, unsigned int total_pixels) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const T pixel_in = input[idx];
  const unsigned int C = 3;
  T* pixel_out = &output[idx * C];
  pixel_out[0] = pixel_in;
  pixel_out[1] = pixel_in;
  pixel_out[2] = pixel_in;
}

auto ConvertGrayToRGB8uKernel = ConvertGrayToRGBKernel<uint8_t>;
auto ConvertGrayToBGR8uKernel = ConvertGrayToRGBKernel<uint8_t>;

__global__ void ConvertGrayToYCbCr8uKernel(const uint8_t *input, uint8_t *output,
                                           unsigned int total_pixels) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const uint8_t pixel_in = input[idx];
  const unsigned int C = 3;
  uint8_t* pixel_out = &output[idx * C];
  pixel_out[0] = pixel_in;
  pixel_out[1] = 128;
  pixel_out[2] = 128;
}

template<typename T = uint8_t>
__global__ void ConvertYCbCrToGrayKernel(const T *input, T *output, unsigned int total_pixels) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const unsigned int C = 3;
  output[idx] = input[idx * C];
}

auto ConvertYCbCrToGray8uKernel = ConvertYCbCrToGrayKernel<uint8_t>;

template<typename T = uint8_t, int pos_R = 0, int pos_G = 1, int pos_B = 2, int pos_A = 3>
__global__ void ConvertYCbCrToRGBAKernel(const T *input, T *output, unsigned int total_pixels) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const unsigned int C_input = 3;
  const unsigned int C_output = 4;
  const T* pixel_in = &input[idx * C_input];
  T* pixel_out = &output[idx * C_output];

  T tmp[3] = { pixel_in[0], pixel_in[1], pixel_in[2] };
  float nY = 1.164f * ((float)tmp[0] - 16.0f);
  float nB = ((float)tmp[1] - 128.0f);
  float nR = ((float)tmp[2] - 128.0f);
  float nG = nY - 0.813f * nR - 0.392f * nB;
  if (nG > 255.0f)
    nG = 255.0f;
  nR = nY + 1.596f * nR;
  if (nR > 255.0f)
    nR = 255.0f;
  nB = nY + 2.017f * nB;
  if (nB > 255.0f)
    nB = 255.0f;
  pixel_out[pos_R] = (T) nR;
  pixel_out[pos_G] = (T) nG;
  pixel_out[pos_B] = (T) nB;
  pixel_out[pos_A] = 255;
}

auto ConvertYCbCrToRGBA8uKernel = ConvertYCbCrToRGBAKernel<uint8_t, 0, 1, 2, 3>;
auto ConvertYCbCrToBGRA8uKernel = ConvertYCbCrToRGBAKernel<uint8_t, 2, 1, 0, 3>;
auto ConvertYCbCrToARGB8uKernel = ConvertYCbCrToRGBAKernel<uint8_t, 1, 2, 3, 0>;
auto ConvertYCbCrToABGR8uKernel = ConvertYCbCrToRGBAKernel<uint8_t, 3, 2, 1, 0>;

template<typename T = uint8_t, int pos_R = 0, int pos_G = 1, int pos_B = 2, int pos_A = 3>
__global__ void ConvertRGBToRGBAKernel(const T *input, T *output, unsigned int total_pixels) {
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const unsigned int C_input = 3;
  const unsigned int C_output = 4;
  const T* pixel_in = &input[idx * C_input];
  T* pixel_out = &output[idx * C_output];

  T tmp[3] = { pixel_in[0], pixel_in[1], pixel_in[2] };
  pixel_out[pos_R] = tmp[0];
  pixel_out[pos_G] = tmp[1];
  pixel_out[pos_B] = tmp[2];
  pixel_out[pos_A] = 255;
}

auto ConvertBGRToBGRA8uKernel = ConvertRGBToRGBAKernel<uint8_t, 0, 1, 2, 3>;
auto ConvertRGBToRGBA8uKernel = ConvertRGBToRGBAKernel<uint8_t, 0, 1, 2, 3>;
auto ConvertBGRToABGR8uKernel = ConvertRGBToRGBAKernel<uint8_t, 1, 2, 3, 0>;
auto ConvertRGBToARGB8uKernel = ConvertRGBToRGBAKernel<uint8_t, 1, 2, 3, 0>;
auto ConvertRGBToBGRA8uKernel = ConvertRGBToRGBAKernel<uint8_t, 2, 1, 0, 3>;
auto ConvertBGRToRGBA8uKernel = ConvertRGBToRGBAKernel<uint8_t, 2, 1, 0, 3>;
auto ConvertRGBToABGR8uKernel = ConvertRGBToRGBAKernel<uint8_t, 3, 2, 1, 0>;
auto ConvertBGRToARGB8uKernel = ConvertRGBToRGBAKernel<uint8_t, 3, 2, 1, 0>;

template<typename T = uint8_t, int pos_R = 0, int pos_G = 1, int pos_B = 2, int pos_A = 3>
__global__ void ConvertGrayToRGBAKernel(const T *input, T *output, unsigned int total_pixels) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const T pixel_in = input[idx];
  const unsigned int C = 4;
  T* pixel_out = &output[idx * C];
  pixel_out[pos_R] = pixel_in;
  pixel_out[pos_G] = pixel_in;
  pixel_out[pos_B] = pixel_in;
  pixel_out[pos_A] = 255;
}

auto ConvertGrayToRGBA8uKernel = ConvertGrayToRGBAKernel<uint8_t, 0, 1, 2, 3>;
auto ConvertGrayToBGRA8uKernel = ConvertGrayToRGBAKernel<uint8_t, 0, 1, 2, 3>;
auto ConvertGrayToARGB8uKernel = ConvertGrayToRGBAKernel<uint8_t, 1, 2, 3, 0>;
auto ConvertGrayToABGR8uKernel = ConvertGrayToRGBAKernel<uint8_t, 1, 2, 3, 0>;

}  // namespace detail

template<>
void ColorSpaceConversion<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8_t>(input.type()),
      "Color space conversion accept only uint8 tensors");
  auto &output = ws->Output<GPUBackend>(idx);

  TensorList<CPUBackend> attr_output_cpu;

  int input_C = NumberOfChannels(input_type_);
  int output_C = NumberOfChannels(output_type_);
  const vector<Dims> input_shape = input.shape();
  vector<Dims> output_shape = input_shape;
  if ( input_C != output_C ) {
    for (unsigned int i = 0; i < input.ntensor(); ++i) {
      DALI_ENFORCE(input_shape[i][2] == input_C,
        "Wrong number of channels for input");
      output_shape[i][2] = output_C;
    }
  }
  output.Resize(output_shape);
  output.set_type(input.type());

  if (input_type_ == output_type_) {
    CUDA_CALL(cudaMemcpyAsync(
      output.raw_mutable_data(),
      input.raw_data(),
      input.nbytes(),
      cudaMemcpyDeviceToDevice,
      ws->stream()));
    return;
  }

  cudaStream_t old_stream = nppGetStream();
  auto stream = ws->stream();
  nppSetStream(stream);

  DALI_ENFORCE(input.GetLayout() == DALI_NHWC,
    "Only NHWC layout is supported");

  // RGB -> BGR || BGR -> RGB
  for (unsigned int i = 0; i < input.ntensor(); ++i) {
    // image dimensions
    NppiSize size;
    size.height = input.tensor_shape(i)[0];
    size.width = input.tensor_shape(i)[1];

    // For CUDA kernel
    const auto shape = input.tensor_shape(i);
    const unsigned int total_size = size.height * size.width;
    const unsigned int block = total_size < 1024 ? total_size : 1024;
    const unsigned int grid = (total_size + block - 1) / block;

    // For NPPI calls
    const int nStepInput = input_C * size.width;
    const int nStepOutput = output_C * size.width;

    // input/output
    const uint8_t* input_data = input.tensor<uint8_t>(i);
    uint8_t* output_data = output.mutable_tensor<uint8_t>(i);

    using ImageTypePair = std::pair<DALIImageType, DALIImageType>;
    ImageTypePair conversion { input_type_, output_type_};

    const ImageTypePair kRGB_TO_BGR { DALI_RGB, DALI_BGR };
    const ImageTypePair kBGR_TO_RGB { DALI_BGR, DALI_RGB };
    const ImageTypePair kRGB_TO_YCbCr { DALI_RGB, DALI_YCbCr };
    const ImageTypePair kBGR_TO_YCbCr { DALI_BGR, DALI_YCbCr };
    const ImageTypePair kRGB_TO_GRAY { DALI_RGB, DALI_GRAY };
    const ImageTypePair kBGR_TO_GRAY { DALI_BGR, DALI_GRAY };
    const ImageTypePair kYCbCr_TO_BGR { DALI_YCbCr, DALI_BGR };
    const ImageTypePair kYCbCr_TO_RGB { DALI_YCbCr, DALI_RGB };
    const ImageTypePair kYCbCr_TO_GRAY { DALI_YCbCr, DALI_GRAY };
    const ImageTypePair kGRAY_TO_RGB { DALI_GRAY, DALI_RGB };
    const ImageTypePair kGRAY_TO_BGR { DALI_GRAY, DALI_BGR };
    const ImageTypePair kGRAY_TO_YCbCr { DALI_GRAY, DALI_YCbCr };
    const ImageTypePair kRGB_TO_RGBA { DALI_RGB, DALI_RGBA };
    const ImageTypePair kBGR_TO_BGRA { DALI_BGR, DALI_BGRA };
    const ImageTypePair kRGB_TO_BGRA { DALI_RGB, DALI_BGRA };
    const ImageTypePair kBGR_TO_RGBA { DALI_BGR, DALI_RGBA };
    const ImageTypePair kGray_TO_RGBA { DALI_GRAY, DALI_RGBA };
    const ImageTypePair kGray_TO_BGRA { DALI_GRAY, DALI_BGRA };
    const ImageTypePair kRGB_TO_ARGB { DALI_RGB, DALI_ARGB };
    const ImageTypePair kBGR_TO_ABGR { DALI_BGR, DALI_ABGR };
    const ImageTypePair kRGB_TO_ABGR { DALI_RGB, DALI_ABGR };
    const ImageTypePair kBGR_TO_ARGB { DALI_BGR, DALI_ARGB };
    const ImageTypePair kGray_TO_ARGB { DALI_GRAY, DALI_ARGB };
    const ImageTypePair kGray_TO_ABGR { DALI_GRAY, DALI_ABGR };
    const ImageTypePair kYCbCr_TO_RGBA { DALI_YCbCr, DALI_RGBA };
    const ImageTypePair kYCbCr_TO_BGRA { DALI_YCbCr, DALI_BGRA };
    const ImageTypePair kYCbCr_TO_ARGB { DALI_YCbCr, DALI_ARGB };
    const ImageTypePair kYCbCr_TO_ABGR { DALI_YCbCr, DALI_ABGR };


    if (conversion == kRGB_TO_BGR || conversion == kBGR_TO_RGB) {
      // RGB -> BGR
      // BGR -> RGB
      detail::ConvertRGBToBGR8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kRGB_TO_YCbCr) {
      // RGB -> YCbCr
      DALI_CHECK_NPP(
        nppiRGBToYCbCr_8u_C3R(
          input_data, nStepInput, output_data, nStepOutput, size));
    } else if (conversion == kBGR_TO_YCbCr) {
      // BGR -> YCbCr
      // First from BGR to RGB
      detail::ConvertBGRToRGB8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
      // Then from RGB to YCbCr
      DALI_CHECK_NPP(
        nppiRGBToYCbCr_8u_C3R(
          output_data, nStepOutput, output_data, nStepOutput, size));
    } else if (conversion == kRGB_TO_GRAY) {
      // RGB -> GRAY
      DALI_CHECK_NPP(
        nppiRGBToGray_8u_C3C1R(
          input_data, nStepInput, output_data, nStepOutput, size));
    } else if (conversion == kBGR_TO_GRAY) {
      // BGR -> GRAY
      const Npp32f aCoefs[3] = {0.114f, 0.587f, 0.299f};
      DALI_CHECK_NPP(
        nppiColorToGray_8u_C3C1R(
          input_data, nStepInput, output_data, nStepOutput, size, aCoefs));
    } else if (conversion == kYCbCr_TO_BGR) {
      // First from YCbCr to RGB
      DALI_CHECK_NPP(
        nppiYCbCrToRGB_8u_C3R(
          input_data, nStepInput, output_data, nStepOutput, size));
      // Then from RGB to BGR
      detail::ConvertRGBToBGR8uKernel<<<grid, block, 0, stream>>>(
        output_data, output_data, total_size);
    } else if (conversion == kYCbCr_TO_RGB) {
      // First from YCbCr to RGB
      DALI_CHECK_NPP(
        nppiYCbCrToRGB_8u_C3R(
          input_data, nStepInput, output_data, nStepOutput, size));
    } else if (conversion == kGRAY_TO_BGR || conversion == kGRAY_TO_RGB) {
      // GRAY -> RGB / BGR
      detail::ConvertGrayToRGB8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kGRAY_TO_YCbCr) {
      // GRAY -> YCbCr
      detail::ConvertGrayToYCbCr8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kYCbCr_TO_GRAY) {
      // YCbCr -> GRAY
      detail::ConvertYCbCrToGray8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kRGB_TO_RGBA || conversion == kBGR_TO_BGRA) {
      // RGB -> RGBA or BGR -> BGRA
      detail::ConvertRGBToRGBA8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kRGB_TO_BGRA || conversion == kBGR_TO_RGBA) {
      // RGB -> BGRA or BGR -> RGBA
      detail::ConvertRGBToBGRA8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kGray_TO_RGBA || conversion == kGray_TO_BGRA) {
      // GRAY -> BGRA or GRAY -> RGBA
      detail::ConvertGrayToRGBA8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kRGB_TO_ARGB || conversion == kBGR_TO_ABGR) {
      // RGB -> ARGB or BGR -> ABGR
      detail::ConvertRGBToARGB8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kRGB_TO_ABGR || conversion == kBGR_TO_ARGB) {
      // RGB -> ABGR or BGR -> ARGB
      detail::ConvertRGBToABGR8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kGray_TO_ARGB || conversion == kGray_TO_ABGR) {
      // GRAY -> ABGR or GRAY -> ARGB
      detail::ConvertGrayToARGB8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kYCbCr_TO_RGBA) {
      // YCbCr to RGBA
      detail::ConvertYCbCrToRGBA8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kYCbCr_TO_BGRA) {
      // YCbCr to BGRA
      detail::ConvertYCbCrToBGRA8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kYCbCr_TO_ARGB) {
      // YCbCr to ARGB
      detail::ConvertYCbCrToARGB8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else if (conversion == kYCbCr_TO_ABGR) {
      // YCbCr to ABGR
      detail::ConvertYCbCrToABGR8uKernel<<<grid, block, 0, stream>>>(
        input_data, output_data, total_size);
    } else {
      DALI_FAIL("conversion from "
        + std::to_string(input_type_) + " to "
        + std::to_string(output_type_) + " not supported");
    }
  }

  nppSetStream(old_stream);
}

DALI_REGISTER_OPERATOR(ColorSpaceConversion, ColorSpaceConversion<GPUBackend>, GPU);

}  // namespace dali
