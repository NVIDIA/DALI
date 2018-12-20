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
#include "dali/pipeline/operators/color_space/color_space_conversion.h"

namespace dali {

namespace detail {

template<size_t C1, size_t C2, size_t C, typename T = uint8_t> 
__global__ void swap_packed_channels_kernel(const T *input, T *output, size_t total_pixels) {
  static_assert(C <= 4,           "C>4 not supported");
  static_assert(C1 < C && C2 < C, "C1 and C2 in the range 0 .. C-1");
  static_assert(C1 != C2,         "C1 and C2 cannot be the same");

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const T* pixel_in = &input[idx * C];
  T* pixel_out = &output[idx * C];

  memcpy(pixel_out, pixel_in, sizeof(T)*C);
  T tmp = pixel_out[C2];
  pixel_out[C2] = pixel_out[C1];
  pixel_out[C1] = tmp;
}

auto ConvertRGBToBGR8UKernel = &swap_packed_channels_kernel<0, 2, 3, uint8_t>;
auto ConvertBGRToRGB8UKernel = ConvertRGBToBGR8UKernel;

template<typename T = uint8_t> 
__global__ void ConvertRGBToGrayKernel(const T *input, T *output, size_t total_pixels) {
  // TODO(janton): make this a template parameter
  const size_t C = 3;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) {
    return;
  }

  const T* pixel_in = &input[idx * C];
  T* pixel_out = &output[idx];

  // TODO(janton): make this configurable
  const float coefs[] = { 0.333333333f, 0.333333333f, 0.333333333f };
  pixel_out[0] = static_cast<T>( 
      coefs[0] * pixel_in[0] 
    + coefs[1] * pixel_in[1] 
    + coefs[2] * pixel_in[2]);
}

auto ConvertRGBToGray8UKernel = ConvertRGBToGrayKernel<uint8_t>;

} // namespace detail

template<>
void ColorSpaceConversion<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  const auto &input = ws->Input<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8_t>(input.type()),
      "Color space conversion accept only uint8 tensors");
  auto output = ws->Output<GPUBackend>(idx);

  TensorList<CPUBackend> attr_output_cpu;
  
  auto input_C = NumberOfChannels(input_type_);
  auto output_C = NumberOfChannels(output_type_);
  vector<Dims> input_shape = input.shape();
  vector<Dims> output_shape = input_shape;
  if ( input_C != output_C ) {
    for (size_t i = 0; i < input.ntensor(); ++i) {
      DALI_ENFORCE(input_shape[i][2] == input_C, "Wrong number of channels for input");
      output_shape[i][2] = output_C;
    }
  }
  output->Resize(output_shape);
  output->set_type(input.type()); // TODO(janton): check

  cudaStream_t old_stream = nppGetStream();
  auto stream = ws->stream();
  nppSetStream(stream);

  // TODO(janton): Add support for NCHW ??
  DALI_ENFORCE(input.GetLayout() == DALI_NHWC,
      "Color space conversion accept only NHWC layout");
  
  // TODO(janton): remove this
  // using NppiConversionFunction = std::function<NppStatus(const Npp8u*, int, Npp8u*, int, NppiSize)>;

  if (input.GetLayout() == DALI_NHWC) {
    // RGB -> BGR || BGR -> RGB
    for (size_t i = 0; i < input.ntensor(); ++i) {
      // image dimensions
      DALISize size;
      size.height = input.tensor_shape(i)[0];
      size.width = input.tensor_shape(i)[1];
      
      // For CUDA kernel
      const auto shape = input.tensor_shape(i);
      const unsigned int total_size = size.height * size.width;
      const unsigned int block = total_size < 1024 ? total_size : 1024;
      const unsigned int grid = (total_size + block - 1) / block;

      // For NPPI calls
      const int input_C = input.tensor_shape(i)[2];
      DALI_ENFORCE( input_C == NumberOfChannels(input_type_), "Incorrect number of channels for input" );
      const int nStepInput = input_C * size.width;  // W * input_C
      const int nStepOutput = nStepInput;

      // input/output
      const uint8_t* input_data = input.tensor<uint8_t>(i);
      uint8_t* output_data = output->mutable_tensor<uint8_t>(i);

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

      if (conversion == kRGB_TO_BGR || conversion == kBGR_TO_RGB) {
        // RGB -> BGR
        // BGR -> RGB
      
        detail::ConvertRGBToBGR8UKernel<<<grid, block, 0, stream>>>(
          input_data, output_data, total_size);
      
      } else if (conversion == kRGB_TO_YCbCr) {
        // RGB -> YCbCr

        DALI_CHECK_NPP(
          nppiRGBToYCbCr_8u_C3R(
            input_data, nStepInput, output_data, nStepOutput, size)
        );
      } else if (conversion == kBGR_TO_YCbCr ) {
        // BGR -> YCbCr

        // First from BGR to RGB
        detail::ConvertBGRToRGB8UKernel<<<grid, block, 0, stream>>>(
          input_data, output_data, total_size);

        // Then from RGB to YCbCr
        DALI_CHECK_NPP(
          nppiRGBToYCbCr_8u_C3R(
            input_data, nStepInput, output_data, nStepOutput, size)
        );
      } else if (conversion == kRGB_TO_GRAY) {        
        // RGB -> GRAY

        detail::ConvertRGBToGray8UKernel<<<grid, block, 0, stream>>>(
          input_data, output_data, total_size);

      } else if (conversion == kBGR_TO_GRAY) {
        // BGR -> GRAY

        // First from BGR to RGB
        detail::ConvertBGRToRGB8UKernel<<<grid, block, 0, stream>>>(
          input_data, output_data, total_size);
        // Then RGB -> GRAY
        detail::ConvertRGBToGray8UKernel<<<grid, block, 0, stream>>>(
          input_data, output_data, total_size);
        
      } else if (conversion == kYCbCr_TO_BGR) {

        // First from YCbCr to RGB
        DALI_CHECK_NPP(
          nppiYCbCrToRGB_8u_C3R(
            input_data, nStepInput, output_data, nStepOutput, size)
        );

        // Then from RGB to BGR
        detail::ConvertRGBToBGR8UKernel<<<grid, block, 0, stream>>>(
          output_data, output_data, total_size);

      } else if (conversion == kYCbCr_TO_RGB) {
         
        // First from YCbCr to RGB
        DALI_CHECK_NPP(
          nppiYCbCrToRGB_8u_C3R(
            input_data, nStepInput, output_data, nStepOutput, size)
        );
      } else {
        
        DALI_ENFORCE( false, "conversion not supported");
      }
    } 
  }

  nppSetStream(old_stream);
}

DALI_REGISTER_OPERATOR(ColorSpaceConversion, ColorSpaceConversion<GPUBackend>, GPU);

}  // namespace dali

