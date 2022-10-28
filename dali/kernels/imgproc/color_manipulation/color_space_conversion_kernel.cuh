// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_KERNEL_CUH_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_KERNEL_CUH_

#include <cuda_runtime_api.h>
#include <utility>
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"
#include "dali/core/cuda_error.h"

namespace dali {
namespace kernels {
namespace color {

template <typename Out, typename In>
struct RGB_to_BGR_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> rgb) {
    vec<out_pixel_sz, Out> out;
    out[0] = ConvertSatNorm<Out>(rgb[2]);
    out[1] = ConvertSatNorm<Out>(rgb[1]);
    out[2] = ConvertSatNorm<Out>(rgb[0]);
    return out;
  }
};

template <typename Out, typename In>
struct RGB_to_Gray_Converter {
  static constexpr int out_pixel_sz = 1;
  static constexpr int in_pixel_sz = 3;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> rgb) {
    return rgb_to_gray<Out>(rgb);
  }
};

template <typename Out, typename In>
struct BGR_to_Gray_Converter {
  static constexpr int out_pixel_sz = 1;
  static constexpr int in_pixel_sz = 3;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> bgr) {
    return RGB_to_Gray_Converter<Out, In>::convert(
      RGB_to_BGR_Converter<In, In>::convert(bgr));
  }
};

template <typename Out, typename In>
struct Gray_to_RGB_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 1;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> gray) {
    return vec<out_pixel_sz, Out>(gray[0]);
  }
};

template <typename Out, typename In>
struct Gray_to_YCbCr_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 1;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> gray) {
    vec<out_pixel_sz, Out> out;
    out[0] = itu_r_bt_601::gray_to_y<Out>(gray[0]);
    out[1] = ConvertNorm<Out>(0.5f);
    out[2] = ConvertNorm<Out>(0.5f);
    return out;
  }
};

template <typename Out, typename In>
struct YCbCr_to_Gray_Converter {
  static constexpr int out_pixel_sz = 1;
  static constexpr int in_pixel_sz = 3;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> ycbcr) {
    vec<out_pixel_sz, Out> out;
    out[0] = itu_r_bt_601::y_to_gray<Out>(ycbcr[0]);
    return out;
  }
};

template <typename Out, typename In>
struct RGB_to_YCbCr_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> rgb) {
    vec<out_pixel_sz, Out> out;
    out[0] = itu_r_bt_601::rgb_to_y<Out>(rgb);
    out[1] = itu_r_bt_601::rgb_to_cb<Out>(rgb);
    out[2] = itu_r_bt_601::rgb_to_cr<Out>(rgb);
    return out;
  }
};

template <typename Out, typename In>
struct BGR_to_YCbCr_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> bgr) {
    return RGB_to_YCbCr_Converter<Out, In>::convert(
      RGB_to_BGR_Converter<In, In>::convert(bgr));
  }
};

template <typename Out, typename In>
struct YCbCr_to_RGB_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> ycbcr) {
    return itu_r_bt_601::ycbcr_to_rgb<Out>(ycbcr);
  }
};

template <typename Out, typename In>
struct YCbCr_to_BGR_Converter {
  static constexpr int out_pixel_sz = 3;
  static constexpr int in_pixel_sz = 3;
  static DALI_HOST_DEV DALI_FORCEINLINE vec<out_pixel_sz, Out> convert(vec<in_pixel_sz, In> ycbcr) {
    return RGB_to_BGR_Converter<Out, Out>::convert(
      YCbCr_to_RGB_Converter<Out, In>::convert(ycbcr));
  }
};

template <typename Converter, typename Out, typename In>
__global__ void ColorSpaceConvKernel(Out *output, const In *input, int64_t sz) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= sz) {
    return;
  }
  vec<Converter::in_pixel_sz, In> pixel;
#pragma unroll
  for (int c = 0; c < Converter::in_pixel_sz; c++) {
    pixel[c] = input[idx * Converter::in_pixel_sz + c];
  }
  auto out_pixel = Converter::convert(pixel);
#pragma unroll
  for (int c = 0; c < Converter::out_pixel_sz; c++) {
    output[idx * Converter::out_pixel_sz + c] = out_pixel[c];
  }
}

// TODO(janton): Write a generic batched color space conversion kernel template
template <typename Out, typename In>
void RunColorSpaceConversionKernel(Out *output, const In *input, DALIImageType out_type,
                                   DALIImageType in_type, int64_t npixels, cudaStream_t stream) {
  // For CUDA kernel
  const unsigned int block = npixels < 1024 ? npixels : 1024;
  const unsigned int grid = (npixels + block - 1) / block;

  using ImageTypePair = std::pair<DALIImageType, DALIImageType>;
  ImageTypePair conversion{in_type, out_type};
  const ImageTypePair kRGB_TO_BGR{DALI_RGB, DALI_BGR};
  const ImageTypePair kBGR_TO_RGB{DALI_BGR, DALI_RGB};
  const ImageTypePair kRGB_TO_YCbCr{DALI_RGB, DALI_YCbCr};
  const ImageTypePair kBGR_TO_YCbCr{DALI_BGR, DALI_YCbCr};
  const ImageTypePair kRGB_TO_GRAY{DALI_RGB, DALI_GRAY};
  const ImageTypePair kBGR_TO_GRAY{DALI_BGR, DALI_GRAY};
  const ImageTypePair kYCbCr_TO_BGR{DALI_YCbCr, DALI_BGR};
  const ImageTypePair kYCbCr_TO_RGB{DALI_YCbCr, DALI_RGB};
  const ImageTypePair kYCbCr_TO_GRAY{DALI_YCbCr, DALI_GRAY};
  const ImageTypePair kGRAY_TO_RGB{DALI_GRAY, DALI_RGB};
  const ImageTypePair kGRAY_TO_BGR{DALI_GRAY, DALI_BGR};
  const ImageTypePair kGRAY_TO_YCbCr{DALI_GRAY, DALI_YCbCr};

  if (conversion == kRGB_TO_BGR || conversion == kBGR_TO_RGB) {
    ColorSpaceConvKernel<RGB_to_BGR_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kRGB_TO_YCbCr) {
    ColorSpaceConvKernel<RGB_to_YCbCr_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kBGR_TO_YCbCr) {
    ColorSpaceConvKernel<BGR_to_YCbCr_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kRGB_TO_GRAY) {
    ColorSpaceConvKernel<RGB_to_Gray_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kBGR_TO_GRAY) {
    ColorSpaceConvKernel<BGR_to_Gray_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kYCbCr_TO_BGR) {
    ColorSpaceConvKernel<YCbCr_to_BGR_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kYCbCr_TO_RGB) {
    ColorSpaceConvKernel<YCbCr_to_RGB_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kGRAY_TO_BGR || conversion == kGRAY_TO_RGB) {
    ColorSpaceConvKernel<Gray_to_RGB_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kGRAY_TO_YCbCr) {
    ColorSpaceConvKernel<Gray_to_YCbCr_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else if (conversion == kYCbCr_TO_GRAY) {
    ColorSpaceConvKernel<YCbCr_to_Gray_Converter<Out, In>, Out, In>
        <<<grid, block, 0, stream>>>(output, input, npixels);
  } else {
    DALI_FAIL(make_string("conversion not supported ", in_type, " to ", out_type));
  }
  CUDA_CALL(cudaGetLastError());
}

}  // namespace color
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_KERNEL_CUH_
