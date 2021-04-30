// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/util/ocv.h"
#include <cmath>
#include <utility>
#include <map>
#include <algorithm>
#include <tuple>
#include "dali/core/error_handling.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"

namespace dali {

int OCVInterpForDALIInterp(DALIInterpType type, int *ocv_type) {
  switch (type) {
  case DALI_INTERP_NN:
    *ocv_type =  cv::INTER_NEAREST;
    break;
  case DALI_INTERP_LINEAR:
    *ocv_type =  cv::INTER_LINEAR;
    break;
  case DALI_INTERP_CUBIC:
    *ocv_type =  cv::INTER_CUBIC;
    break;
  default:
    return DALIError;
  }
  return DALISuccess;
}

int GetOpenCvChannelType(size_t c) {
  return ( c == 3 ) ? CV_8UC3 : CV_8UC1;
}

static cv::ColorConversionCodes GetOpenCvColorConversionCode(
  DALIImageType input_type, DALIImageType output_type) {
    using ColorConversionPair = std::pair<DALIImageType, DALIImageType>;
    using ColorConversionMap = std::map< ColorConversionPair, cv::ColorConversionCodes >;
    static const ColorConversionMap color_conversion_map = {
        { {DALI_RGB, DALI_BGR},  cv::COLOR_RGB2BGR },
        { {DALI_RGB, DALI_GRAY}, cv::COLOR_RGB2GRAY },

        { {DALI_BGR, DALI_RGB},  cv::COLOR_BGR2RGB },
        { {DALI_BGR, DALI_GRAY}, cv::COLOR_BGR2GRAY },

        { {DALI_GRAY, DALI_RGB}, cv::COLOR_GRAY2RGB },
        { {DALI_GRAY, DALI_BGR}, cv::COLOR_GRAY2BGR },
    };

    const ColorConversionPair color_conversion_pair{ input_type, output_type };
    const auto it = color_conversion_map.find(color_conversion_pair);
    if (it == color_conversion_map.end()) {
      return cv::COLOR_COLORCVT_MAX;
    }
    return it->second;
}

template <DALIImageType input_type, DALIImageType output_type>
void custom_conversion_pixel(const uint8_t* input, uint8_t* output);

template <>
inline void custom_conversion_pixel<DALI_RGB, DALI_YCbCr>(const uint8_t* input, uint8_t* output) {
  vec<3, uint8_t> rgb = {input[0], input[1], input[2]};
  output[0] = kernels::color::itu_r_bt_601::rgb_to_y<uint8_t>(rgb);
  output[1] = kernels::color::itu_r_bt_601::rgb_to_cb<uint8_t>(rgb);
  output[2] = kernels::color::itu_r_bt_601::rgb_to_cr<uint8_t>(rgb);
}

template <>
inline void custom_conversion_pixel<DALI_BGR, DALI_YCbCr>(const uint8_t* input, uint8_t* output) {
  vec<3, uint8_t> rgb = {input[2], input[1], input[0]};
  output[0] = kernels::color::itu_r_bt_601::rgb_to_y<uint8_t>(rgb);
  output[1] = kernels::color::itu_r_bt_601::rgb_to_cb<uint8_t>(rgb);
  output[2] = kernels::color::itu_r_bt_601::rgb_to_cr<uint8_t>(rgb);
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_RGB>(const uint8_t* input, uint8_t* output) {
  vec<3, uint8_t> ycbcr{input[0], input[1], input[2]};
  auto rgb = kernels::color::itu_r_bt_601::ycbcr_to_rgb<uint8_t>(ycbcr);
  output[0] = rgb[0];
  output[1] = rgb[1];
  output[2] = rgb[2];
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_BGR>(const uint8_t* input, uint8_t* output) {
  vec<3, uint8_t> ycbcr{input[0], input[1], input[2]};
  auto rgb = kernels::color::itu_r_bt_601::ycbcr_to_rgb<uint8_t>(ycbcr);
  output[0] = rgb[2];
  output[1] = rgb[1];
  output[2] = rgb[0];
}

template <>
inline void custom_conversion_pixel<DALI_GRAY, DALI_YCbCr>(const uint8_t* input, uint8_t* output) {
  output[0] = kernels::color::itu_r_bt_601::gray_to_y<uint8_t>(input[0]);
  output[1] = 128;
  output[2] = 128;
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_GRAY>(const uint8_t* input, uint8_t* output) {
  output[0] = kernels::color::itu_r_bt_601::y_to_gray<uint8_t>(input[0]);
}

template <DALIImageType input_type, DALIImageType output_type>
inline void custom_conversion(const cv::Mat& img, cv::Mat& output_img) {
  const std::size_t input_C = img.elemSize();
  const std::size_t output_C = output_img.elemSize();
  const std::size_t total_size = img.rows * img.cols;
  for (std::size_t i = 0; i < total_size; i ++) {
    custom_conversion_pixel<input_type, output_type>(
      img.data + i*input_C,
      output_img.data + i*output_C);
  }
}

void OpenCvColorConversion(DALIImageType input_type, const cv::Mat& input_img,
                           DALIImageType output_type, cv::Mat& output_img) {
  DALI_ENFORCE(input_img.elemSize() == static_cast<size_t>(NumberOfChannels(input_type)),
    "Incorrect number of channels");
  DALI_ENFORCE(output_img.elemSize() == static_cast<size_t>(NumberOfChannels(output_type)),
    "Incorrect number of channels");
  auto ocv_conversion_code = GetOpenCvColorConversionCode(input_type, output_type);
  bool ocv_supported = (ocv_conversion_code != cv::COLOR_COLORCVT_MAX);
  if ( ocv_supported ) {
    cv::cvtColor(input_img, output_img, ocv_conversion_code);
    return;
  }
  using ColorConversionPair = std::pair<DALIImageType, DALIImageType>;
  ColorConversionPair conversion { input_type, output_type };
  // Handle special cases
  const ColorConversionPair kRGBToYCbCr  { DALI_RGB,   DALI_YCbCr };
  const ColorConversionPair kBGRToYCbCr  { DALI_BGR,   DALI_YCbCr };
  const ColorConversionPair kYCbCrToRGB  { DALI_YCbCr, DALI_RGB };
  const ColorConversionPair kYCbCrToBGR  { DALI_YCbCr, DALI_BGR };
  const ColorConversionPair kGrayToYCbCr { DALI_GRAY,  DALI_YCbCr };
  const ColorConversionPair kYCbCrToGray { DALI_YCbCr, DALI_GRAY };

  if ( conversion == kRGBToYCbCr ) {
    custom_conversion<DALI_RGB, DALI_YCbCr>(input_img, output_img);
    return;
  } else if ( conversion == kBGRToYCbCr ) {
    custom_conversion<DALI_BGR, DALI_YCbCr>(input_img, output_img);
    return;
  } else if ( conversion == kYCbCrToRGB ) {
    custom_conversion<DALI_YCbCr, DALI_RGB>(input_img, output_img);
    return;
  } else if ( conversion == kYCbCrToBGR ) {
    custom_conversion<DALI_YCbCr, DALI_BGR>(input_img, output_img);
    return;
  } else if ( conversion == kGrayToYCbCr ) {
    custom_conversion<DALI_GRAY, DALI_YCbCr>(input_img, output_img);
    return;
  } else if ( conversion == kYCbCrToGray ) {
    custom_conversion<DALI_YCbCr, DALI_GRAY>(input_img, output_img);
    return;
  }

  DALI_FAIL("conversion not supported");
}

}  // namespace dali
