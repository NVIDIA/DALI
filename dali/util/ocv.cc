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
#include "dali/util/color_space_conversion_utils.h"

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
  const auto r = input[0];
  const auto g = input[1];
  const auto b = input[2];
  output[0] = Y<uint8_t>(r, g, b);
  output[1] = Cb<uint8_t>(r, g, b);
  output[2] = Cr<uint8_t>(r, g, b);
}

template <>
inline void custom_conversion_pixel<DALI_BGR, DALI_YCbCr>(const uint8_t* input, uint8_t* output) {
  const auto b = input[0];
  const auto g = input[1];
  const auto r = input[2];
  output[0] = Y<uint8_t>(r, g, b);
  output[1] = Cb<uint8_t>(r, g, b);
  output[2] = Cr<uint8_t>(r, g, b);
}

inline static uint8_t clip(float x, float max = 255.0f) {
  return static_cast<uint8_t>( std::min(std::max(x, 0.0f), max) );
}

inline static std::tuple<uint8_t, uint8_t, uint8_t> RGB(uint8_t y, uint8_t cb, uint8_t cr) {
  const float nY = 1.164f * (static_cast<float>(y) - 16.0f);
  float nR = (static_cast<float>(cr) - 128.0f);
  float nB = (static_cast<float>(cb) - 128.0f);
  float nG = nY - 0.813f * nR - 0.392f * nB;
  nG = std::min(nG, 255.0f);
  nR = nY + 1.596f * nR;
  nR = std::min(nR, 255.0f);
  nB = nY + 2.017f * nB;
  nB = std::min(nB, 255.0f);
  const uint8_t r = clip(nR);
  const uint8_t g = clip(nG);
  const uint8_t b = clip(nB);
  return std::tuple<uint8_t, uint8_t, uint8_t>{ r, g, b };
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_RGB>(const uint8_t* input, uint8_t* output) {
  const auto y   = input[0];
  const auto cb  = input[1];
  const auto cr  = input[2];
  const auto rgb = RGB(y, cb, cr);
  output[0] = std::get<0>(rgb);
  output[1] = std::get<1>(rgb);
  output[2] = std::get<2>(rgb);
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_BGR>(const uint8_t* input, uint8_t* output) {
  const auto y   = input[0];
  const auto cb  = input[1];
  const auto cr  = input[2];
  const auto rgb = RGB(y, cb, cr);
  output[0] = std::get<2>(rgb);
  output[1] = std::get<1>(rgb);
  output[2] = std::get<0>(rgb);
}

template <>
inline void custom_conversion_pixel<DALI_GRAY, DALI_YCbCr>(const uint8_t* input, uint8_t* output) {
  const auto y = input[0];
  output[0] = y;
  output[1] = 128;
  output[2] = 128;
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_GRAY>(const uint8_t* input, uint8_t* output) {
  const auto y = input[0];
  output[0] = y;
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
