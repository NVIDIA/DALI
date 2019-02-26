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

#include <cmath>
#include <utility>
#include <map>
#include <algorithm>
#include <tuple>
#include "dali/util/ocv.h"
#include "dali/error_handling.h"

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
  switch (c) {
    case 4:
      return CV_8UC4;
    case 3:
      return CV_8UC3;
    case 1:
    default:
      return CV_8UC1;
  }
}

static cv::ColorConversionCodes GetOpenCvColorConversionCode(
  DALIImageType input_type, DALIImageType output_type) {
    using ColorConversionPair = std::pair<DALIImageType, DALIImageType>;
    using ColorConversionMap = std::map< ColorConversionPair, cv::ColorConversionCodes >;
    static const ColorConversionMap color_conversion_map = {
        { {DALI_RGB, DALI_BGR},  cv::COLOR_RGB2BGR },
        { {DALI_RGB, DALI_GRAY}, cv::COLOR_RGB2GRAY },
        { {DALI_RGB, DALI_RGBA}, cv::COLOR_RGB2RGBA },
        { {DALI_RGB, DALI_BGRA}, cv::COLOR_RGB2BGRA },

        { {DALI_BGR, DALI_RGB},  cv::COLOR_BGR2RGB },
        { {DALI_BGR, DALI_GRAY}, cv::COLOR_BGR2GRAY },
        { {DALI_BGR, DALI_BGRA}, cv::COLOR_BGR2BGRA },
        { {DALI_BGR, DALI_RGBA}, cv::COLOR_BGR2RGBA },

        { {DALI_GRAY, DALI_RGB}, cv::COLOR_GRAY2RGB },
        { {DALI_GRAY, DALI_BGR}, cv::COLOR_GRAY2BGR },
        { {DALI_GRAY, DALI_RGBA}, cv::COLOR_GRAY2RGBA },
        { {DALI_GRAY, DALI_BGRA}, cv::COLOR_GRAY2BGRA },

        { {DALI_RGBA, DALI_RGB}, cv::COLOR_RGBA2RGB },
        { {DALI_RGBA, DALI_BGR}, cv::COLOR_RGBA2BGR },
        { {DALI_RGBA, DALI_GRAY}, cv::COLOR_RGBA2GRAY },
        { {DALI_RGBA, DALI_BGRA}, cv::COLOR_RGBA2BGRA },

        { {DALI_BGRA, DALI_RGB}, cv::COLOR_BGRA2RGB },
        { {DALI_BGRA, DALI_BGR}, cv::COLOR_BGRA2BGR },
        { {DALI_BGRA, DALI_RGBA}, cv::COLOR_BGRA2RGBA },
        { {DALI_BGRA, DALI_GRAY}, cv::COLOR_BGRA2GRAY }
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

inline static uint8_t Y(uint8_t r, uint8_t g, uint8_t b) {
  return static_cast<uint8_t>(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);
}

inline static uint8_t Cb(uint8_t r, uint8_t g, uint8_t b) {
  return (r == g && g == b) ? 128 :
    static_cast<uint8_t>(-0.148f * r - 0.291f * g + 0.439f * b + 128.0f);
}

inline static uint8_t Cr(uint8_t r, uint8_t g, uint8_t b) {
  return (r == g && g == b) ? 128 :
    static_cast<uint8_t>(0.439f * r - 0.368f * g - 0.071f * b + 128.0f);
}

template <>
inline void custom_conversion_pixel<DALI_RGB, DALI_YCbCr>(const uint8_t* input, uint8_t* output) {
  const auto r = input[0];
  const auto g = input[1];
  const auto b = input[2];
  output[0] = Y(r, g, b);
  output[1] = Cb(r, g, b);
  output[2] = Cr(r, g, b);
}

template <>
inline void custom_conversion_pixel<DALI_BGR, DALI_YCbCr>(const uint8_t* input, uint8_t* output) {
  const auto b = input[0];
  const auto g = input[1];
  const auto r = input[2];
  output[0] = Y(r, g, b);
  output[1] = Cb(r, g, b);
  output[2] = Cr(r, g, b);
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

void prepend_alpha(const uint8_t* input, uint8_t* output) {
  output[0] = 255;
  output[1] = input[0];
  output[2] = input[1];
  output[3] = input[2];
}

inline void prepend_alpha_and_reverse(const uint8_t* input, uint8_t* output) {
  output[0] = 255;
  output[1] = input[2];
  output[2] = input[1];
  output[3] = input[0];
}

template <>
inline void custom_conversion_pixel<DALI_RGB, DALI_ARGB>(const uint8_t* input, uint8_t* output) {
  prepend_alpha(input, output);
}

template <>
inline void custom_conversion_pixel<DALI_BGR, DALI_ABGR>(const uint8_t* input, uint8_t* output) {
  prepend_alpha(input, output);
}

template <>
inline void custom_conversion_pixel<DALI_BGR, DALI_ARGB>(const uint8_t* input, uint8_t* output) {
  prepend_alpha_and_reverse(input, output);
}

template <>
inline void custom_conversion_pixel<DALI_RGB, DALI_ABGR>(const uint8_t* input, uint8_t* output) {
  prepend_alpha_and_reverse(input, output);
}

inline void gray_to_AXYX(const uint8_t* input, uint8_t* output) {
  output[0] = 255;
  output[1] = input[0];
  output[2] = input[0];
  output[3] = input[0];
}

template <>
inline void custom_conversion_pixel<DALI_GRAY, DALI_ARGB>(const uint8_t* input, uint8_t* output) {
  gray_to_AXYX(input, output);
}

template <>
inline void custom_conversion_pixel<DALI_GRAY, DALI_ABGR>(const uint8_t* input, uint8_t* output) {
  gray_to_AXYX(input, output);
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_BGRA>(const uint8_t* input, uint8_t* output) {
  const auto y   = input[0];
  const auto cb  = input[1];
  const auto cr  = input[2];
  const auto rgb = RGB(y, cb, cr);
  output[0] = std::get<2>(rgb);
  output[1] = std::get<1>(rgb);
  output[2] = std::get<0>(rgb);
  output[3] = 255;
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_ABGR>(const uint8_t* input, uint8_t* output) {
  const auto y   = input[0];
  const auto cb  = input[1];
  const auto cr  = input[2];
  const auto rgb = RGB(y, cb, cr);
  output[0] = 255;
  output[1] = std::get<2>(rgb);
  output[2] = std::get<1>(rgb);
  output[3] = std::get<0>(rgb);
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_RGBA>(const uint8_t* input, uint8_t* output) {
  const auto y   = input[0];
  const auto cb  = input[1];
  const auto cr  = input[2];
  const auto rgb = RGB(y, cb, cr);
  output[0] = std::get<0>(rgb);
  output[1] = std::get<1>(rgb);
  output[2] = std::get<2>(rgb);
  output[3] = 255;
}

template <>
inline void custom_conversion_pixel<DALI_YCbCr, DALI_ARGB>(const uint8_t* input, uint8_t* output) {
  const auto y   = input[0];
  const auto cb  = input[1];
  const auto cr  = input[2];
  const auto rgb = RGB(y, cb, cr);
  output[0] = 255;
  output[1] = std::get<0>(rgb);
  output[2] = std::get<1>(rgb);
  output[3] = std::get<2>(rgb);
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
  DALI_ENFORCE(input_img.elemSize()  == NumberOfChannels(input_type),
    "Incorrect number of channels");
  DALI_ENFORCE(output_img.elemSize() == NumberOfChannels(output_type),
    "Incorrect number of channels");
  auto ocv_conversion_code = GetOpenCvColorConversionCode(input_type, output_type);
  bool ocv_supported = (ocv_conversion_code != cv::COLOR_COLORCVT_MAX);
  if ( ocv_supported ) {
    cv::cvtColor(input_img, output_img, ocv_conversion_code);
    return;
  }
  using ColorConversionPair = std::pair<DALIImageType, DALIImageType>;
  ColorConversionPair conversion { input_type, output_type };

  using ConversionFunction = std::function<void (const cv::Mat& img, cv::Mat& output_img)>;
  std::map<ColorConversionPair, ConversionFunction> custom_conversions;

  if (custom_conversions.empty()) {
    using namespace std::placeholders;
    using std::bind;
    custom_conversions[{DALI_RGB, DALI_YCbCr}] =
      bind(custom_conversion<DALI_RGB, DALI_YCbCr>, _1, _2);
    custom_conversions[{DALI_BGR, DALI_YCbCr}] =
      bind(custom_conversion<DALI_BGR, DALI_YCbCr>, _1, _2);
    custom_conversions[{DALI_YCbCr, DALI_RGB}] =
      bind(custom_conversion<DALI_YCbCr, DALI_RGB>, _1, _2);
    custom_conversions[{DALI_YCbCr, DALI_BGR}] =
      bind(custom_conversion<DALI_YCbCr, DALI_BGR>, _1, _2);
    custom_conversions[{DALI_GRAY, DALI_YCbCr}] =
      bind(custom_conversion<DALI_GRAY, DALI_YCbCr>, _1, _2);
    custom_conversions[{DALI_YCbCr, DALI_GRAY}] =
      bind(custom_conversion<DALI_YCbCr, DALI_GRAY>, _1, _2);
    custom_conversions[{DALI_RGB, DALI_ARGB}] =
      bind(custom_conversion<DALI_RGB, DALI_ARGB>, _1, _2);
    custom_conversions[{DALI_BGR, DALI_ARGB}] =
      bind(custom_conversion<DALI_BGR, DALI_ARGB>, _1, _2);
    custom_conversions[{DALI_GRAY, DALI_ARGB}] =
      bind(custom_conversion<DALI_GRAY, DALI_ARGB>, _1, _2);
    custom_conversions[{DALI_RGB, DALI_ABGR}] =
      bind(custom_conversion<DALI_RGB, DALI_ABGR>, _1, _2);
    custom_conversions[{DALI_BGR, DALI_ABGR}] =
      bind(custom_conversion<DALI_BGR, DALI_ABGR>, _1, _2);
    custom_conversions[{DALI_GRAY, DALI_ABGR}] =
      bind(custom_conversion<DALI_GRAY, DALI_ABGR>, _1, _2);
    custom_conversions[{DALI_YCbCr, DALI_RGBA}] =
      bind(custom_conversion<DALI_YCbCr, DALI_RGBA>, _1, _2);
    custom_conversions[{DALI_YCbCr, DALI_BGRA}] =
      bind(custom_conversion<DALI_YCbCr, DALI_BGRA>, _1, _2);
    custom_conversions[{DALI_YCbCr, DALI_ARGB}] =
      bind(custom_conversion<DALI_YCbCr, DALI_ARGB>, _1, _2);
    custom_conversions[{DALI_YCbCr, DALI_ABGR}] =
      bind(custom_conversion<DALI_YCbCr, DALI_ABGR>, _1, _2);
  }

  auto it = custom_conversions.find(conversion);
  if (it == custom_conversions.end()) {
    DALI_FAIL("conversion from "
      + std::to_string(input_type) + " to "
      + std::to_string(output_type) + " not supported");
  }
  auto &functor = it->second;
  functor(input_img, output_img);
}

}  // namespace dali
