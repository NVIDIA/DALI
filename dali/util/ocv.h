// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_OCV_H_
#define DALI_UTIL_OCV_H_

#include <opencv2/opencv.hpp>
#include "dali/core/boundary.h"
#include "dali/core/common.h"
#include "dali/core/dali_data_type.h"
#include "dali/core/error_handling.h"

namespace dali {

cv::InterpolationFlags inline OCVInterpForDALIInterp(DALIInterpType type) {
  switch (type) {
    case DALIInterpType::DALI_INTERP_NN:
      return cv::InterpolationFlags::INTER_NEAREST;
    case DALIInterpType::DALI_INTERP_LINEAR:
      return cv::InterpolationFlags::INTER_LINEAR;
    case DALIInterpType::DALI_INTERP_CUBIC:
      return cv::InterpolationFlags::INTER_CUBIC;
    default:
      DALI_FAIL("OpenCV does not support DALI InterpolationType `", to_string(type), "`.");
  }
}

int inline OCVBorderForDALIBoundary(boundary::BoundaryType border_type) {
  using namespace boundary;  // NOLINT(build/namespaces)
  switch (border_type) {
    case BoundaryType::CONSTANT:
      return cv::BORDER_CONSTANT;
    case BoundaryType::CLAMP:
      return cv::BORDER_REPLICATE;
    case BoundaryType::REFLECT_1001:
      return cv::BORDER_REFLECT;
    case BoundaryType::REFLECT_101:
      return cv::BORDER_REFLECT_101;
    case BoundaryType::WRAP:
      return cv::BORDER_WRAP;
    case BoundaryType::TRANSPARENT:
      return cv::BORDER_TRANSPARENT;
    case BoundaryType::ISOLATED:
      return cv::BORDER_ISOLATED;
    default:
      DALI_FAIL("OpenCV does not support DALI BoundaryType`", to_string(border_type), "`.");
  }
}

/**
 * @brief Convert runtime DALIDataType to OpenCV data type
 *
 * @note will throw if the type is not supported
 */
int inline OCVMatTypeForDALIData(daliDataType_t dtype) {
  switch (dtype) {
    case DALI_UINT8:
      return CV_8U;
    case DALI_INT16:
      return CV_16S;
    case DALI_UINT16:
      return CV_16U;
    case DALI_INT32:
      return CV_32S;
    case DALI_FLOAT16:
      return CV_16F;
    case DALI_FLOAT:
      return CV_32F;
    case DALI_FLOAT64:
      return CV_64F;
    default:
      DALI_FAIL("OpenCV does not support DALI DataType: `", daliDataTypeName(dtype), "`.");
  }
}

/**
 * @brief Convert runtime DALIDataType to OpenCV data type including number of channels
 *
 * @note will throw if the type is not supported
 */
int inline OCVMatTypeForDALIData(daliDataType_t dtype, int channels) {
  DALI_ENFORCE(1 <= channels && channels <= 4,
               "OpenCV supports only 1-4 channels, got " + std::to_string(channels) + " channels.");
  return CV_MAKETYPE(OCVMatTypeForDALIData(dtype), channels);
}

template <typename T>
inline cv::Mat DLL_PUBLIC CreateMatFromPtr(int H, int W, int type, const T* ptr,
                                           size_t step = cv::Mat::AUTO_STEP) {
  // Note: OpenCV can't take a const pointer to wrap even when the cv::Mat is const. This
  // is kinda icky to const_cast away the const-ness, but there isn't another way
  // (that I know of) without making the input argument non-const.
  return cv::Mat(H, W, type, const_cast<T*>(ptr), step);
}

template <typename T>
struct type2ocv {};

template <>
struct type2ocv<uint8_t> {
  static int value(int c) {
    return CV_8UC(c);
  }
};

template <>
struct type2ocv<int8_t> {
  static int value(int c) {
    return CV_8SC(c);
  }
};

template <>
struct type2ocv<uint16_t> {
  static int value(int c) {
    return CV_16UC(c);
  }
};

template <>
struct type2ocv<int16_t> {
  static int value(int c) {
    return CV_16SC(c);
  }
};

template <>
struct type2ocv<int32_t> {
  static int value(int c) {
    return CV_32SC(c);
  }
};

template <>
struct type2ocv<float> {
  static int value(int c) {
    return CV_32FC(c);
  }
};

int DLL_PUBLIC GetOpenCvChannelType(size_t c);

void DLL_PUBLIC OpenCvColorConversion(DALIImageType input_type, const cv::Mat& input_img,
                                      DALIImageType output_type, cv::Mat& output_img);

}  // namespace dali

// OpenCV 2.0 Compatibility
#if CV_MAJOR_VERSION == 2
namespace cv {
using InterpolationFlags = int;
using ColorConversionCodes = int;
}  // namespace cv

#endif

#endif  // DALI_UTIL_OCV_H_
