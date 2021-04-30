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

#ifndef DALI_UTIL_NPP_H_
#define DALI_UTIL_NPP_H_

#include <npp.h>

#include "dali/core/error_handling.h"
#include "dali/core/common.h"

namespace dali {

#if NPP_VERSION_MAJOR < 8
#error "Only Support Cuda 8 or Higher"
#elif NPP_VERSION_MAJOR == 8
// We only consider the case that
//   oDstRectROI.width  == oDstSize.width
//   oDstRectROI.height == oDstSize.height
//   oDstRectROI.x == 0
//   oDstRectROI.y == 0
// because it is the only case used in DALI for now
// neep help to complete the arguments conversion
inline static
NppStatus
nppiResize_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                        Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                  int eInterpolation) {
  if (
    oDstRectROI.width  == oDstSize.width &&
    oDstRectROI.height == oDstSize.height &&
    oDstRectROI.x == 0 &&
    oDstRectROI.y == 0
  )
    return nppiResize_8u_C1R(pSrc, oSrcSize, nSrcStep, oSrcRectROI,
                             pDst, nDstStep, oDstSize,
                             static_cast<double>(oDstRectROI.width)  / oSrcRectROI.width,
                             static_cast<double>(oDstRectROI.height) / oSrcRectROI.height,
                             eInterpolation);
  else
    return NPP_NOT_SUPPORTED_MODE_ERROR;
}

inline static
NppStatus
nppiResize_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                        Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                  int eInterpolation) {
  if (
    oDstRectROI.width  == oDstSize.width &&
    oDstRectROI.height == oDstSize.height &&
    oDstRectROI.x == 0 &&
    oDstRectROI.y == 0
  )
    return nppiResize_8u_C1R(pSrc, oSrcSize, nSrcStep, oSrcRectROI,
                             pDst, nDstStep, oDstSize,
                             static_cast<double>(oDstRectROI.width)  / oSrcRectROI.width,
                             static_cast<double>(oDstRectROI.height) / oSrcRectROI.height,
                             eInterpolation);
  else
    return NPP_NOT_SUPPORTED_MODE_ERROR;
}
#endif  // NPP_VERSION_MAJOR == 8

int NPPInterpForDALIInterp(DALIInterpType type, NppiInterpolationMode *npp_type);

static const char *nppErrorString(NppStatus error) {
  switch (error) {
  case NPP_NOT_SUPPORTED_MODE_ERROR:
    return "NPP_NOT_SUPPORTED_MODE_ERROR";

  case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
    return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

  case NPP_RESIZE_NO_OPERATION_ERROR:
    return "NPP_RESIZE_NO_OPERATION_ERROR";

  case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
    return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

  case NPP_BAD_ARG_ERROR:
    return "NPP_BAD_ARGUMENT_ERROR";

  case NPP_COEFF_ERROR:
    return "NPP_COEFFICIENT_ERROR";

  case NPP_RECT_ERROR:
    return "NPP_RECTANGLE_ERROR";

  case NPP_QUAD_ERROR:
    return "NPP_QUADRANGLE_ERROR";

  case NPP_MEM_ALLOC_ERR:
    return "NPP_MEMORY_ALLOCATION_ERROR";

  case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
    return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

  case NPP_INVALID_INPUT:
    return "NPP_INVALID_INPUT";

  case NPP_POINTER_ERROR:
    return "NPP_POINTER_ERROR";

  case NPP_WARNING:
    return "NPP_WARNING";

  case NPP_ODD_ROI_WARNING:
    return "NPP_ODD_ROI_WARNING";
#else

    // These are for CUDA 5.5 or higher
  case NPP_BAD_ARGUMENT_ERROR:
    return "NPP_BAD_ARGUMENT_ERROR";

  case NPP_COEFFICIENT_ERROR:
    return "NPP_COEFFICIENT_ERROR";

  case NPP_RECTANGLE_ERROR:
    return "NPP_RECTANGLE_ERROR";

  case NPP_QUADRANGLE_ERROR:
    return "NPP_QUADRANGLE_ERROR";

  case NPP_MEMORY_ALLOCATION_ERR:
    return "NPP_MEMORY_ALLOCATION_ERROR";

  case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
    return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

  case NPP_INVALID_HOST_POINTER_ERROR:
    return "NPP_INVALID_HOST_POINTER_ERROR";

  case NPP_INVALID_DEVICE_POINTER_ERROR:
    return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

  case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
    return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

  case NPP_TEXTURE_BIND_ERROR:
    return "NPP_TEXTURE_BIND_ERROR";

  case NPP_WRONG_INTERSECTION_ROI_ERROR:
    return "NPP_WRONG_INTERSECTION_ROI_ERROR";

  case NPP_NOT_EVEN_STEP_ERROR:
    return "NPP_NOT_EVEN_STEP_ERROR";

  case NPP_INTERPOLATION_ERROR:
    return "NPP_INTERPOLATION_ERROR";

  case NPP_RESIZE_FACTOR_ERROR:
    return "NPP_RESIZE_FACTOR_ERROR";

  case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
    return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";


#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

  case NPP_MEMFREE_ERR:
    return "NPP_MEMFREE_ERR";

  case NPP_MEMSET_ERR:
    return "NPP_MEMSET_ERR";

  case NPP_MEMCPY_ERR:
    return "NPP_MEMCPY_ERROR";

  case NPP_MIRROR_FLIP_ERR:
    return "NPP_MIRROR_FLIP_ERR";
#else

  case NPP_MEMFREE_ERROR:
    return "NPP_MEMFREE_ERROR";

  case NPP_MEMSET_ERROR:
    return "NPP_MEMSET_ERROR";

  case NPP_MEMCPY_ERROR:
    return "NPP_MEMCPY_ERROR";

  case NPP_MIRROR_FLIP_ERROR:
    return "NPP_MIRROR_FLIP_ERROR";
#endif

  case NPP_ALIGNMENT_ERROR:
    return "NPP_ALIGNMENT_ERROR";

  case NPP_STEP_ERROR:
    return "NPP_STEP_ERROR";

  case NPP_SIZE_ERROR:
    return "NPP_SIZE_ERROR";

  case NPP_NULL_POINTER_ERROR:
    return "NPP_NULL_POINTER_ERROR";

  case NPP_CUDA_KERNEL_EXECUTION_ERROR:
    return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

  case NPP_NOT_IMPLEMENTED_ERROR:
    return "NPP_NOT_IMPLEMENTED_ERROR";

  case NPP_ERROR:
    return "NPP_ERROR";

  case NPP_SUCCESS:
    return "NPP_SUCCESS";

  case NPP_WRONG_INTERSECTION_QUAD_WARNING:
    return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

  case NPP_MISALIGNED_DST_ROI_WARNING:
    return "NPP_MISALIGNED_DST_ROI_WARNING";

  case NPP_AFFINE_QUAD_INCORRECT_WARNING:
    return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

  case NPP_DOUBLE_SIZE_WARNING:
    return "NPP_DOUBLE_SIZE_WARNING";

  case NPP_WRONG_INTERSECTION_ROI_WARNING:
    return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
    /* These are 6.0 or higher */
  case NPP_LUT_PALETTE_BITSIZE_ERROR:
    return "NPP_LUT_PALETTE_BITSIZE_ERROR";

  case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
    return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

  case NPP_QUALITY_INDEX_ERROR:
    return "NPP_QUALITY_INDEX_ERROR";

  case NPP_CHANNEL_ORDER_ERROR:
    return "NPP_CHANNEL_ORDER_ERROR";

  case NPP_ZERO_MASK_VALUE_ERROR:
    return "NPP_ZERO_MASK_VALUE_ERROR";

  case NPP_NUMBER_OF_CHANNELS_ERROR:
    return "NPP_NUMBER_OF_CHANNELS_ERROR";

  case NPP_COI_ERROR:
    return "NPP_COI_ERROR";

  case NPP_DIVISOR_ERROR:
    return "NPP_DIVISOR_ERROR";

  case NPP_CHANNEL_ERROR:
    return "NPP_CHANNEL_ERROR";

  case NPP_STRIDE_ERROR:
    return "NPP_STRIDE_ERROR";

  case NPP_ANCHOR_ERROR:
    return "NPP_ANCHOR_ERROR";

  case NPP_MASK_SIZE_ERROR:
    return "NPP_MASK_SIZE_ERROR";

  case NPP_MOMENT_00_ZERO_ERROR:
    return "NPP_MOMENT_00_ZERO_ERROR";

  case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
    return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

  case NPP_THRESHOLD_ERROR:
    return "NPP_THRESHOLD_ERROR";

  case NPP_CONTEXT_MATCH_ERROR:
    return "NPP_CONTEXT_MATCH_ERROR";

  case NPP_FFT_FLAG_ERROR:
    return "NPP_FFT_FLAG_ERROR";

  case NPP_FFT_ORDER_ERROR:
    return "NPP_FFT_ORDER_ERROR";

  case NPP_SCALE_RANGE_ERROR:
    return "NPP_SCALE_RANGE_ERROR";

  case NPP_DATA_TYPE_ERROR:
    return "NPP_DATA_TYPE_ERROR";

  case NPP_OUT_OFF_RANGE_ERROR:
    return "NPP_OUT_OFF_RANGE_ERROR";

  case NPP_DIVIDE_BY_ZERO_ERROR:
    return "NPP_DIVIDE_BY_ZERO_ERROR";

  case NPP_RANGE_ERROR:
    return "NPP_RANGE_ERROR";

  case NPP_NO_MEMORY_ERROR:
    return "NPP_NO_MEMORY_ERROR";

  case NPP_ERROR_RESERVED:
    return "NPP_ERROR_RESERVED";

  case NPP_NO_OPERATION_WARNING:
    return "NPP_NO_OPERATION_WARNING";

  case NPP_DIVIDE_BY_ZERO_WARNING:
    return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
    /* These are 7.0 or higher */
  case NPP_OVERFLOW_ERROR:
    return "NPP_OVERFLOW_ERROR";

  case NPP_CORRUPTED_DATA_ERROR:
    return "NPP_CORRUPTED_DATA_ERROR";
#endif
  }

  return "<unknown>";
}

template <typename T>
NppiSize ToNppiSize(const T& size) {
  NppiSize out;
  out.width = size.width;
  out.height = size.height;
  return out;
}

}  // namespace dali

// For checking npp return errors in dali library functions
#define DALI_CHECK_NPP(code)                              \
  do {                                                    \
    NppStatus status = code;                              \
    if (status != NPP_SUCCESS) {                          \
    dali::string file = __FILE__;                         \
      dali::string line = std::to_string(__LINE__);       \
      dali::string error = "[" + file + ":" + line +      \
        "]: NPP error \"" +                               \
        dali::nppErrorString(status) + "\"";              \
      DALI_FAIL(error);                                   \
    }                                                     \
  } while (0)

#endif  // DALI_UTIL_NPP_H_
