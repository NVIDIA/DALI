// Copyright (c) 2017-2018, 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_UTIL_NPP_H_
#define DALI_OPERATORS_UTIL_NPP_H_

#include <npp.h>
#include <string>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"
#include "dali/core/format.h"

#if WITH_DYNAMIC_CUDA_TOOLKIT_ENABLED
  bool nppIsSymbolAvailable(const char *name);
#else
  #define nppIsSymbolAvailable(T) (true)
#endif

namespace dali {

class NppError : public std::runtime_error {
 public:
  explicit NppError(NppStatus result, const char *details = nullptr)
  : std::runtime_error(Message(result, details))
  , result_(result) {}

  static const char *ErrorString(NppStatus result) {
    switch (result) {
      case NPP_NOT_SUPPORTED_MODE_ERROR:
        return "NPP_NOT_SUPPORTED_MODE_ERROR";

      case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
        return "Unsupported round mode";

      case NPP_RESIZE_NO_OPERATION_ERROR:
        return "One of the output image dimensions is less than 1 pixel.";

      case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
        return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

      case NPP_BAD_ARG_ERROR:
        return "NPP_BAD_ARGUMENT_ERROR";

      case NPP_COEFF_ERROR:
        return "Unallowable values of the transformation coefficients.";

      case NPP_RECT_ERROR:
        return "Size of the rectangle region is less than or equal to 1.";

      case NPP_QUAD_ERROR:
        return "The quadrangle is nonconvex or degenerates into triangle, line or point.";

      case NPP_MEM_ALLOC_ERR:
        return "NPP_MEMORY_ALLOCATION_ERROR";

      case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
        return "Number of levels for histogram is less than 2.";

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
        return "Unallowable values of the transformation coefficients.";

      case NPP_RECTANGLE_ERROR:
        return "Size of the rectangle region is less than or equal to 1.";

      case NPP_QUADRANGLE_ERROR:
        return "The quadrangle is nonconvex or degenerates into triangle, line or point.";

      case NPP_MEMORY_ALLOCATION_ERR:
        return "NPP_MEMORY_ALLOCATION_ERROR";

      case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
        return "Number of levels for histogram is less than 2.";

      case NPP_INVALID_HOST_POINTER_ERROR:
        return "NPP_INVALID_HOST_POINTER_ERROR";

      case NPP_INVALID_DEVICE_POINTER_ERROR:
        return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

      case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
        return "Number of levels for LUT is less than .2";

      case NPP_TEXTURE_BIND_ERROR:
        return "NPP_TEXTURE_BIND_ERROR";

      case NPP_WRONG_INTERSECTION_ROI_ERROR:
        return "NPP_WRONG_INTERSECTION_ROI_ERROR";

      case NPP_NOT_EVEN_STEP_ERROR:
        return "Step value is not pixel multiple.";

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
        return "Step is less or equal zero.";

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
        return "The given quadrangle has no intersection with either the source or "
               "destination ROI. Thus no operation was performed.";

      case NPP_MISALIGNED_DST_ROI_WARNING:
        return "Speed reduction due to uncoalesced memory accesses warning.";

      case NPP_AFFINE_QUAD_INCORRECT_WARNING:
        return "Indicates that the quadrangle passed to one of affine warping functions doesn't "
               "have necessary properties. First 3 vertices are used, the fourth vertex discarded.";

      case NPP_DOUBLE_SIZE_WARNING:
        return "Image size isn't multiple of two. Indicates that in case of 422/411/420 "
               "sampling the ROI width/height was modified for proper processing.";

      case NPP_WRONG_INTERSECTION_ROI_WARNING:
        return "The given ROI has no interestion with either the source or destination ROI. "
               "Thus no operation was performed.";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
      /* These are 6.0 or higher */
      case NPP_LUT_PALETTE_BITSIZE_ERROR:
        return "NPP_LUT_PALETTE_BITSIZE_ERROR";

      case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
        return "ZeroCrossing mode not supported.";

      case NPP_QUALITY_INDEX_ERROR:
        return "Image pixels are constant for quality index.";

      case NPP_CHANNEL_ORDER_ERROR:
        return "Wrong order of the destination channels.";

      case NPP_ZERO_MASK_VALUE_ERROR:
        return "All values of the mask are zero.";

      case NPP_NUMBER_OF_CHANNELS_ERROR:
        return "Bad or unsupported number of channel.";

      case NPP_COI_ERROR:
        return "Channel of interest is not 1, 2, or 3.";

      case NPP_DIVISOR_ERROR:
        return "Divisor is equal to zero.";

      case NPP_CHANNEL_ERROR:
        return "Illegal channel index.";

      case NPP_STRIDE_ERROR:
        return "Stride is less than the row length.";

      case NPP_ANCHOR_ERROR:
        return "Anchor point is outside mask.";

      case NPP_MASK_SIZE_ERROR:
        return "Lower bound is larger than upper bound.";

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
        return "Indicates that no operation was performed.";

      case NPP_DIVIDE_BY_ZERO_WARNING:
        return "Divisor is zero however does not terminate the execution.";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
        /* These are 7.0 or higher */
      case NPP_OVERFLOW_ERROR:
        return "Number overflows the upper or lower limit of the data type.";

      case NPP_CORRUPTED_DATA_ERROR:
        return "Processed data is corrupted.";
#endif
      default:
        return "< unknown error >";
    }
  }

  static std::string Message(NppStatus result, const char *details) {
    if (details && *details) {
      return make_string("npp error (", result, "): ", ErrorString(result),
                         "\nDetails:\n", details);
    } else {
      return make_string("npp error (", result, "): ", ErrorString(result));
    }
  }


  NppStatus result() const { return result_; }

 private:
  NppStatus result_;
};

template <>
inline void cudaResultCheck<NppStatus>(NppStatus status) {
  switch (status) {
  case NPP_SUCCESS:
    return;
  default:
    throw dali::NppError(status);
  }
}

// Obtain NPP library version or -1 if it is not available
int NPPGetVersion();

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_NPP_H_
