// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/util/ocv.h"

#include "ndll/error_handling.h"

namespace ndll {

int OCVInterpForNDLLInterp(NDLLInterpType type, int *ocv_type) {
  switch (type) {
  case NDLL_INTERP_NN:
    *ocv_type =  cv::INTER_NEAREST;
    break;
  case NDLL_INTERP_LINEAR:
    *ocv_type =  cv::INTER_LINEAR;
    break;
  case NDLL_INTERP_CUBIC:
    *ocv_type =  cv::INTER_CUBIC;
    break;
  default:
    return NDLLError;
  }
  return NDLLSuccess;
}

}  // namespace ndll
