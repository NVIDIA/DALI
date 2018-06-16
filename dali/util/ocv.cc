// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

}  // namespace dali
