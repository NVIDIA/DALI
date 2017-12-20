// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_UTIL_OCV_H_
#define NDLL_UTIL_OCV_H_

#include <opencv2/opencv.hpp>

#include "ndll/common.h"

namespace ndll {

int OCVInterpForNDLLInterp(NDLLInterpType type, int *ocv_type);

}  // namespace ndll

#endif  // NDLL_UTIL_OCV_H_
