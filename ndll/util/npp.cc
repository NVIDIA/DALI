// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/util/npp.h"

#include "ndll/error_handling.h"

namespace ndll {

int NPPInterpForNDLLInterp(NDLLInterpType type, NppiInterpolationMode *npp_type) {
  switch (type) {
  case NDLL_INTERP_NN:
    *npp_type =  NPPI_INTER_NN;
    break;
  case NDLL_INTERP_LINEAR:
    *npp_type =  NPPI_INTER_LINEAR;
    break;
  case NDLL_INTERP_CUBIC:
    *npp_type =  NPPI_INTER_CUBIC;
    break;
  default:
    return NDLLError;
  }
  return NDLLSuccess;
}

}  // namespace ndll
