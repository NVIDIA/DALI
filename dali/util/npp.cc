// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/util/npp.h"

#include "dali/error_handling.h"

namespace dali {

int NPPInterpForDALIInterp(DALIInterpType type, NppiInterpolationMode *npp_type) {
  switch (type) {
  case DALI_INTERP_NN:
    *npp_type =  NPPI_INTER_NN;
    break;
  case DALI_INTERP_LINEAR:
    *npp_type =  NPPI_INTER_LINEAR;
    break;
  case DALI_INTERP_CUBIC:
    *npp_type =  NPPI_INTER_CUBIC;
    break;
  default:
    return DALIError;
  }
  return DALISuccess;
}

}  // namespace dali
