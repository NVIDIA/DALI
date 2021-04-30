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

#include "dali/util/npp.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"

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
