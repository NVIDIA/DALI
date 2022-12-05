// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_NPP_DEBAYER_CALL_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_NPP_DEBAYER_CALL_H_

#include "dali/npp/npp.h"

namespace dali {
namespace kernels {
namespace debayer {

inline auto npp_debayer_call(const uint8_t *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI,
                             uint8_t *pDst, int nDstStep, NppiBayerGridPosition eGrid,
                             NppStreamContext nppStreamCtx) {
  // eInterpolation MUST be set to NPPI_INTER_UNDEFINED according to the NPP docs
  const auto eInterpolation = NPPI_INTER_UNDEFINED;
  return nppiCFAToRGB_8u_C1C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, eGrid,
                                   eInterpolation, nppStreamCtx);
}

inline auto npp_debayer_call(const uint16_t *pSrc, int nSrcStep, NppiSize oSrcSize,
                             NppiRect oSrcROI, uint16_t *pDst, int nDstStep,
                             NppiBayerGridPosition eGrid, NppStreamContext nppStreamCtx) {
  // eInterpolation MUST be set to NPPI_INTER_UNDEFINED according to the NPP docs
  const auto eInterpolation = NPPI_INTER_UNDEFINED;
  return nppiCFAToRGB_16u_C1C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, eGrid,
                                    eInterpolation, nppStreamCtx);
}

}  // namespace debayer
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_DEBAYER_NPP_DEBAYER_CALL_H_
