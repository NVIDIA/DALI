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

#ifndef DALI_KERNELS_IMGPROC_GEOM_NPP_REMAP_CALL_H_
#define DALI_KERNELS_IMGPROC_GEOM_NPP_REMAP_CALL_H_

#include "dali/npp/npp.h"

namespace dali {
namespace kernels {
namespace remap {
namespace detail {

/*
 * Faking type specializations for NPP C-API.
 *
 * Since NPP has C-API, there are no overloads. Furthermore, NPP uses non-native types for data.
 * In order to be able to call the NPP function for any given type in the kernel with one call,
 * these overloads are implemented here, together with native types for data where necessary.
 */

template<int nchannels = 3>
inline auto
npp_remap_call(const uint8_t *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
               const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep,
               uint8_t *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
               NppStreamContext nppStreamCtx) {
  return nppiRemap_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                              pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}


template<>
inline auto
npp_remap_call<1>(const uint8_t *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                  const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep,
                  uint8_t *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                  NppStreamContext nppStreamCtx) {
  return nppiRemap_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                              pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx);
}


template<int nchannels = 3>
inline auto
npp_remap_call(const uint16_t *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
               const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep,
               uint16_t *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
               NppStreamContext nppStreamCtx) {
  return nppiRemap_16u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap,
                               nYMapStep, pDst, nDstStep, oDstSizeROI, eInterpolation,
                               nppStreamCtx);
}


template<>
inline auto
npp_remap_call<1>(const uint16_t *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                  const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep,
                  uint16_t *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                  NppStreamContext nppStreamCtx) {
  return nppiRemap_16u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap,
                               nYMapStep, pDst, nDstStep, oDstSizeROI, eInterpolation,
                               nppStreamCtx);
}


template<int nchannels = 3>
inline auto
npp_remap_call(const int16_t *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
               const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep,
               int16_t *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
               NppStreamContext nppStreamCtx) {
  return nppiRemap_16s_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap,
                               nYMapStep, pDst, nDstStep, oDstSizeROI, eInterpolation,
                               nppStreamCtx);
}


template<>
inline auto
npp_remap_call<1>(const int16_t *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                  const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep,
                  int16_t *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                  NppStreamContext nppStreamCtx) {
  return nppiRemap_16s_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap,
                               nYMapStep, pDst, nDstStep, oDstSizeROI, eInterpolation,
                               nppStreamCtx);
}


template<int nchannels = 3>
inline auto
npp_remap_call(const float *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
               const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep,
               float *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
               NppStreamContext nppStreamCtx) {
  return nppiRemap_32f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap,
                               nYMapStep, pDst, nDstStep, oDstSizeROI, eInterpolation,
                               nppStreamCtx);
}


template<>
inline auto
npp_remap_call<1>(const float *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                  const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep,
                  float *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                  NppStreamContext nppStreamCtx) {
  return nppiRemap_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap,
                               nYMapStep, pDst, nDstStep, oDstSizeROI, eInterpolation,
                               nppStreamCtx);
}


template<int nchannels = 3>
inline auto
npp_remap_call(const double *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
               const double *pXMap, int nXMapStep, const double *pYMap, int nYMapStep,
               double *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
               NppStreamContext nppStreamCtx) {
  return nppiRemap_64f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap,
                               nYMapStep, pDst, nDstStep, oDstSizeROI, eInterpolation,
                               nppStreamCtx);
}


template<>
inline auto
npp_remap_call<1>(const double *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                  const double *pXMap, int nXMapStep, const double *pYMap, int nYMapStep,
                  double *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                  NppStreamContext nppStreamCtx) {
  return nppiRemap_64f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap,
                               nYMapStep, pDst, nDstStep, oDstSizeROI, eInterpolation,
                               nppStreamCtx);
}

}  // namespace detail
}  // namespace remap
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_GEOM_NPP_REMAP_CALL_H_
