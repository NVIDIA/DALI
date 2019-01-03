// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_CUDA_UTILS_H_
#define DALI_UTIL_CUDA_UTILS_H_

#include <cuda_fp16.h>  // for __half & related methods
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>  // for __align__ & CUDART_VERSION
#include "dali/util/dynlink_cuda.h"
#include "dali/error_handling.h"

namespace dali {

// Only supported on the GPU
typedef __half float16;

// Compatible wrapper for CUDA 8 which does not have builtin
// static_cast<float16>
template <typename dst>
__device__ inline dst StaticCastGpu(float val) {
  return static_cast<dst>(val);
}

#if defined(__CUDACC__) && defined(CUDART_VERSION) && CUDART_VERSION < 9000
template <>
__device__ inline float16 StaticCastGpu(float val) {
  return __float2half(static_cast<float>(val));
}
#endif  // defined(CUDART_VERSION) && CUDART_VERSION < 9000

// Starts profiling DALI
inline void DALIProfilerStart() { cudaProfilerStart(); }

inline void DALIProfilerStop() { cudaProfilerStop(); }

// CUDA checking
template <typename T>
inline void cudaResultCheck(T status);

template <typename T>
inline void cudaResultCheck(T status) {}

template <>
inline void cudaResultCheck<cudaError_t>(cudaError_t status) {
    if (status != cudaSuccess) {
      dali::string error = dali::string("CUDA runtime api error \"") +
        cudaGetErrorString(status) + "\"";
      DALI_FAIL(error);
    }
}

template <>
inline void cudaResultCheck<CUresult>(CUresult status) {
    if (status != CUDA_SUCCESS) {
      const char *cudaErrorStr;
      cuGetErrorString(status, &cudaErrorStr);
      dali::string error = dali::string("CUDA driver api error \"") +
        dali::string(cudaErrorStr) + "\"";
      DALI_FAIL(error);
    }
}

}  // end namespace dali

// For calling CUDA library functions (cudaError_t from runtime API and CUresult from driver API)
#define CUDA_CALL(code)                 \
  do {                                  \
    using CUDA_TYPE = decltype(code);   \
    CUDA_TYPE status = code;            \
    dali::cudaResultCheck<CUDA_TYPE>(status); \
  } while (0)

#endif  // DALI_UTIL_CUDA_UTILS_H_
