// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_SIGNAL_FFT_TRANSPOSE_CUH_
#define DALI_KERNELS_SIGNAL_FFT_TRANSPOSE_CUH_

#include <cuda_runtime.h>
#include "dali/core/geom/vec.h"

namespace dali {
namespace kernels {
namespace signal {

template <typename T>
struct TransposeBatchSampleDesc {
  T *out;
  const T *in;
  ptrdiff_t in_stride;
  ptrdiff_t out_size;
};

template <typename T>
__global__ void TransposeBatch(TransposeBatchSampleDesc<T> *samples, BlockDesc *blocks) {

}

}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif
