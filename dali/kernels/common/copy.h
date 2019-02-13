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

#ifndef DALI_KERNELS_COMMON_COPY_H_
#define DALI_KERNELS_COMMON_COPY_H_

#include <cuda_runtime.h>
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/backend_tags.h"

namespace dali {
namespace kernels {

template <typename Storage1, typename T1, int ndim1, typename Storage2, typename T2, int ndim2>
void copy(const TensorView<Storage1, T1, ndim1> &out,  // NOLINT
          const TensorView<Storage2, T2, ndim2> &in,
          cudaStream_t stream = 0) {
  static_assert(sizeof(T1) == sizeof(T2), "Tensor elements must be of equal size");
  assert(in.shape == out.shape);

  if (!is_gpu_accessible<Storage1>::value) {
    if (is_cpu_accessible<Storage2>::value) {
      if (is_gpu_accessible<Storage2>::value)
        cudaStreamSynchronize(stream);  // or cudaDeviceSynchronize?
      memcpy(out.data, in.data, in.num_elements() * sizeof(T1));
    } else {
      cudaMemcpyAsync(out.data, in.data, in.num_elements() * sizeof(T1),
                      cudaMemcpyDeviceToHost, stream);
    }
  } else {
    if (is_gpu_accessible<Storage2>::value) {
      cudaMemcpyAsync(out.data, in.data, in.num_elements() * sizeof(T1),
                      cudaMemcpyDeviceToDevice, stream);
    } else {
      cudaMemcpyAsync(out.data, in.data, in.num_elements() * sizeof(T1),
                      cudaMemcpyHostToDevice, stream);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_COPY_H_
