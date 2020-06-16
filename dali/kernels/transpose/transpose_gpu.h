// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_H_
#define DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_H_

#include <memory>
#include "dali/core/tensor_view.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

class DLL_PUBLIC TransposeGPU {
 public:
  TransposeGPU();
  ~TransposeGPU();

  KernelRequirements Setup(
        KernelContext &ctx,
        const TensorListShape<> &in_shape,
        span<const int> permutation,
        int element_size);

  void Run(KernelContext &ctx, void *const *out, const void *const *in);

  template <typename T>
  void Run(KernelContext &ctx, const OutListGPU<T> &out, const InListGPU<T> &in) {
    CheckShapes(in.shape, out.shape, sizeof(T));
    Run(ctx,
        reinterpret_cast<void *const*>(out.data.data()),
        reinterpret_cast<const void *const*>(in.data.data()));
  }

 private:
  void CheckShapes(const TensorListShape<> &in_shape,
                   const TensorListShape<> &out_shape,
                   int element_size);

  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_H_
