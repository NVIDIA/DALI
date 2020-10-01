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

#ifndef DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_H_
#define DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_H_

#include "dali/kernels/kernel.h"
#include "dali/kernels/common/type_erasure.h"

namespace dali {
namespace kernels {
namespace tensor_join {

template <typename T, bool new_axis>
class DLL_PUBLIC TensorJoinImplGPU {
 public:
  static_assert(std::is_same<T, type_of_size<sizeof(T)>>::value,
                "This class must be used with a type prouced by `type_of_size<size>`");

  void Setup(KernelContext &ctx, span<const TensorListShape<> *> &in_shapes, int axis);

  void Run(KernelContext &ctx, OutListGPU<T> &out, span<const InListGPU<T> *> &in_lists);
 private:
  struct Impl;
};

}  // namespate tensor_join
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_IMPL_H_
