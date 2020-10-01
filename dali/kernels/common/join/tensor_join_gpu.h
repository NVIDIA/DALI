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

#ifndef DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_H_
#define DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_H_

#include "dali/kernels/kernel.h"
#include "dali/kernels/common/type_erasure.h"
#include "dali/kernels/common/join/tensor_join_gpu_impl.h"

namespace dali {
namespace kernels {

template <typename T, bool new_axis>
class TensorJoinGPU : public TensorJoinImplGPU<type_of_size<sizeof(T)>, new_axis> {
 public:
  using Placeholder = type_of_size<sizeof(T)>;
  using Base = TensorJoinImplGPU<Placeholder, new_axis>;

  template <int in_ndim>
  void Run(KernelContext &ctx, const OutListGPU<T> &out,
           span<const InListGPU<T, in_ndim> *> &in_lists) {
    using U = type_of_size<sizeof(T)>;
    auto *lists = reinterpret_cast<const InListGPU<Placeholder, in_ndim> *>(in_lists.data());
    Base::Run(ctx,
        reinterpret_cast<const OutListGPU<U> &>(out),
        make_span(lists, in_lists.size())
    );
  }
 private:

};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_JOIN_TENSOR_JOIN_GPU_H_