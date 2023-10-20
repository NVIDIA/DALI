// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_CAST_GPU_H_
#define DALI_KERNELS_COMMON_CAST_GPU_H_

#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {
namespace cast {

/**
 * @brief Converts from `In` to `Out`, via ConvertSat
 * 
 * @tparam Out 
 * @tparam In 
 */
template <typename Out, typename In>
struct DLL_PUBLIC CastGPU {
  void Run(KernelContext& ctx,
           const TensorListView<StorageGPU, Out, 1>& out,
           const TensorListView<StorageGPU, const In, 1>& in);
};

}  // namespace cast
}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_COMMON_CAST_GPU_H_
