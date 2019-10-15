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

#ifndef DALI_KERNELS_KERNEL_PARAMS_H_
#define DALI_KERNELS_KERNEL_PARAMS_H_

#include <type_traits>
#include "dali/core/tensor_view.h"

namespace dali {
namespace kernels {

template <typename StorageBackend, typename ElementType, int sample_dim>
using InList = TensorListView<StorageBackend, const ElementType, sample_dim>;

template <typename StorageBackend, typename ElementType, int sample_dim>
using OutList = TensorListView<StorageBackend, ElementType, sample_dim>;

template <typename ElementType, int sample_dim>
using InListCPU = InList<StorageCPU, ElementType, sample_dim>;

template <typename ElementType, int sample_dim>
using InListGPU = InList<StorageGPU, ElementType, sample_dim>;

template <typename ElementType, int sample_dim>
using OutListCPU = OutList<StorageCPU, ElementType, sample_dim>;

template <typename ElementType, int sample_dim>
using OutListGPU = OutList<StorageGPU, ElementType, sample_dim>;

template <typename StorageBackend, typename ElementType, int sample_dim>
using InTensor = TensorView<StorageBackend, const ElementType, sample_dim>;

template <typename StorageBackend, typename ElementType, int sample_dim>
using OutTensor = TensorView<StorageBackend, ElementType, sample_dim>;

template <typename ElementType, int sample_dim>
using InTensorCPU = InTensor<StorageCPU, ElementType, sample_dim>;

template <typename ElementType, int sample_dim>
using InTensorGPU = InTensor<StorageGPU, ElementType, sample_dim>;

template <typename ElementType, int sample_dim>
using OutTensorCPU = OutTensor<StorageCPU, ElementType, sample_dim>;

template <typename ElementType, int sample_dim>
using OutTensorGPU = OutTensor<StorageGPU, ElementType, sample_dim>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_KERNEL_PARAMS_H_
