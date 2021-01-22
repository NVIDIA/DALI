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

#ifndef DALI_PIPELINE_DATA_TYPE_TRAITS_H_
#define DALI_PIPELINE_DATA_TYPE_TRAITS_H_

namespace dali {

template <typename Backend>
class TensorVector;

template <typename Backend>
class TensorList;

class CPUBackend;
class MixedBackend;
class GPUBackend;

template <typename Backend>
struct is_backend {
  static constexpr bool value = std::is_same<Backend, CPUBackend>::value ||
                                std::is_same<Backend, MixedBackend>::value ||
                                std::is_same<Backend, GPUBackend>::value;
};

template <template <typename> class MaybeTensorVector, typename Backend>
struct is_tensor_vector {
  static constexpr bool value =
      std::is_same<MaybeTensorVector<Backend>, TensorVector<Backend>>::value;
};

template <template <typename> class MaybeTensorList, typename Backend>
struct is_tensor_list {
  static constexpr bool value = std::is_same<MaybeTensorList<Backend>, TensorList<Backend>>::value;
};

/**
 * Verifies, that T is proper batch container for DALI
 *
 * Batch container is proper when it has a defined backend and is TensorVector or a TensorList
 */
template <template <typename Backend_> class T, typename Backend>
struct is_batch_container {
  static constexpr bool value =
      is_backend<Backend>::value &&
      (is_tensor_vector<T, Backend>::value || is_tensor_list<T, Backend>::value);
};

namespace test {
static_assert(is_batch_container<TensorVector, CPUBackend>::value, "Test failed");
static_assert(is_batch_container<TensorVector, GPUBackend>::value, "Test failed");
static_assert(is_batch_container<TensorList, CPUBackend>::value, "Test failed");
static_assert(is_batch_container<TensorList, GPUBackend>::value, "Test failed");
}  // namespace test

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TYPE_TRAITS_H_
