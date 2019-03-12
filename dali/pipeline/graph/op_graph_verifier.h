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

#ifndef DALI_PIPELINE_GRAPH_OP_GRAPH_VERIFIER_H_
#define DALI_PIPELINE_GRAPH_OP_GRAPH_VERIFIER_H_

#include <set>
#include <vector>

#include "dali/pipeline/graph/op_graph.h"

namespace dali {

template <typename T, size_t N>
constexpr size_t Size(T (&)[N]) {
  return N;
}

template <typename T>
constexpr auto Size(const T &t) -> decltype(t.size()) {
  return t.size();
}

template <OpType op_type>
struct parent_constraints;

template <>
struct parent_constraints<OpType::SUPPORT> {
  static constexpr OpType allowed_parents[] = {OpType::SUPPORT};
  static constexpr StorageDevice allowed_input_tensors[] = {StorageDevice::CPU};
  static constexpr OpType allowed_input_ops[] = {OpType::SUPPORT};
  static constexpr bool supports_argument_inputs = false;
};

template <>
struct parent_constraints<OpType::CPU> {
  static constexpr OpType allowed_parents[] = {OpType::CPU, OpType::SUPPORT};
  static constexpr StorageDevice allowed_input_tensors[] = {StorageDevice::CPU};
  static constexpr OpType allowed_input_ops[] = {OpType::CPU};
  static constexpr bool supports_argument_inputs = true;
};

template <>
struct parent_constraints<OpType::MIXED> {
  static constexpr OpType allowed_parents[] = {OpType::CPU, OpType::SUPPORT};
  static constexpr StorageDevice allowed_input_tensors[] = {StorageDevice::CPU};
  static constexpr OpType allowed_input_ops[] = {OpType::CPU};
  static constexpr bool supports_argument_inputs = true;
};

template <>
struct parent_constraints<OpType::GPU> {
  static constexpr OpType allowed_parents[] = {OpType::GPU, OpType::MIXED, OpType::SUPPORT};
  static constexpr StorageDevice allowed_input_tensors[] = {StorageDevice::CPU, StorageDevice::GPU};
  static constexpr OpType allowed_input_ops[] = {OpType::MIXED, OpType::GPU};
  static constexpr bool supports_argument_inputs = true;
};

template <typename T, size_t N>
static constexpr bool is_in_array(const T value, const T (&arr)[N], size_t idx) {
  // we encountered the end of array -> false
  // otherwise we found at current position or search in next position recursivelly
  return idx < N && (value == arr[idx] || is_in_array(value, arr, idx + 1));
}

template <OpType op_type>
static constexpr bool allows_op_input(OpType parent) {
  return is_in_array(parent, parent_constraints<op_type>::allowed_input_ops, 0);
}

template <OpType op_type>
static constexpr bool allows_tensor_input(StorageDevice device) {
  return is_in_array(device, parent_constraints<op_type>::allowed_input_tensors, 0);
}


DLL_PUBLIC std::vector<int> ArgumentInputConstraints();
DLL_PUBLIC std::vector<std::set<OpType>> ParentOpTypeConstraints();

// NB: we could collect all the errors in graph before reporting them to user
DLL_PUBLIC void CheckGraphConstraints(const OpGraph &op_graph);

}  // namespace dali

#endif  // DALI_PIPELINE_GRAPH_OP_GRAPH_VERIFIER_H_
