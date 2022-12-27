// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_
#define DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_

#include <memory>
#include <utility>
#include <vector>

#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/workspace.h"

#include "dali/core/static_switch.h"
#include "dali/core/tuple_helpers.h"

namespace dali {

/*
 * Mappings from OpType, StorageDevice to index in tensor_data_store_queue_t
 */

constexpr size_t GetTensorStoreIndex(OpType op_type, StorageDevice device) {
  return static_cast<size_t>(op_type) * static_cast<size_t>(StorageDevice::COUNT) +
         static_cast<size_t>(device);
}

constexpr size_t GetMaxTensorStoreIndex() {
  return GetTensorStoreIndex(OpType::COUNT, static_cast<StorageDevice>(0));
}

constexpr OpType GetOpType(size_t storage_idx) {
  return static_cast<OpType>(storage_idx / static_cast<size_t>(StorageDevice::COUNT));
}

constexpr StorageDevice GetStorageDevice(size_t storage_idx) {
  return static_cast<StorageDevice>(storage_idx % static_cast<size_t>(StorageDevice::COUNT));
}

/**
 * @brief Maps storage index to the output type stored in the workspace
 */
template <int storage_idx>
struct workspace_out_data_type_gen {
  using type = std::shared_ptr<TensorList<storage_backend_t<GetStorageDevice(storage_idx)>>>;
};

// Helper struct for buffering Storage for Workspace Output Types
// If we have more than one elements, we are Queueing them, and we allow indexing into queue.
// If we only store 1 element, this means we are not queueing, and we always return that single
// element
template <typename StoredType>
struct StoreBufferQueue {
  static constexpr size_t unbuffered_size = 1;
  std::vector<StoredType> store;

  StoreBufferQueue() = default;

  explicit StoreBufferQueue(size_t initial_size) {
    store.resize(initial_size);
  }

  bool IsBuffered() const {
    return store.size() > unbuffered_size;
  }

  void Queue(size_t elements) {
    store.resize(elements);
  }

  auto operator[] (size_t index) -> decltype(store[index]) {
    if (!IsBuffered()) {
      return store[0];
    }
    return store[index];
  }

  auto begin() -> decltype(store.begin()) {
    return store.begin();
  }

  auto end() -> decltype(store.end()) {
    return store.end();
  }

  auto operator[] (size_t index) const -> decltype(store[index]) {
    if (!IsBuffered()) {
      return store[0];
    }
    return store[index];
  }

  auto begin() const -> decltype(store.begin()) {
    return store.begin();
  }

  auto end() const -> decltype(store.end()) {
    return store.end();
  }

  size_t size() const {
    return store.size();
  }
};

// Generator for Queue of Worskpace Output Type, indexed by GetTensorStoreIndex()
template <int storage_idx>
using workspace_out_data_queue_t =
    StoreBufferQueue<typename workspace_out_data_type_gen<storage_idx>::type>;

// OpType -> StorageDevice -> Workspace Output Type
template <OpType op_type, StorageDevice device>
using tensor_store_elem_t =
    typename workspace_out_data_type_gen<GetTensorStoreIndex(op_type, device)>::type;

// Tuple containing Queues for Workspace Output Type generated for all values
using tensor_data_store_queue_t =
    detail::tuple_generator_t<workspace_out_data_queue_t,
                              detail::build_seq_t<0, GetMaxTensorStoreIndex()>>;

using workspace_blob_gen_type = std::vector<Workspace>;

template <OpType op_type, StorageDevice device>
struct BatchFactoryImpl {
  static tensor_store_elem_t<op_type, device> CreateOutputBatch(int batch_size) {
    using BatchType = typename tensor_store_elem_t<op_type, device>::element_type;
    auto output = std::make_shared<BatchType>(batch_size);
    if (op_type == OpType::CPU) {
      output->set_pinned(false);
    }
    return output;
  }
};


template <OpType op_type, StorageDevice device>
struct FillStorageOwner {
  tensor_data_store_queue_t operator()(int batch_size, int queue_size) {
    tensor_data_store_queue_t result;
    auto& queue = std::get<GetTensorStoreIndex(op_type, device)>(result);
    queue.Queue(queue_size);
    for (auto& elem : queue) {
      elem = BatchFactoryImpl<op_type, device>::CreateOutputBatch(batch_size);
    }
    return result;
  }
};

/**
 * @brief Used to go over all cases of OpType and StorageDevice and call appropriate
 * template based on runtime values of those enums
 *
 * @tparam ToExecute Functor class parametrized by OpType and StorageDevice
 * @tparam Ret return type
 * @tparam T Arguments to operator() of `ToExecute` functor
 * @param op_type runtime value of OpType used to chose Functor
 * @param device runtime value of StorageDevice used to chose Functor
 * @param args Runtime arguments to operator()
 * @return Ret
 */
template <template <OpType, StorageDevice> class ToExecute, typename Ret, typename... T>
Ret Switch_OpType_Device(OpType op_type, StorageDevice device, T &&... args) {
  Ret ret;
  VALUE_SWITCH(op_type, op_type_static,
      (OpType::GPU, OpType::CPU, OpType::MIXED),
  (
    VALUE_SWITCH(device, device_static, (StorageDevice::CPU, StorageDevice::GPU),
    (
      ret = ToExecute<op_type_static, device_static>{}(std::forward<T>(args)...);
    ), DALI_FAIL("Unexpected device"))  // NOLINT(whitespace/parens)
  ), DALI_FAIL("Unexpected op_type"));  // NOLINT(whitespace/parens)
  return ret;
}

template <template <OpType> class ToExecute, typename Ret, typename... T>
Ret Switch_OpType(OpType op_type, T &&... args) {
  Ret ret;
  VALUE_SWITCH(op_type, op_type_static,
      (OpType::GPU, OpType::CPU, OpType::MIXED),
  (
    ret = ToExecute<op_type_static>{}(std::forward<T>(args)...);
  ), DALI_FAIL("Unexpected op_type"));  // NOLINT(whitespace/parens)
  return ret;
}

template <template <StorageDevice> class ToExecute, typename Ret, typename... T>
Ret Switch_Device(StorageDevice device, T &&... args) {
  Ret ret;
  VALUE_SWITCH(device, device_static, (StorageDevice::CPU, StorageDevice::GPU),
  (
    ret = ToExecute<device_static>{}(std::forward<T>(args)...);
  ), DALI_FAIL("Unexpected device"));  // NOLINT(whitespace/parens)
  return ret;
}

/**
 * @brief Create the instance of of Workspace::OutputType for runtime pair of op_type and device
 * storing it at apropraite index of tensor_data_store_queue_t tuple
 * Other entries of tuple are empty
 */
inline tensor_data_store_queue_t BatchFactory(OpType op_type, StorageDevice device,
                                              int batch_size, int queue_size = 1) {
  return Switch_OpType_Device<FillStorageOwner, tensor_data_store_queue_t>(op_type, device,
                                                                           batch_size, queue_size);
}

/**
 * @brief Set of wrappers for std::get, that for specified op_type and device
 * retrive the appropriate Workspace::OutputType from the tensor_data_store_queue_t tuple
 */
template <OpType op_type, StorageDevice device>
std::tuple_element_t<GetTensorStoreIndex(op_type, device), tensor_data_store_queue_t> &
get_queue(tensor_data_store_queue_t &owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <OpType op_type, StorageDevice device>
std::tuple_element_t<GetTensorStoreIndex(op_type, device),
                     tensor_data_store_queue_t> &&
    get_queue(tensor_data_store_queue_t &&owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <OpType op_type, StorageDevice device>
std::tuple_element_t<GetTensorStoreIndex(op_type, device),
                     tensor_data_store_queue_t> const &
get_queue(const tensor_data_store_queue_t &owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <OpType op_type, StorageDevice device>
std::tuple_element_t<GetTensorStoreIndex(op_type, device),
                     tensor_data_store_queue_t> const &&
get_queue(const tensor_data_store_queue_t &&owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_
