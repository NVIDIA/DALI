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

#ifndef DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_
#define DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_

#include <memory>
#include <utility>
#include <vector>

#include "dali/pipeline/graph/op_graph.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/host_workspace.h"
#include "dali/pipeline/workspace/mixed_workspace.h"
#include "dali/pipeline/workspace/support_workspace.h"

#include "dali/kernels/static_switch.h"
#include "dali/kernels/tuple_helpers.h"
#include "dali/pipeline/util/op_type_to_workspace.h"

namespace dali {


template <typename Backend>
void SetPinned(SupportWorkspace::output_t<Backend> &t, bool pinned) {
  t->set_pinned(pinned);
}

template <typename Backend>
void SetPinned(HostWorkspace::output_t<Backend> &t, bool pinned) {
  for (auto &tensor_ptr : t) {
    tensor_ptr->set_pinned(pinned);
  }
}

// Device is the same as Mixed
template <typename Backend>
void SetPinned(DeviceWorkspace::output_t<Backend> &t, bool pinned) {
  t->set_pinned(pinned);
}

template <typename Backend>
bool IsPinned(SupportWorkspace::output_t<Backend> &t) {
  return t->is_pinned();
}

template <typename Backend>
bool IsPinned(HostWorkspace::output_t<Backend> &t) {
  for (auto &tensor_ptr : t) {
    if (!tensor_ptr->is_pinned()) {
      return false;
    }
  }
  return true;
}

// Device is the same as Mixed
template <typename Backend>
bool IsPinned(DeviceWorkspace::output_t<Backend> &t) {
  return t->is_pinned();
}

template <typename Backend>
void Reserve(SupportWorkspace::output_t<Backend> &t, size_t new_num_bytes, int batch_size) {
  t->reserve(new_num_bytes);
}

template <typename Backend>
void Reserve(HostWorkspace::output_t<Backend> &t, size_t new_num_bytes, int batch_size) {
  for (auto &tensor_ptr : t) {
    tensor_ptr->reserve(new_num_bytes);
  }
}

// Device is the same as Mixed
template <typename Backend>
void Reserve(DeviceWorkspace::output_t<Backend> &t, size_t new_num_bytes, int batch_size) {
  t->reserve(new_num_bytes * batch_size);
}

template <typename Backend>
void Reserve(SupportWorkspace::output_t<Backend> &t, size_t new_num_bytes) {
  t->reserve(new_num_bytes);
}

template <typename Backend>
void Reserve(HostWorkspace::output_t<Backend> &t, size_t new_num_bytes) {
  for (auto &tensor_ptr : t) {
    tensor_ptr->reserve(new_num_bytes);
  }
}

// Device is the same as Mixed
template <typename Backend>
bool Reserve(DeviceWorkspace::output_t<Backend> &t, size_t new_num_bytes) {
  return t->reserve(new_num_bytes);
}


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

// We use a tuple that can hold Output Type from Device, Host, Mixed and Support workspaces,
// so we have a unifided place that can own any of this type.
// Additionally, we use order of those types deifned by GetTensorStoreIndex
// We have 4 workspaces with two possible Backends, obtatining 8 types
// This can be clearer as
// std::tuple<DeviceOutputType<CPUBackend>, DeviceOutputType<GPUBackend>,
//            HostOutputType<CPUBackend>, ...
// but that way we ensure correct order of types and not use 8 static_asserts
// :: Int -> Workspace Output Type
template <int storage_idx>
struct workspace_out_data_type_gen {
  using type = typename op_type_to_workspace_t<GetOpType(storage_idx)>::
      template output_t<storage_backend_t<GetStorageDevice(storage_idx)>>;
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

template <int op_type>
using workspace_blob_gen_type = std::vector<op_type_to_workspace_t<static_cast<OpType>(op_type)>>;

// Tuple used for generic workspace blob = tuple containing vectors for all Workspace types
using workspace_store_t =
    detail::tuple_generator_t<workspace_blob_gen_type,
                              detail::build_seq_t<0, static_cast<int>(OpType::COUNT)>>;

template <OpType op_type, StorageDevice device>
struct BatchFactoryImpl {
  static tensor_store_elem_t<op_type, device> CreateOutputBatch(int batch_size) {
    // Output batch from GPU, MIXED and SUPPORT Ops are shared_ptr<Something>
    using BatchType = typename tensor_store_elem_t<op_type, device>::element_type;
    return std::make_shared<BatchType>();
  }
  static_assert(op_type == OpType::GPU || op_type == OpType::MIXED,
                "Only GPU and MIXED handled by default case due to use of outermost shared_ptr and "
                "pinned mem usage");
};

// TODO(klecki): Should we use make_shared here as well?
template <StorageDevice device>
struct BatchFactoryImpl<OpType::CPU, device> {
  static tensor_store_elem_t<OpType::CPU, device> CreateOutputBatch(
      int batch_size) {
    DALI_ENFORCE(device == StorageDevice::CPU, "Only CPU outputs allowed");
    // Allocate `batch_size` Tensors for this ops
    // results and add them to the workspace.
    tensor_store_elem_t<OpType::CPU, device> output(batch_size, nullptr);
    for (auto &tensor_ptr : output) {
      tensor_ptr.reset(new Tensor<storage_backend_t<device>>);
    }
    SetPinned(output, false);
    return output;
  }
};

template <StorageDevice device>
struct BatchFactoryImpl<OpType::SUPPORT, device> {
  static tensor_store_elem_t<OpType::SUPPORT, device> CreateOutputBatch(
      int batch_size) {
    DALI_ENFORCE(device == StorageDevice::CPU, "Only CPU outputs allowed");
    tensor_store_elem_t<OpType::SUPPORT, device> output(
        new Tensor<storage_backend_t<device>>);
    SetPinned(output, false);
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
  VALUE_SWITCH(op_type, op_type_static,
      (OpType::GPU, OpType::CPU, OpType::MIXED, OpType::SUPPORT),
  (
    VALUE_SWITCH(device, device_static, (StorageDevice::CPU, StorageDevice::GPU),
    (
      return ToExecute<op_type_static, device_static>{}(std::forward<T>(args)...);
    ), DALI_FAIL("Unexpected device"))  // NOLINT(whitespace/parens)
  ), DALI_FAIL("Unexpected op_type"));  // NOLINT(whitespace/parens)
}

template <template <OpType> class ToExecute, typename Ret, typename... T>
Ret Switch_OpType(OpType op_type, T &&... args) {
  VALUE_SWITCH(op_type, op_type_static,
      (OpType::GPU, OpType::CPU, OpType::MIXED, OpType::SUPPORT),
  (
    return ToExecute<op_type_static>{}(std::forward<T>(args)...);
  ), DALI_FAIL("Unexpected op_type"));  // NOLINT(whitespace/parens)
}

template <template <StorageDevice> class ToExecute, typename Ret, typename... T>
Ret Switch_Device(StorageDevice device, T &&... args) {
  VALUE_SWITCH(device, device_static, (StorageDevice::CPU, StorageDevice::GPU),
  (
    return ToExecute<device_static>{}(std::forward<T>(args)...);
  ), DALI_FAIL("Unexpected device"));  // NOLINT(whitespace/parens)
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
typename std::tuple_element<GetTensorStoreIndex(op_type, device), tensor_data_store_queue_t>::type &
get_queue(tensor_data_store_queue_t &owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <OpType op_type, StorageDevice device>
typename std::tuple_element<GetTensorStoreIndex(op_type, device),
                            tensor_data_store_queue_t>::type &&
    get_queue(tensor_data_store_queue_t &&owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <OpType op_type, StorageDevice device>
typename std::tuple_element<GetTensorStoreIndex(op_type, device),
                            tensor_data_store_queue_t>::type const &
get_queue(const tensor_data_store_queue_t &owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <OpType op_type, StorageDevice device>
typename std::tuple_element<GetTensorStoreIndex(op_type, device),
                            tensor_data_store_queue_t>::type const &&
get_queue(const tensor_data_store_queue_t &&owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_
