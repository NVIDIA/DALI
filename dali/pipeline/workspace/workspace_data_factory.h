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
  bool is_pinned = true;
  for (auto &tensor_ptr : t) {
    is_pinned = is_pinned && tensor_ptr->is_pinned();
  }
  return is_pinned;
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
 * Mappings from DALIOpType, DALITensorDevice to index in tensor_data_store_t
 */

constexpr size_t GetTensorStoreIndex(DALIOpType op_type, DALITensorDevice device) {
  return static_cast<size_t>(op_type) * static_cast<size_t>(DALITensorDevice::COUNT) +
         static_cast<size_t>(device);
}

constexpr size_t GetMaxTensorStoreIndex() {
  return GetTensorStoreIndex(DALIOpType::COUNT, static_cast<DALITensorDevice>(0));
}

constexpr DALIOpType GetOpType(size_t storage_idx) {
  return static_cast<DALIOpType>(storage_idx / static_cast<size_t>(DALITensorDevice::COUNT));
}

constexpr DALITensorDevice GetStorageDevice(size_t storage_idx) {
  return static_cast<DALITensorDevice>(storage_idx % static_cast<size_t>(DALITensorDevice::COUNT));
}

// We use a tuple that can hold Output Type from Device, Host, Mixed and Support workspaces,
// so we have a unifided place that can own any of this type.
// Additionally, we use order of those types deifned by GetTensorStoreIndex
// We have 4 workspaces with two possible Backends, obtatining 8 types
// TODO(klecki): this can be clearer as
// std::tuple<DeviceOutputType<CPUBackend>, DeviceOutputType<GPUBackend>,
//            HostOutputType<CPUBackend>, ...
// but that way we ensure correct order of types and not use 8 static_asserts
template <int storage_idx>
struct workspace_out_data_type_gen {
  using type = typename workspace_t<GetOpType(
      storage_idx)>::template output_t<storage_backend_t<GetStorageDevice(storage_idx)>>;
};

template <int storage_idx>
using workspace_out_data_t = typename workspace_out_data_type_gen<storage_idx>::type;

template <DALIOpType op_type, DALITensorDevice device>
using tensor_store_elem_t = workspace_out_data_t<GetTensorStoreIndex(op_type, device)>;

// using tensor_data_store_t =
//     std::tuple<workspace_out_data_t<0>, workspace_out_data_t<1>, workspace_out_data_t<2>,
//                workspace_out_data_t<3>, workspace_out_data_t<4>, workspace_out_data_t<5>,
//                workspace_out_data_t<6>, workspace_out_data_t<7>>;

using tensor_data_store_t = detail::tuple_generator_t<workspace_out_data_t,
                                                  detail::build_seq_t<0, GetMaxTensorStoreIndex()>>;

template <int op_type>
using workspace_blob_gen_type = std::vector<workspace_t<static_cast<DALIOpType>(op_type)>>;

using workspace_store_t =
    detail::tuple_generator_t<workspace_blob_gen_type,
                              detail::build_seq_t<0, static_cast<int>(DALIOpType::COUNT)>>;

template <DALIOpType op_type, DALITensorDevice device>
struct BatchFactoryImpl {
  static tensor_store_elem_t<op_type, device> CreateOutputBatch(int batch_size) {
    // Output batch from GPU, MIXED and SUPPORT Ops are shared_ptr<Something>
    using BatchType = typename tensor_store_elem_t<op_type, device>::element_type;
    return std::make_shared<BatchType>();
  }
  static_assert(op_type == DALIOpType::GPU || op_type == DALIOpType::MIXED,
                "Only GPU and MIXED handled by default case due to use of outermost shared_ptr and "
                "pinned mem usage");
};

// TODO(klecki): Should we use make_shared here as well?
template <DALITensorDevice device>
struct BatchFactoryImpl<DALIOpType::CPU, device> {
  static tensor_store_elem_t<DALIOpType::CPU, device> CreateOutputBatch(
      int batch_size) {
    DALI_ENFORCE(device == DALITensorDevice::CPU, "Only CPU outputs allowed");
    // Allocate `batch_size` Tensors for this ops
    // results and add them to the workspace.
    tensor_store_elem_t<DALIOpType::CPU, device> output(batch_size, nullptr);
    for (auto &tensor_ptr : output) {
      tensor_ptr.reset(new Tensor<storage_backend_t<device>>);
      // tensor_ptr->set_pinned(false);
    }
    SetPinned(output, false);
    return output;
  }
};

template <DALITensorDevice device>
struct BatchFactoryImpl<DALIOpType::SUPPORT, device> {
  static tensor_store_elem_t<DALIOpType::SUPPORT, device> CreateOutputBatch(
      int batch_size) {
    DALI_ENFORCE(device == DALITensorDevice::CPU, "Only CPU outputs allowed");
    tensor_store_elem_t<DALIOpType::SUPPORT, device> output(
        new Tensor<storage_backend_t<device>>);
    // output->set_pinned(false);
    SetPinned(output, false);
    return output;
  }
};

template <DALIOpType op_type, DALITensorDevice device>
struct FillStorageOwner {
  tensor_data_store_t operator()(int batch_size) {
    tensor_data_store_t result;
    std::get<GetTensorStoreIndex(op_type, device)>(result) =
        BatchFactoryImpl<op_type, device>::CreateOutputBatch(batch_size);
    return result;
  }
};

/**
 * @brief Used to go over all cases of DALIOpType and DALITensorDevice and call appropriate
 * template based on runtime values of those enums
 *
 * @tparam ToExecute Functor class parametrized by DALIOpType and DALITensorDevice
 * @tparam Ret return type
 * @tparam T Arguments to operator() of `ToExecute` functor
 * @param op_type runtime value of DALIOpType used to chose Functor
 * @param device runtime value of DALITensorDevice used to chose Functor
 * @param args Runtime arguments to operator()
 * @return Ret
 */
template <template <DALIOpType, DALITensorDevice> class ToExecute, typename Ret, typename... T>
Ret Switch_OpType_Device(DALIOpType op_type, DALITensorDevice device, T &&... args) {
  VALUE_SWITCH(op_type, op_type_static,
      (DALIOpType::GPU, DALIOpType::CPU, DALIOpType::MIXED, DALIOpType::SUPPORT),
  (
    VALUE_SWITCH(device, device_static, (DALITensorDevice::CPU, DALITensorDevice::GPU),
    (
      return ToExecute<op_type_static, device_static>{}(std::forward<T>(args)...);
    ), DALI_FAIL("Unexpected device"))  // NOLINT(whitespace/parens)
  ), DALI_FAIL("Unexpected op_type"));  // NOLINT(whitespace/parens)
}

template <template <DALIOpType> class ToExecute, typename Ret, typename... T>
Ret Switch_OpType(DALIOpType op_type, T &&... args) {
  VALUE_SWITCH(op_type, op_type_static,
      (DALIOpType::GPU, DALIOpType::CPU, DALIOpType::MIXED, DALIOpType::SUPPORT),
  (
    return ToExecute<op_type_static>{}(std::forward<T>(args)...);
  ), DALI_FAIL("Unexpected op_type"));  // NOLINT(whitespace/parens)
}

template <template <DALITensorDevice> class ToExecute, typename Ret, typename... T>
Ret Switch_Device(DALITensorDevice device, T &&... args) {
  VALUE_SWITCH(device, device_static, (DALITensorDevice::CPU, DALITensorDevice::GPU),
  (
    return ToExecute<device_static>{}(std::forward<T>(args)...);
  ), DALI_FAIL("Unexpected device"));  // NOLINT(whitespace/parens)
}

/**
 * @brief Create the instance of of Workspace::OutputType for runtime pair of op_type and device
 * storing it at apropraite index of tensor_data_store_t tuple
 * Other entries of tuple are empty
 */
inline tensor_data_store_t BatchFactory(DALIOpType op_type, DALITensorDevice device,
                                          int batch_size) {
  return Switch_OpType_Device<FillStorageOwner, tensor_data_store_t>(op_type, device, batch_size);
}

/**
 * @brief Set of wrappers for std::get, that for specified op_type and device
 * retrive the appropriate Workspace::OutputType from the tensor_data_store_t tuple
 */
template <DALIOpType op_type, DALITensorDevice device>
typename std::tuple_element<GetTensorStoreIndex(op_type, device), tensor_data_store_t>::type&
get_storage(tensor_data_store_t& owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <DALIOpType op_type, DALITensorDevice device>
typename std::tuple_element<GetTensorStoreIndex(op_type, device), tensor_data_store_t>::type&&
get_storage(tensor_data_store_t&& owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <DALIOpType op_type, DALITensorDevice device>
typename std::tuple_element<GetTensorStoreIndex(op_type, device), tensor_data_store_t>::type const&
get_storage(const tensor_data_store_t& owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}

template <DALIOpType op_type, DALITensorDevice device>
typename std::tuple_element<GetTensorStoreIndex(op_type, device), tensor_data_store_t>::type const&&
get_storage(const tensor_data_store_t&& owner) noexcept {
  return std::get<GetTensorStoreIndex(op_type, device)>(owner);
}





}  // namespace dali

#endif  // DALI_PIPELINE_WORKSPACE_WORKSPACE_DATA_FACTORY_H_
