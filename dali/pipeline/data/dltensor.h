// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_DLTENSOR_H_
#define DALI_PIPELINE_DATA_DLTENSOR_H_

#include <cuda_runtime_api.h>
#include <cassert>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "third_party/dlpack/include/dlpack/dlpack.h"

#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

//////////////////////////////////////////////////////////////////////////////
// DLPack utilities

using DLMTensorPtr = std::unique_ptr<DLManagedTensor, void(*)(DLManagedTensor*)>;

/** A deleter which calls `DLManagedTensor::deleter` */
DLL_PUBLIC void DLMTensorPtrDeleter(DLManagedTensor* dlm_tensor_ptr);

/** Converts a DALI type to DLPack type. */
DLL_PUBLIC DLDataType ToDLType(DALIDataType type);

/** Converts a DLPack type to DALI type. */
DLL_PUBLIC DALIDataType ToDALIType(const DLDataType &dl_type);

/** Returns type string for given DLPack type
 *
 * The text representation looks like:
 * <type><bits>[x<lanes>]
 * with x<lanes> present only if the number of lanes is > 1
 *
 * Examples:
 * u8     - 8-bit unsigned integer
 * i32    - 32-bit signed integer
 * f64    - 64-bit floating point number
 * bf16   - bfloat16
 * b8     - 8-bit boolean
 * c64    - 64-bit complex number
 * f32x4 - 128-bit vector consisting of 4 32-bit floating point numbers
 *
 * If the code is unknown, the type code is replaced by '<unknown:value>' - a type with an unknown
 * code 42, 2 lanes and 32-bits would look like <unknown:42>32x2
 */
inline std::string to_string(const DLDataType &dl_type) {
  const char *code_str[] = {
    "i", "u", "f", "p", "bf", "c", "b"
  };
  std::stringstream ss;
  if (dl_type.code < std::size(code_str))
    ss << code_str[dl_type.code];
  else
    ss << "<unknown:" << dl_type.code + 0 << ">";
  ss << dl_type.bits + 0;
  if (dl_type.lanes > 1)
    ss << 'x' << dl_type.lanes + 0;
  return ss.str();
}

inline std::ostream &operator<<(std::ostream &os, const DLDataType &dl_type) {
  return os << to_string(dl_type);
}

constexpr DLDevice ToDLDevice(bool is_device, bool is_pinned, int device_id) {
  if (is_device)
    return {kDLCUDA, device_id};
  else
    return {is_pinned ? kDLCUDAHost : kDLCPU, 0};
}

//////////////////////////////////////////////////////////////////////////////
// DLTensorResource

/** Default non-owning payload for DLPack tensors. */
struct TensorViewPayload {
  TensorShape<> shape, strides;
};

DLL_PUBLIC void EnqueueForDeletion(std::shared_ptr<void> data, int device_id);

/** Default ownership-sharing payload for DLPack tensors. */
struct SharedTensorPayload : TensorViewPayload {
  std::shared_ptr<void> data;

  SharedTensorPayload() = default;
  SharedTensorPayload(TensorShape<> shape, TensorShape<> strides, std::shared_ptr<void> data)
  : TensorViewPayload{ std::move(shape), std::move(strides) }
  , data(std::move(data)) {}
};


/** A wrapper for DLManagedTensor along with its `context_manager`.
 *
 * This is a non-intuitive circular-reference structure.
 * DLManagedTensor lives inside the "resource", but the context_manager points to the resource.
 *
 * The diagram below depicts a typical relationship between DLTensorResource and its members:
 *
 * ```
 * DLTensorResource   <-------------+
 * |                                |
 * +-- DLManagedTensor              |
 * |   |                            |
 * |   +-- DLTensor                 |
 * |   |   |                        |
 * |   |   +-- *shape --------+     |
 * |   |   +-- *strides ------)--+  |
 * |   |   +-- ...            |  |  |
 * |   |                      |  |  |
 * |   +- *context_manager ---)--)--+
 * |   +- *deleter            |  |
 * |                          |  |
 * +-- Payload                |  |
 *     |                      |  |
 *     +-- shape  <-----------+  |
 *     +-- strides  <------------+
 *     +-- ...
 * ```
 *
 * You can use any payload structure of your choice, but it must provide the storage for DLTensor's
 * `shapes` (and `strides`, if necessary).
 */
template <typename Payload>
struct DLTensorResource {
  template <typename... PayloadArgs>
  explicit DLTensorResource(PayloadArgs &&...args)
  : dlm_tensor{{}, this, dlm_deleter}
  , payload{std::forward<PayloadArgs>(args)...} {}

  ~DLTensorResource() {}


  DLManagedTensor dlm_tensor{};
  Payload payload;

  template <typename... PayloadArgs>
  static std::unique_ptr<DLTensorResource> Create(PayloadArgs &&...args) {
    return std::make_unique<DLTensorResource>(std::forward<PayloadArgs>(args)...);
  }

  static void dlm_deleter(DLManagedTensor *tensor) {
    if (tensor == nullptr)
      return;
    auto *This = static_cast<DLTensorResource *>(tensor->manager_ctx);
    assert(&This->dlm_tensor == tensor);  // is that always the case?
    delete This;
  }
};

template <>
inline DLTensorResource<SharedTensorPayload>::~DLTensorResource() {
  if (dlm_tensor.dl_tensor.device.device_type == kDLCUDAHost ||
      dlm_tensor.dl_tensor.device.device_type == kDLCUDAManaged) {
    int current_dev = 0;
    CUDA_DTOR_CALL(cudaGetDevice(&current_dev));
    EnqueueForDeletion(std::move(payload.data), current_dev);
  } else if (dlm_tensor.dl_tensor.device.device_type == kDLCUDA) {
    EnqueueForDeletion(std::move(payload.data), dlm_tensor.dl_tensor.device.device_id);
  }
}

/** Type-erases the DLTensorResource and returns a smart pointer to the contained DLManagedTensor.
 */
template <typename Payload>
DLMTensorPtr ToDLMTensor(std::unique_ptr<DLTensorResource<Payload>> rsrc) {
  return { &rsrc.release()->dlm_tensor, DLMTensorPtrDeleter };
}

namespace detail {
/** Populates the DLTensor stored in `rsrc`. Shapes and strides will point to `rsrc.payload`. */
template <typename Payload>
void InitResourceDLTensor(DLTensorResource<Payload> &rsrc,
                          void *data, DALIDataType type,
                          bool device, bool pinned, int device_id) {
  auto &tensor = rsrc.dlm_tensor.dl_tensor;
  tensor = {};
  tensor.data = data;
  tensor.shape = rsrc.payload.shape.data();
  tensor.ndim = rsrc.payload.shape.size();
  tensor.strides = rsrc.payload.strides.empty() ? nullptr : rsrc.payload.strides.data();
  tensor.device = ToDLDevice(device, pinned, device_id);
  tensor.dtype = ToDLType(type);
}
}  // namespace detail

/** Constructs a DLTensorResource WITHOUT data ownership. */
inline auto MakeDLTensorResource(void *data, DALIDataType type,
                                 bool device, bool pinned, int device_id,
                                 const TensorShape<> &shape,
                                 const TensorShape<> &strides = {}) {
  if (!strides.empty() && strides.size() != shape.size())
    throw std::invalid_argument("If `strides` are not empty they must have the same number "
                                "of elements as `shape`.");
  auto rsrc = DLTensorResource<TensorViewPayload>::Create(shape, strides);
  detail::InitResourceDLTensor(*rsrc, data, type, device, pinned, device_id);
  return rsrc;
}

/** Constructs a DLManagedTensor WITHOUT data ownership. */
inline DLMTensorPtr MakeDLTensor(void *data, DALIDataType type,
                                 bool device, bool pinned, int device_id,
                                 const TensorShape<> &shape,
                                 const TensorShape<> &strides = {}) {
  return ToDLMTensor(MakeDLTensorResource(data, type, device, pinned, device_id, shape, strides));
}

/** Constructs a DLTensorResource sharing the data ownership. */
inline auto MakeDLTensorResource(std::shared_ptr<void> data, DALIDataType type,
                                 bool device, bool pinned, int device_id,
                                 const TensorShape<> &shape,
                                 const TensorShape<> &strides = {}) {
  if (!strides.empty() && strides.size() != shape.size())
    throw std::invalid_argument("If `strides` are not empty they must have the same number "
                                "of elements as `shape`.");
  auto rsrc = DLTensorResource<SharedTensorPayload>::Create(shape, strides, std::move(data));
  detail::InitResourceDLTensor(*rsrc, rsrc->payload.data.get(), type, device, pinned, device_id);
  return rsrc;
}

/** Constructs a DLManagedTensor sharing the data ownership. */
inline DLMTensorPtr MakeDLTensor(std::shared_ptr<void> data, DALIDataType type,
                                 bool device, bool pinned, int device_id,
                                 const TensorShape<> &shape,
                                 const TensorShape<> &strides = {}) {
  return ToDLMTensor(MakeDLTensorResource(
    std::move(data), type, device, pinned, device_id, shape, strides));
}

/** Gets a DLManagedTensor which does not hold a reference on the data.
 *
 * This function constructs a DLTensor whose context manager stores only the shape data.
 * The returned DLPack tensor must not outlive the original Tensor.
 */
template <typename Backend>
DLMTensorPtr GetDLTensorView(const SampleView<Backend> &tensor, bool pinned, int device_id) {
  auto rsrc = MakeDLTensorResource(
                  tensor.raw_mutable_data(), tensor.type(),
                  std::is_same_v<Backend, GPUBackend>, pinned, device_id,
                  tensor.shape());
  return ToDLMTensor(std::move(rsrc));
}


/** Gets a list of DLManagedTensors which do not hold a reference on the data.
 *
 * This function constructs a list of DLTensors whose context managers store only the shape data.
 * The returned DLPack tensors must not outlive the original TensorList.
 */
template <typename Backend>
std::vector<DLMTensorPtr> GetDLTensorListView(TensorList<Backend> &tensor_list) {
  int device_id = tensor_list.device_id();
  bool pinned = tensor_list.is_pinned();

  std::vector<DLMTensorPtr> dl_tensors{};
  dl_tensors.reserve(tensor_list.num_samples());

  for (int i = 0; i < tensor_list.num_samples(); ++i)
    dl_tensors.push_back(GetDLTensorView(tensor_list[i], pinned, device_id));
  return dl_tensors;
}


/** Gets a DLManagedTensor which shares the buffer ownership with a tensor.
 *
 * This function constructs a DLTensor whose context manager stores a shared pointer to the
 * tensor contents.
 * It can be used to remove data ownership from DALI to an external library.
 */
template <typename Backend>
DLMTensorPtr GetSharedDLTensor(Tensor<Backend> &tensor) {
  auto rsrc = MakeDLTensorResource(
                  tensor.get_data_ptr(), tensor.type(),
                  std::is_same_v<Backend, GPUBackend>, tensor.is_pinned(), tensor.device_id(),
                  tensor.shape());
  return ToDLMTensor(std::move(rsrc));
}

/** Gets a DLManagedTensor which shares the buffer ownership with a tensor.
 *
 * This function constructs a DLTensor whose context manager stores a shared pointer to the
 * tensor contents.
 * It can be used to remove data ownership from DALI to an external library.
 */
template <typename Backend>
DLMTensorPtr GetSharedDLTensor(const SampleView<Backend> &tensor,
                               std::shared_ptr<void> data, bool pinned, int device_id) {
  assert(tensor.raw_mutable_data() == data.get());
  auto rsrc = MakeDLTensorResource(
                  std::move(data), tensor.type(),
                  std::is_same_v<Backend, GPUBackend>, pinned, device_id,
                  tensor.shape());
  return ToDLMTensor(std::move(rsrc));
}


/** Gets a vector of DLManagedTensors which share the buffer ownership with a TensorList.
 *
 * This function constructs a list DLTensor whose context managers store shared pointers to the
 * samples in the TensorList.
 * It can be used to remove data ownership from DALI to an external library.
 */
template <typename Backend>
std::vector<DLMTensorPtr> GetSharedDLTensorList(TensorList<Backend> &tensor_list) {
  int device_id = tensor_list.device_id();
  bool pinned = tensor_list.is_pinned();

  std::vector<DLMTensorPtr> dl_tensors{};
  dl_tensors.reserve(tensor_list.num_samples());

  for (int i = 0; i < tensor_list.num_samples(); ++i)
    dl_tensors.push_back(GetSharedDLTensor(
        tensor_list[i],
        unsafe_sample_owner(tensor_list, i),
        pinned,
        device_id));
  return dl_tensors;
}


}  // namespace dali
#endif  // DALI_PIPELINE_DATA_DLTENSOR_H_
