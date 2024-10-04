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

#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "third_party/dlpack/include/dlpack/dlpack.h"

#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor_list.h"

namespace dali {

using DLMTensorPtr = std::unique_ptr<DLManagedTensor, void(*)(DLManagedTensor*)>;

DLL_PUBLIC DLDataType GetDLType(DALIDataType type);

struct ShapeAndStride {
  TensorShape<> shape, stride;
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
 */
template <typename Payload = ShapeAndStride>
struct DLTensorResource {
  DLManagedTensor dlm_tensor{};
  Payload payload{};

  template <typename... PayloadArgs>
  DLMTensorPtr Create(const DLTensor &tensor, PayloadArgs &&...args) {
    auto rsrc = std::make_unique<DLTensorResource>(
      tensor, std::forward<PayloadArgs>(args)...);
    return { rsrc.release(), uptr_deleter };
  }

 protected:
  template <typename PayloadArgs>
  explicit DLTensorResource(DLTensor tensor, PayloadArgs &&...args)
  : dlm_tensor{tensor, this, dlm_deleter}
  , payload(std::forward<PayloadArgs>(args)...) {}

  static void uptr_deleter(DLManagedTensor *tensor) {
    if (tensor && tensor->deleter) {
      tensor->deleter(tensor);
    }
  }

  static void dlm_deleter(DLManagedTensor *tensor) {
    if (tensor == nullptr)
      return;
    auto *This = static_cast<DLTensorResource *>(tensor->manager_ctx);
    assert(&This->dlm_tensor == tensor);  // is that always the case?
    delete This;
  }
};


DLL_PUBLIC DLTensor PopulateDLTensor(void *data, DALIDataType type,
                                     std::optional<int> device_id,
                                     span<int64_t> shape,
                                     span<int64_t> strides = {});

template <int ndim>
DLTensor PopulateDLTensor(void *data, DALIDataType type,
                          std::optional<int> device_id,
                          TensorShape<ndim> shape) {
  return PopulateDLTensor(data, type, device_id, make_span(shape));
}

template <typename Backend>
DLTensor PopulateDLTensor(SampleView<Backend> tensor, int device_id) {
  return PopulateDLTensor(tensor.raw_mutable_data(),
                          tensor.type(),
                          std::is_same<Backend, GPUBackend>::value ? device_id : std::nullopt,
                          make_span(tensor.shape));
}

template <typename Backend>
DLMTensorPtr GetDLTensorView(SampleView<Backend> tensor, int device_id) {
  return DLTensorResource<>::Create(
      PopulateDLTensor(tensor.raw_mutable_data(),
                       tensor.type(),
                       std::is_same<Backend, GPUBackend>::value,
                       tensor.shape(),
                       device_id), tensor.shape);
}

template <typename Backend>
std::vector<DLMTensorPtr> GetDLTensorListView(TensorList<Backend> &tensor_list) {
  std::optional<int> device_id = std::is_same<Backend, GPUBackend>::value
                               ? tensor_list.device_id()
                               : std::nullopt;

  std::vector<DLMTensorPtr> dl_tensors{};
  dl_tensors.reserve(tensor_list.num_samples());

  for (int i = 0; i < tensor_list.num_samples(); ++i) {
    const auto &shape = tensor_list.tensor_shape(i);
    dl_tensors.push_back(DLTensorResource<>::Create(
        MakeDLTensor(tensor_list.raw_mutable_tensor(i),
                     tensor_list.type(),
                     device_id,
                     shape)));
  }
  return dl_tensors;
}

DLL_PUBLIC DALIDataType DLToDALIType(const DLDataType &dl_type);

}  // namespace dali
#endif  // DALI_PIPELINE_DATA_DLTENSOR_H_
