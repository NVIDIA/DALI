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

#ifndef DALI_PIPELINE_DATA_DLTENSOR_H_
#define DALI_PIPELINE_DATA_DLTENSOR_H_

#include <memory>
#include <vector>
#include <utility>
#include "third_party/dlpack/include/dlpack/dlpack.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

DLL_PUBLIC void DLManagedTensorDeleter(DLManagedTensor *self);

DLL_PUBLIC void DLMTensorPtrDeleter(DLManagedTensor *ptr);

using DLMTensorPtr = std::unique_ptr<DLManagedTensor, void(*)(DLManagedTensor*)>;

DLL_PUBLIC DLDataType GetDLType(const TypeInfo &type);

struct DLTensorResource {
  explicit DLTensorResource(TensorShape<> shape)
  : shape(std::move(shape))
  , strides() {}

  TensorShape<> shape;
  TensorShape<> strides;
  DLManagedTensor dlm_tensor{};

  virtual ~DLTensorResource() = default;
};

DLL_PUBLIC DLMTensorPtr MakeDLTensor(void *data, const TypeInfo &type,
                                     bool device, int device_id,
                                     std::unique_ptr<DLTensorResource> resource);

template <typename Backend>
DLMTensorPtr GetDLTensorView(Tensor<Backend> &tensor) {
  return MakeDLTensor(tensor.raw_mutable_data(),
                      tensor.type(),
                      std::is_same<Backend, GPUBackend>::value,
                      tensor.device_id(),
                      std::make_unique<DLTensorResource>(tensor.shape()));
}

template <typename Backend>
std::vector<DLMTensorPtr> GetDLTensorListView(TensorList<Backend> &tensor_list) {
  std::vector<DLMTensorPtr> dl_tensors{};
  dl_tensors.reserve(tensor_list.ntensor());
  for (size_t i = 0; i < tensor_list.ntensor(); ++i) {
    const auto &shape = tensor_list.tensor_shape(i);
    dl_tensors.push_back(MakeDLTensor(tensor_list.raw_mutable_tensor(i),
                                      tensor_list.type(),
                                      std::is_same<Backend, GPUBackend>::value,
                                      tensor_list.device_id(),
                                      std::make_unique<DLTensorResource>(shape)));
  }
  return dl_tensors;
}

DLL_PUBLIC DALIDataType DLToDALIType(const DLDataType &dl_type);

}  // namespace dali
#endif  // DALI_PIPELINE_DATA_DLTENSOR_H_
