// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include "third_party/dlpack/dlpack.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

DLL_PUBLIC void NonOwningDlTensorDestructor(DLManagedTensor *self);

DLL_PUBLIC void DLMTensorPtrDeleter(DLManagedTensor *ptr);

using DLMTensorPtr = std::unique_ptr<DLManagedTensor, void(*)(DLManagedTensor*)>;

DLL_PUBLIC DLDataType GetDLType(const TypeInfo &type);

namespace detail {

template <typename Backend>
DLMTensorPtr MakeDLTensor(void *data, const TypeInfo &type,
                          const kernels::TensorShape<> &shape, int device_id) {
  DLTensor dl_tensor{};
  dl_tensor.data = data;
  dl_tensor.ndim = shape.size();
  dl_tensor.shape = new int64_t[shape.size()];
  std::copy(shape.begin(), shape.end(), dl_tensor.shape);
  if (std::is_same<Backend, GPUBackend>::value) {
    dl_tensor.ctx = {kDLGPU, device_id};
  } else {
    dl_tensor.ctx = {kDLCPU, 0};
  }
  dl_tensor.dtype = GetDLType(type);
  return {new DLManagedTensor{dl_tensor, nullptr, &NonOwningDlTensorDestructor},
          &DLMTensorPtrDeleter};
}

}  // namespace detail

template <typename Backend>
DLMTensorPtr GetDLTensor(Tensor<Backend> &tensor) {
  return detail::MakeDLTensor<Backend>(tensor.raw_mutable_data(), tensor.type(),
                                       tensor.shape(), tensor.device_id());
}

template <typename Backend>
std::vector<DLMTensorPtr> GetDLTensors(TensorList<Backend> &tensor_list) {
  std::vector<DLMTensorPtr> dl_tensors{};
  dl_tensors.reserve(tensor_list.ntensor());
  for (size_t i = 0; i < tensor_list.ntensor(); ++i) {
    dl_tensors.push_back(detail::MakeDLTensor<Backend>(tensor_list.raw_mutable_tensor(i),
                                                  tensor_list.type(),
                                                  tensor_list.tensor_shape(i),
                                                  tensor_list.device_id()));
  }
  return dl_tensors;
}

}  // namespace dali
#endif  // DALI_PIPELINE_DATA_DLTENSOR_H_
