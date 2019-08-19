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

#include "dali/pipeline/data/dltensor.h"

namespace dali {

DLDataType GetDLType(const TypeInfo &type) {
  DLDataType dl_type{};
  DALI_TYPE_SWITCH(type.id(), T,
      dl_type.bits = sizeof(T) * 8;
      dl_type.lanes = 1;
      if (std::is_floating_point<T>::value) {
        dl_type.code = kDLFloat;
      } else if (std::is_unsigned<T>::value) {
        dl_type.code = kDLUInt;
      } else if (std::is_integral<T>::value) {
        dl_type.code = kDLInt;
      } else {
        DALI_FAIL("This data type (" + type.name() + ") cannot be handled by DLTensor.");
      })
  return dl_type;
}

void DLManagedTensorDeleter(DLManagedTensor *self) {
  delete static_cast<DLTensorResource*>(self->manager_ctx);
}

void DLMTensorPtrDeleter(DLManagedTensor* dlm_tensor_ptr) {
  if (dlm_tensor_ptr) {
    dlm_tensor_ptr->deleter(dlm_tensor_ptr);
    delete dlm_tensor_ptr;
  }
}

DLMTensorPtr MakeDLTensor(void* data, const TypeInfo& type,
                          bool device, int device_id,
                          std::unique_ptr<DLTensorResource> resource) {
  DLTensor dl_tensor{};
  dl_tensor.data = data;
  dl_tensor.ndim = resource->shape.size();
  dl_tensor.shape = resource->shape.begin();
  if (device) {
    dl_tensor.ctx = {kDLGPU, device_id};
  } else {
    dl_tensor.ctx = {kDLCPU, 0};
  }
  dl_tensor.dtype = GetDLType(type);
  return {new DLManagedTensor{dl_tensor, resource.release(), &DLManagedTensorDeleter},
          &DLMTensorPtrDeleter};
}

}  // namespace dali
