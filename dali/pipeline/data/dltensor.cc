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

#include "dali/pipeline/data/dltensor.h"

namespace dali {

DLDataType GetDLType(const TypeInfo &type) {
  DLDataType dl_type{};
  DALI_TYPE_SWITCH(type.id(), T,
      dl_type.bits = sizeof(T) * 8;
      dl_type.lanes = 1;
      if (std::is_floating_point<T>::value) {
        dl_type.code = 2U;
      } else if (std::is_unsigned<T>::value) {
        dl_type.code = 1U;
      } else if (std::is_integral<T>::value) {
        dl_type.code = 0U;
      } else {
        DALI_FAIL("This data type (" + type.name() + ") cannot be handled by DLTensor.");
      })
  return dl_type;
}

void NonOwningDlTensorDestructor(DLManagedTensor *self) {
  delete self->dl_tensor.shape;
  delete self->dl_tensor.strides;
}

void DLMTensorPtrDeleter(DLManagedTensor* dlm_tensor_ptr) {
  if (dlm_tensor_ptr) {
    dlm_tensor_ptr->deleter(dlm_tensor_ptr);
    delete dlm_tensor_ptr;
  }
}

}  // namespace dali
