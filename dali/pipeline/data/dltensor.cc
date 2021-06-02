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
#include <string>

namespace dali {

DLDataType GetDLType(const TypeInfo &type) {
  DLDataType dl_type{};
  DALI_TYPE_SWITCH_WITH_FP16(type.id(), T,
      dl_type.bits = sizeof(T) * 8;
      dl_type.lanes = 1;
      if (dali::is_fp_or_half<T>::value) {
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
  if (dlm_tensor_ptr && dlm_tensor_ptr->deleter) {
    dlm_tensor_ptr->deleter(dlm_tensor_ptr);
  }
}

DLMTensorPtr MakeDLTensor(void* data, const TypeInfo& type,
                          bool device, int device_id,
                          std::unique_ptr<DLTensorResource> resource) {
  DLManagedTensor *dlm_tensor_ptr = &resource->dlm_tensor;
  DLTensor &dl_tensor = dlm_tensor_ptr->dl_tensor;
  dl_tensor.data = data;
  dl_tensor.ndim = resource->shape.size();
  dl_tensor.shape = resource->shape.begin();
  if (!resource->strides.empty()) {
    dl_tensor.strides = resource->strides.data();
  }
  if (device) {
    dl_tensor.device = {kDLCUDA, device_id};
  } else {
    dl_tensor.device = {kDLCPU, 0};
  }
  dl_tensor.dtype = GetDLType(type);
  dlm_tensor_ptr->deleter = &DLManagedTensorDeleter;
  dlm_tensor_ptr->manager_ctx = resource.release();
  return {dlm_tensor_ptr, &DLMTensorPtrDeleter};
}

inline std::string to_string(const DLDataType &dl_type) {
  return std::string("{code: ")
    + (dl_type.code ? ((dl_type.code == 2) ? "kDLFloat" : "kDLUInt") : "kDLInt")
    + ", bits: " + std::to_string(dl_type.bits) + ", lanes: " + std::to_string(dl_type.lanes) + "}";
}

DALIDataType DLToDALIType(const DLDataType &dl_type) {
  DALI_ENFORCE(dl_type.lanes == 1,
               "DALI Tensors do no not support types with the number of lanes other than 1");
  switch (dl_type.code) {
    case kDLUInt: {
      switch (dl_type.bits) {
        case 8: return DALI_UINT8;
        case 16: return DALI_UINT16;
        case 32: return DALI_UINT32;
        case 64: return DALI_UINT64;
      }
      break;
    }
    case kDLInt: {
      switch (dl_type.bits) {
        case 8: return DALI_INT8;
        case 16: return DALI_INT16;
        case 32: return DALI_INT32;
        case 64: return DALI_INT64;
      }
      break;
    }
    case kDLFloat: {
      switch (dl_type.bits) {
        case 16: return DALI_FLOAT16;
        case 32: return DALI_FLOAT;
        case 64: return DALI_FLOAT64;
      }
      break;
    }
  }
  DALI_FAIL("Could not convert DLPack tensor of unsupported type " + to_string(dl_type));
}

}  // namespace dali
