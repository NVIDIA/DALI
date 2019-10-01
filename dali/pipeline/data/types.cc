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


#define DALI_TYPENAME_REGISTERER(TypeString) \
{                                            \
  return TypeString;                         \
}

#define DALI_TYPEID_REGISTERER(Type, dtype)                                      \
{                                                                                \
  static DALIDataType type_id = TypeTable::instance().RegisterType<Type>(dtype); \
  return type_id;                                                                \
}

#define DALI_REGISTER_TYPE_IMPL(Type, Name, Id) \
const auto &_type_info_##Id = TypeTable::GetTypeID<Type>()

#include "dali/pipeline/data/types.h"
#include "dali/util/half.hpp"

#include "dali/pipeline/data/backend.h"

namespace dali {

TypeTable &TypeTable::instance() {
  static TypeTable singleton;
  return singleton;
}

template <>
void TypeInfo::Copy<CPUBackend, CPUBackend>(void *dst,
    const void *src, Index n, cudaStream_t /* unused */) {
  // Call our copy function
  copier_(dst, src, n);
}

// For any GPU related copy, we do a plain memcpy
template <>
void TypeInfo::Copy<CPUBackend, GPUBackend>(void *dst,
    const void *src, Index n, cudaStream_t stream) {
  MemCopy(dst, src, n*size(), stream);
}

template <>
void TypeInfo::Copy<GPUBackend, CPUBackend>(void *dst,
    const void *src, Index n, cudaStream_t stream) {
  MemCopy(dst, src, n*size(), stream);
}

template <>
void TypeInfo::Copy<GPUBackend, GPUBackend>(void *dst,
    const void *src, Index n, cudaStream_t stream) {
  MemCopy(dst, src, n*size(), stream);
}


}  // namespace dali
