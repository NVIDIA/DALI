// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#define NDLL_TYPENAME_REGISTERER(Type) \
{                                      \
  return #Type;                        \
}

#define NDLL_TYPEID_REGISTERER(Type, dtype)                           \
{                                                                     \
  std::lock_guard<std::mutex> lock(mutex_);                           \
  static NDLLDataType type_id = TypeTable::RegisterType<Type>(dtype); \
  return type_id;                                                     \
}


#include "ndll/pipeline/data/types.h"

#include "ndll/pipeline/data/backend.h"

namespace ndll {
std::mutex TypeTable::mutex_;
std::unordered_map<std::type_index, NDLLDataType> TypeTable::type_map_;
int TypeTable::index_ = NDLL_DATATYPE_END;

template <>
void TypeInfo::Construct<CPUBackend>(void *ptr, Index n) {
  // Call our constructor function
  constructor_(ptr, n);
}

template <>
void TypeInfo::Construct<GPUBackend>(void *, Index) {
  // NoOp. GPU types must not require constructor
}

template <>
void TypeInfo::Destruct<CPUBackend>(void *ptr, Index n) {
  // Call our destructor function
  destructor_(ptr, n);
}

template <>
void TypeInfo::Destruct<GPUBackend>(void *, Index) {
  // NoOp. GPU types must not require destructor
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


}  // namespace ndll
