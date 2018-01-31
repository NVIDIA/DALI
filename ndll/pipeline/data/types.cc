// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/data/types.h"

#include "ndll/pipeline/data/backend.h"

namespace ndll {
std::mutex TypeTable::mutex_;
std::unordered_map<std::type_index, NDLLDataType> TypeTable::type_map_;

template <>
void TypeInfo::Construct<CPUBackend>(void *ptr, Index n) {
  // Call our constructor function
  constructor_(ptr, n);
}

template <>
void TypeInfo::Construct<GPUBackend>(void *ptr, Index n) {
  // NoOp. GPU types must not require constructor
}

template <>
void TypeInfo::Destruct<CPUBackend>(void *ptr, Index n) {
  // Call our destructor function
  destructor_(ptr, n);
}

template <>
void TypeInfo::Destruct<GPUBackend>(void *ptr, Index n) {
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

// Instantiate some basic types
NDLL_REGISTER_TYPE(NoType, NDLL_NO_TYPE);
NDLL_REGISTER_TYPE(uint8, NDLL_UINT8);
NDLL_REGISTER_TYPE(int16, NDLL_INT16);
NDLL_REGISTER_TYPE(int32, NDLL_INT32);
NDLL_REGISTER_TYPE(int64, NDLL_INT64);
//NDLL_REGISTER_TYPE(long long);  // NOLINT
NDLL_REGISTER_TYPE(float16, NDLL_FLOAT16);
NDLL_REGISTER_TYPE(float, NDLL_FLOAT);
NDLL_REGISTER_TYPE(double, NDLL_FLOAT64);
NDLL_REGISTER_TYPE(bool, NDLL_BOOL);

}  // namespace ndll
