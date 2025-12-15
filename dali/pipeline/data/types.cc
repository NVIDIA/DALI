// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define DALI_TYPENAME_REGISTERER(Type, dtype)                                    \
{                                                                                \
  return dali::TypeName(dtype);                                                  \
}

#define DALI_TYPEID_REGISTERER(Type, dtype)                                      \
{                                                                                \
  static DALIDataType type_id = TypeTable::instance().RegisterType<Type>(dtype); \
  return type_id;                                                                \
}

#define DALI_REGISTER_TYPE_IMPL(Type, Id) \
const auto &_type_info_##Id = TypeTable::GetTypeId<Type>()

#include "dali/pipeline/data/types.h"
#include "dali/core/float16.h"
#include "dali/pipeline/data/backend.h"
#include "dali/core/per_stream_pool.h"
#include "dali/kernels/common/scatter_gather.h"

namespace dali {

namespace detail {

static constexpr size_t kMaxSizePerBlock = 1 << 18;  // 256 kB per block
using ScatterGatherPool = PerStreamPool<kernels::ScatterGatherGPU, spinlock, true>;
ScatterGatherPool& ScatterGatherPoolInstance() {
  static ScatterGatherPool scatter_gather_pool_;
  return scatter_gather_pool_;
}

void ScatterGatherCopy(void **dsts, const void **srcs, const Index *sizes, int n, int element_size,
                       cudaStream_t stream) {
  auto sc = ScatterGatherPoolInstance().Get(stream, kMaxSizePerBlock);
  for (int i = 0; i < n; i++) {
    sc->AddCopy(dsts[i], srcs[i], sizes[i] * element_size);
  }
  sc->Run(stream, true, kernels::ScatterGatherGPU::Method::Kernel);
}

void ScatterGatherCopy(void *dst, const void **srcs, const Index *sizes, int n, int element_size,
                       cudaStream_t stream) {
  auto sc = ScatterGatherPoolInstance().Get(stream, kMaxSizePerBlock);
  auto *sample_dst = reinterpret_cast<uint8_t*>(dst);
  for (int i = 0; i < n; i++) {
    auto nbytes = sizes[i] * element_size;
    sc->AddCopy(sample_dst, srcs[i], nbytes);
    sample_dst += nbytes;
  }
  sc->Run(stream, true, kernels::ScatterGatherGPU::Method::Kernel);
}

void ScatterGatherCopy(void **dsts, const void *src, const Index *sizes, int n, int element_size,
                       cudaStream_t stream) {
  auto sc = ScatterGatherPoolInstance().Get(stream, kMaxSizePerBlock);
  auto *sample_src = reinterpret_cast<const uint8_t*>(src);
  for (int i = 0; i < n; i++) {
    auto nbytes = sizes[i] * element_size;
    sc->AddCopy(dsts[i], sample_src, nbytes);
    sample_src += nbytes;
  }
  sc->Run(stream, true, kernels::ScatterGatherGPU::Method::Kernel);
}

}  // namespace detail

TypeTable &TypeTable::instance() {
  static TypeTable singleton;
  return singleton;
}

template <typename DstBackend, typename SrcBackend>
void TypeInfo::Copy(
      void *dst,
      std::optional<int> dst_dev_id,
      const void *src,
      std::optional<int> src_dev_id,
      Index n,
      cudaStream_t stream,
      bool use_copy_kernel) const {
  constexpr bool is_host_to_host = std::is_same<DstBackend, CPUBackend>::value &&
                                   std::is_same<SrcBackend, CPUBackend>::value;
  if (n == 0)
    return;

  bool cross_device = src_dev_id && dst_dev_id && *src_dev_id != *dst_dev_id;

  if (is_host_to_host) {
    // We don't honor stream order for H2H copies so we error out if a stream is passed
    if (stream)
      throw std::logic_error("Cannot issue a H2H copy on a stream");

    // Call our copy function
    copier_(dst, src, n);
  } else if (use_copy_kernel && !cross_device) {
    detail::LaunchCopyKernel(dst, src, n * size(), stream);
  } else {
    if (cross_device) {
      CUDA_CALL(cudaMemcpyPeerAsync(dst, *dst_dev_id, src, *src_dev_id, n*size(), stream));
    } else {
      MemCopy(dst, src, n*size(), stream);
    }
  }
}

template void TypeInfo::Copy<CPUBackend, CPUBackend>(
    void *dst,
    std::optional<int> dst_dev_id,
    const void *src,
    std::optional<int> src_dev_id,
    Index n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<CPUBackend, GPUBackend>(
    void *dst,
    std::optional<int> dst_dev_id,
    const void *src,
    std::optional<int> src_dev_id,
    Index n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, CPUBackend>(
    void *dst,
    std::optional<int> dst_dev_id,
    const void *src,
    std::optional<int> src_dev_id,
    Index n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, GPUBackend>(
    void *dst,
    std::optional<int> dst_dev_id,
    const void *src,
    std::optional<int> src_dev_id,
    Index n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template <typename DstBackend, typename SrcBackend>
void TypeInfo::Copy(
      void **dsts,
      std::optional<int> dst_dev_id,
      const void** srcs,
      std::optional<int> src_dev_id,
      const Index* sizes,
      int n,
      cudaStream_t stream,
      bool use_copy_kernel) const {
  constexpr bool is_host_to_host = std::is_same<DstBackend, CPUBackend>::value &&
                                   std::is_same<SrcBackend, CPUBackend>::value;
  // assume same device if either is not set
  bool cross_device = src_dev_id && dst_dev_id && *src_dev_id != *dst_dev_id;
  if (!is_host_to_host && use_copy_kernel && !cross_device) {
    detail::ScatterGatherCopy(dsts, srcs, sizes, n, size(), stream);
  } else {
    for (int i = 0; i < n; i++) {
      Copy<DstBackend, SrcBackend>(dsts[i], dst_dev_id, srcs[i], src_dev_id, sizes[i], stream);
    }
  }
}

template void TypeInfo::Copy<CPUBackend, CPUBackend>(
    void **dsts,
    std::optional<int> dst_dev_id,
    const void **src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<CPUBackend, GPUBackend>(
    void **dsts,
    std::optional<int> dst_dev_id,
    const void **src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, CPUBackend>(
    void **dsts,
    std::optional<int> dst_dev_id,
    const void **src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, GPUBackend>(
    void **dsts,
    std::optional<int> dst_dev_id,
    const void **src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;


template <typename DstBackend, typename SrcBackend>
void TypeInfo::Copy(
      void *dst,
      std::optional<int> dst_dev_id,
      const void** srcs,
      std::optional<int> src_dev_id,
      const Index* sizes,
      int n,
      cudaStream_t stream,
      bool use_copy_kernel) const {
  constexpr bool is_host_to_host = std::is_same<DstBackend, CPUBackend>::value &&
                                   std::is_same<SrcBackend, CPUBackend>::value;
  // assume same device if either is not set
  bool cross_device = src_dev_id && dst_dev_id && *src_dev_id != *dst_dev_id;
  if (!is_host_to_host && use_copy_kernel && !cross_device) {
    detail::ScatterGatherCopy(dst, srcs, sizes, n, size(), stream);
  } else {
    auto sample_dst = static_cast<uint8_t*>(dst);
    for (int i = 0; i < n; i++) {
      Copy<DstBackend, SrcBackend>(sample_dst, dst_dev_id, srcs[i], src_dev_id, sizes[i], stream);
      sample_dst += sizes[i] * size();
    }
  }
}


template void TypeInfo::Copy<CPUBackend, CPUBackend>(
    void *dst,
    std::optional<int> dst_dev_id,
    const void **src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<CPUBackend, GPUBackend>(
    void *dst,
    std::optional<int> dst_dev_id,
    const void **src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, CPUBackend>(
    void *dst,
    std::optional<int> dst_dev_id,
    const void **src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, GPUBackend>(
    void *dst,
    std::optional<int> dst_dev_id,
    const void **src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;


template <typename DstBackend, typename SrcBackend>
void TypeInfo::Copy(
      void **dsts,
      std::optional<int> dst_dev_id,
      const void* src,
      std::optional<int> src_dev_id,
      const Index* sizes,
      int n,
      cudaStream_t stream,
      bool use_copy_kernel) const {
  constexpr bool is_host_to_host = std::is_same<DstBackend, CPUBackend>::value &&
                                   std::is_same<SrcBackend, CPUBackend>::value;
  // assume same device if either is not set
  bool cross_device = src_dev_id && dst_dev_id && *src_dev_id != *dst_dev_id;
  if (!is_host_to_host && use_copy_kernel && !cross_device) {
    detail::ScatterGatherCopy(dsts, src, sizes, n, size(), stream);
  } else {
    auto sample_src = reinterpret_cast<const uint8_t*>(src);
    for (int i = 0; i < n; i++) {
      Copy<DstBackend, SrcBackend>(dsts[i], dst_dev_id, sample_src, src_dev_id, sizes[i], stream);
      sample_src += sizes[i] * size();
    }
  }
}

template void TypeInfo::Copy<CPUBackend, CPUBackend>(
    void **dsts,
    std::optional<int> dst_dev_id,
    const void *src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<CPUBackend, GPUBackend>(
    void **dsts,
    std::optional<int> dst_dev_id,
    const void *src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, CPUBackend>(
    void **dsts,
    std::optional<int> dst_dev_id,
    const void *src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, GPUBackend>(
    void **dsts,
    std::optional<int> dst_dev_id,
    const void *src,
    std::optional<int> src_dev_id,
    const Index *sizes,
    int n,
    cudaStream_t stream,
    bool use_copy_kernel) const;

}  // namespace dali
