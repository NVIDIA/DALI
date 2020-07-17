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

#include <memory>

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

namespace detail {

class ScatterGatherPool {
 public:
  static void Copy(void *dst, const void **srcs, const Index *sizes, int n, int element_size,
                   cudaStream_t stream);

 private:
  static ScatterGatherPool& instance();

  // ScatterGatherPool should only be referenced through its static members
  ScatterGatherPool() {}

  spinlock lock_;
  std::unordered_map<cudaStream_t, std::unique_ptr<kernels::ScatterGatherGPU>>
      scatter_gather_instances_;
};

ScatterGatherPool& ScatterGatherPool::instance() {
  static ScatterGatherPool singleton;
  return singleton;
}

void ScatterGatherPool::Copy(void *dst, const void **srcs, const Index *sizes, int n,
                             int element_size, cudaStream_t stream) {
  auto &inst = instance();
  std::lock_guard<spinlock> guard(inst.lock_);
  auto& scatter_gather = inst.scatter_gather_instances_[stream];
  constexpr int kBlockSize = 1 << 19;  // 512kB per block
  if (!scatter_gather)
    scatter_gather.reset(new kernels::ScatterGatherGPU(kBlockSize, n));

  ptrdiff_t offset = 0;
  for (int i = 0; i < n; i++) {
    auto nbytes = sizes[i] * element_size;
    scatter_gather->AddCopy(reinterpret_cast<uint8_t*>(dst) + offset, srcs[i], nbytes);
    offset += nbytes;
  }
  scatter_gather->Run(stream);
}

}  // namespace detail

TypeTable &TypeTable::instance() {
  static TypeTable singleton;
  return singleton;
}

template <typename DstBackend, typename SrcBackend>
void TypeInfo::Copy(void *dst,
    const void *src, Index n, cudaStream_t stream, bool use_copy_kernel) const {
  constexpr bool is_src_to_src = std::is_same<DstBackend, CPUBackend>::value &&
                                 std::is_same<SrcBackend, CPUBackend>::value;
  if (is_src_to_src) {
    // Call our copy function
    copier_(dst, src, n);
  } else if (use_copy_kernel) {
    detail::LaunchCopyKernel(dst, src, n * size(), stream);
  } else {
    MemCopy(dst, src, n*size(), stream);
  }
}

template void TypeInfo::Copy<CPUBackend, CPUBackend>(void *dst,
    const void *src, Index n, cudaStream_t stream, bool use_copy_kernel) const;

template void TypeInfo::Copy<CPUBackend, GPUBackend>(void *dst,
    const void *src, Index n, cudaStream_t stream, bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, CPUBackend>(void *dst,
    const void *src, Index n, cudaStream_t stream, bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, GPUBackend>(void *dst,
    const void *src, Index n, cudaStream_t stream, bool use_copy_kernel) const;

template <typename DstBackend, typename SrcBackend>
void TypeInfo::Copy(void *dst, const void** srcs, const Index* sizes, int n,
                    cudaStream_t stream, bool use_copy_kernel) const {
  constexpr bool is_src_to_src = std::is_same<DstBackend, CPUBackend>::value &&
                                 std::is_same<SrcBackend, CPUBackend>::value;
  if (!is_src_to_src && use_copy_kernel) {
    detail::ScatterGatherPool::Copy(dst, srcs, sizes, n, size(), stream);
  } else {
    uint8_t *sample_dst = reinterpret_cast<uint8_t*>(dst);
    for (int i = 0; i < n; i++) {
      Copy<DstBackend, SrcBackend>(sample_dst, srcs[i], sizes[i], stream);
      sample_dst += sizes[i] * size();
    }
  }
}

template void TypeInfo::Copy<CPUBackend, CPUBackend>(void *dst,
    const void **src, const Index *sizes, int n, cudaStream_t stream, bool use_copy_kernel) const;

template void TypeInfo::Copy<CPUBackend, GPUBackend>(void *dst,
    const void **src, const Index *sizes, int n, cudaStream_t stream, bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, CPUBackend>(void *dst,
    const void **src, const Index *sizes, int n, cudaStream_t stream, bool use_copy_kernel) const;

template void TypeInfo::Copy<GPUBackend, GPUBackend>(void *dst,
    const void **src, const Index *sizes, int n, cudaStream_t stream, bool use_copy_kernel) const;

}  // namespace dali
