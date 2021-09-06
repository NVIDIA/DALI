// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_COPY_TO_EXTERNAL_H_
#define DALI_PIPELINE_DATA_COPY_TO_EXTERNAL_H_

#include <cuda_runtime.h>
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/types.h"
#include "dali/core/mm/memory_kind.h"

namespace dali {

namespace detail {

template <typename MemoryKind>
struct kind2backend {
  using type = CPUBackend;
};

template <>
struct kind2backend<mm::memory_kind::device> {
  using type = GPUBackend;
};

template <>
struct kind2backend<mm::memory_kind::managed> {
  using type = GPUBackend;
};

}  // namespace detail

template <typename DstBackend, typename SrcBackend>
inline void CopyToExternalImpl(void* dst,
                               const Buffer<SrcBackend> &src,
                               cudaStream_t stream, bool use_copy_kernel) {
  DeviceGuard d(src.device_id());
  const auto &type_info = src.type();
  type_info.template Copy<DstBackend, SrcBackend>(dst, src.raw_data(), src.size(), stream,
                                                  use_copy_kernel);
}

template <typename DstBackend, typename SrcBackend>
inline void CopyToExternalImpl(void** dsts,
                               const TensorList<SrcBackend> &src,
                               cudaStream_t stream, bool use_copy_kernel) {
  DeviceGuard d(src.device_id());

  const auto &type_info = src.type();
  auto src_shape = src.shape();

  SmallVector<int64_t, 256> sizes;
  int nsamples = src_shape.size();
  sizes.reserve(nsamples);
  for (int i = 0; i < nsamples; i++) {
    if (dsts[i] == nullptr)
      continue;
    sizes.push_back(src_shape.tensor_size(i));
  }
  int samples_to_copy = sizes.size();

  if (samples_to_copy < nsamples) {
    SmallVector<const void *, 256> from;
    from.reserve(samples_to_copy);
    SmallVector<void *, 256> to;
    to.reserve(samples_to_copy);
    for (int i = 0; i < nsamples; i++) {
      if (dsts[i] == nullptr)
        continue;
      from.push_back(src.raw_tensor(i));
      to.push_back(dsts[i]);
    }

    type_info.template Copy<DstBackend, SrcBackend>(to.data(), from.data(), sizes.data(),
                                                    samples_to_copy, stream, use_copy_kernel);
  } else {
    type_info.template Copy<DstBackend, SrcBackend>(dsts, src.raw_data(), sizes.data(), nsamples,
                                                    stream, use_copy_kernel);
  }
}

template <typename DstKind, typename SrcBackend>
inline void CopyToExternal(void* dst, const Buffer<SrcBackend> &src,
                           cudaStream_t stream, bool use_copy_kernel) {
  bool src_device_access = (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned());
  bool dst_device_access = cuda::kind_has_property<DstKind, cuda::memory_access::device>::value;
  use_copy_kernel &= dst_device_access && src_device_access;
  using DstBackend = typename detail::kind2backend<DstKind>::type;
  CopyToExternalImpl<DstBackend, SrcBackend>(dst, src, stream, use_copy_kernel);
}

/**
 * @brief Run-time dispatch - DO NOT USE unless really necessary
 */
template <typename SrcBackend>
inline void CopyToExternal(void* dst, mm::memory_kind_id kind_id, const Buffer<SrcBackend> &src,
                           cudaStream_t stream, bool use_copy_kernel) {
  TYPE_SWITCH(kind_id, mm::kind2id, Kind, (mm::memory_kind::host,
                                           mm::memory_kind::pinned,
                                           mm::memory_kind::device,
                                           mm::memory_kind::managed),
    (CopyToExternal<Kind, SrcBackend>(dst, src, stream, use_copy_kernel)),
    (throw std::logic_error("Unreachable code - invalid memory kind.")));
}

template <typename DstKind, typename SrcBackend>
inline void CopyToExternal(void** dsts, const TensorList<SrcBackend> &src,
                           cudaStream_t stream, bool use_copy_kernel) {
  bool src_device_access = (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned());
  bool dst_device_access = cuda::kind_has_property<DstKind, cuda::memory_access::device>::value;
  use_copy_kernel &= dst_device_access && src_device_access;
  using DstBackend = typename detail::kind2backend<DstKind>::type;
  CopyToExternalImpl<DstBackend, SrcBackend>(dsts, src, stream, use_copy_kernel);
}

/**
 * @brief Run-time dispatch - DO NOT USE unless really necessary
 */
template <typename SrcBackend>
inline void CopyToExternal(void** dsts, mm::memory_kind_id kind_id,
                           const TensorList<SrcBackend> &src,
                           cudaStream_t stream, bool use_copy_kernel) {
  TYPE_SWITCH(kind_id, mm::kind2id, Kind, (mm::memory_kind::host,
                                           mm::memory_kind::pinned,
                                           mm::memory_kind::device,
                                           mm::memory_kind::managed),
    (CopyToExternal<Kind, SrcBackend>(dsts, src, stream, use_copy_kernel)),
    (throw std::logic_error("Unreachable code - invalid memory kind.")));
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_COPY_TO_EXTERNAL_H_
