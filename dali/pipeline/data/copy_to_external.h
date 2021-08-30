// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/kernels/alloc_type.h"
#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/types.h"

namespace dali {

using kernels::AllocType;

template <AllocType alloc_type>
struct alloc_to_backend {
  using type = CPUBackend;
};

template <>
struct alloc_to_backend<AllocType::GPU> {
  using type = GPUBackend;
};

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
inline void CopyToExternal(void* dst, AllocType dst_alloc_type,
                           const Buffer<SrcBackend> &src,
                           cudaStream_t stream, bool use_copy_kernel) {
  VALUE_SWITCH(dst_alloc_type, DstType, (AllocType::Host, AllocType::Pinned, AllocType::GPU), (
    use_copy_kernel &= (DstType == AllocType::GPU || DstType == AllocType::Pinned) &&
                       (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned());
    using DstBackend = alloc_to_backend<DstType>::type;
    CopyToExternalImpl<DstBackend, SrcBackend>(dst, src, stream, use_copy_kernel);
  ), ());  // NOLINT
}

template <typename SrcBackend>
inline void CopyToExternal(void** dsts, AllocType dst_alloc_type,
                           const TensorList<SrcBackend> &src,
                           cudaStream_t stream, bool use_copy_kernel) {
  VALUE_SWITCH(dst_alloc_type, DstType, (AllocType::Host, AllocType::Pinned, AllocType::GPU), (
    use_copy_kernel &= (DstType == AllocType::GPU || DstType == AllocType::Pinned) &&
                       (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned());
    using DstBackend = alloc_to_backend<DstType>::type;
    CopyToExternalImpl<DstBackend, SrcBackend>(dsts, src, stream, use_copy_kernel);
  ), ());  // NOLINT
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_COPY_TO_EXTERNAL_H_
