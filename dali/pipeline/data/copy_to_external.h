// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                               const Tensor<SrcBackend> &src,
                               AccessOrder order, bool use_copy_kernel) {
  DeviceGuard d(src.device_id());
  const auto &type_info = src.type_info();
  type_info.template Copy<DstBackend, SrcBackend>(dst, src.raw_data(), src.size(), order.stream(),
                                                  use_copy_kernel);
}

template <typename DstBackend, typename SrcBackend>
inline void CopyToExternalImpl(void* dst,
                               const TensorList<SrcBackend> &src,
                               AccessOrder order, bool use_copy_kernel) {
  DeviceGuard d(src.device_id());
  const auto &type_info = src.type_info();

  // TODO(klecki): Add a proper test for non-contiguous access when we can have non-contiguous
  // data here.

  constexpr bool is_gpu_copy = std::is_same_v<DstBackend, GPUBackend> ||
                               std::is_same_v<SrcBackend, GPUBackend>;
  if constexpr (is_gpu_copy) {
    src.order().wait(order);
  }

  if (src.IsContiguous()) {
    type_info.template Copy<DstBackend, SrcBackend>(dst, unsafe_raw_data(src), src._num_elements(),
                                                    order.stream(), use_copy_kernel);
  } else {
    const auto &src_shape = src.shape();
    SmallVector<const void *, 256> from;
    SmallVector<int64_t, 256> sizes;
    int num_samples = src_shape.num_samples();
    from.reserve(num_samples);
    sizes.reserve(num_samples);
    for (int i = 0; i < num_samples; i++) {
      from.push_back(src.raw_tensor(i));
      sizes.push_back(src_shape.tensor_size(i));
    }

    type_info.template Copy<DstBackend, SrcBackend>(dst, from.data(), sizes.data(),
                                                    num_samples, order.stream(), use_copy_kernel);
  }
}

template <typename DstBackend, typename SrcBackend>
inline void CopyToExternalImpl(void** dsts,
                               const TensorList<SrcBackend> &src,
                               AccessOrder order, bool use_copy_kernel) {
  DeviceGuard d(src.device_id());

  constexpr bool is_gpu_copy = std::is_same_v<DstBackend, GPUBackend> ||
                               std::is_same_v<SrcBackend, GPUBackend>;
  if constexpr (is_gpu_copy) {
    src.order().wait(order);
  }

  const auto &type_info = src.type_info();
  const auto &src_shape = src.shape();

  SmallVector<int64_t, 256> sizes;
  int num_samples = src_shape.num_samples();
  sizes.reserve(num_samples);
  for (int i = 0; i < num_samples; i++) {
    if (dsts[i] == nullptr)
      continue;
    sizes.push_back(src_shape.tensor_size(i));
  }
  int samples_to_copy = sizes.size();

  if (src.IsContiguous() && samples_to_copy == num_samples) {
    type_info.template Copy<DstBackend, SrcBackend>(dsts, unsafe_raw_data(src), sizes.data(),
                                                    num_samples, order.stream(), use_copy_kernel);

  } else {
    SmallVector<const void *, 256> from;
    from.reserve(samples_to_copy);
    SmallVector<void *, 256> to;
    to.reserve(samples_to_copy);
    for (int i = 0; i < num_samples; i++) {
      if (dsts[i] == nullptr)
        continue;
      from.push_back(src.raw_tensor(i));
      to.push_back(dsts[i]);
    }

    type_info.template Copy<DstBackend, SrcBackend>(
          to.data(), from.data(), sizes.data(), samples_to_copy, order.stream(), use_copy_kernel);
  }
}

template <typename DstKind, typename SrcBackend>
inline void CopyToExternal(void* dst, const Tensor<SrcBackend> &src,
                           AccessOrder order, bool use_copy_kernel) {
  const bool src_device_access = (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned());
  const bool dst_device_access = cuda::kind_has_property<DstKind,
                                                         cuda::memory_access::device>::value;
  use_copy_kernel &= dst_device_access && src_device_access;
  using DstBackend = typename detail::kind2backend<DstKind>::type;
  CopyToExternalImpl<DstBackend, SrcBackend>(dst, src, order, use_copy_kernel);
}

template <typename DstKind, typename SrcBackend>
inline void CopyToExternal(void* dst, const TensorList<SrcBackend> &src,
                           AccessOrder order, bool use_copy_kernel) {
  const bool src_device_access = (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned());
  const bool dst_device_access = cuda::kind_has_property<DstKind,
                                                         cuda::memory_access::device>::value;
  use_copy_kernel &= dst_device_access && src_device_access;
  using DstBackend = typename detail::kind2backend<DstKind>::type;
  CopyToExternalImpl<DstBackend, SrcBackend>(dst, src, order, use_copy_kernel);
}

/**
 * @brief Run-time dispatch - DO NOT USE unless really necessary
 */
template <typename SrcBackend>
inline void CopyToExternal(void* dst, mm::memory_kind_id kind_id, const Tensor<SrcBackend> &src,
                           AccessOrder order, bool use_copy_kernel) {
  TYPE_SWITCH(kind_id, mm::kind2id, Kind, (mm::memory_kind::host,
                                           mm::memory_kind::pinned,
                                           mm::memory_kind::device,
                                           mm::memory_kind::managed),
    (CopyToExternal<Kind, SrcBackend>(dst, src, order, use_copy_kernel)),
    (throw std::logic_error("Unreachable code - invalid memory kind.")));
}

template <typename SrcBackend>
inline void CopyToExternal(void* dst, mm::memory_kind_id kind_id, const TensorList<SrcBackend> &src,
                           AccessOrder order, bool use_copy_kernel) {
  TYPE_SWITCH(kind_id, mm::kind2id, Kind, (mm::memory_kind::host,
                                           mm::memory_kind::pinned,
                                           mm::memory_kind::device,
                                           mm::memory_kind::managed),
    (CopyToExternal<Kind, SrcBackend>(dst, src, order, use_copy_kernel)),
    (throw std::logic_error("Unreachable code - invalid memory kind.")));
}

template <typename DstKind, typename SrcBackend>
inline void CopyToExternal(void** dsts, const TensorList<SrcBackend> &src,
                           AccessOrder order, bool use_copy_kernel) {
  bool src_device_access = (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned());
  bool dst_device_access = cuda::kind_has_property<DstKind, cuda::memory_access::device>::value;
  use_copy_kernel &= dst_device_access && src_device_access;
  using DstBackend = typename detail::kind2backend<DstKind>::type;
  CopyToExternalImpl<DstBackend, SrcBackend>(dsts, src, order, use_copy_kernel);
}

/**
 * @brief Run-time dispatch - DO NOT USE unless really necessary
 */
template <typename SrcBackend>
inline void CopyToExternal(void** dsts, mm::memory_kind_id kind_id,
                           const TensorList<SrcBackend> &src,
                           AccessOrder order, bool use_copy_kernel) {
  TYPE_SWITCH(kind_id, mm::kind2id, Kind, (mm::memory_kind::host,
                                           mm::memory_kind::pinned,
                                           mm::memory_kind::device,
                                           mm::memory_kind::managed),
    (CopyToExternal<Kind, SrcBackend>(dsts, src, order, use_copy_kernel)),
    (throw std::logic_error("Unreachable code - invalid memory kind.")));
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_COPY_TO_EXTERNAL_H_
