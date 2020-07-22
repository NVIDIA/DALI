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

template <typename SrcBackend>
inline void CopyToExternal(void* dst, AllocType dst_alloc_type,
                           const Buffer<SrcBackend> &src,
                           cudaStream_t stream, bool sync, bool use_copy_kernel) {
  VALUE_SWITCH(dst_alloc_type, DstType, (AllocType::Host, AllocType::Pinned, AllocType::GPU), (
    using DstBackend = alloc_to_backend<DstType>::type;
    use_copy_kernel &= (DstType == AllocType::GPU || DstType == AllocType::Pinned) &&
                       (std::is_same<SrcBackend, GPUBackend>::value || src.is_pinned());
    DeviceGuard d(src.device_id());
    auto &type_info = TypeTable::GetTypeInfo(DALIDataType::DALI_UINT8);
    type_info.template Copy<DstBackend, SrcBackend>(dst, src.raw_data(), src.nbytes(), stream,
                                                    use_copy_kernel);
  ), ());  // NOLINT

  if (sync)
    CUDA_CALL(cudaStreamSynchronize(stream));
}

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_COPY_TO_EXTERNAL_H_
