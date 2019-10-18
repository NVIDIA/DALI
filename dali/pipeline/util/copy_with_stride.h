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

#ifndef DALI_PIPELINE_UTIL_COPY_WITH_STRIDE_H_
#define DALI_PIPELINE_UTIL_COPY_WITH_STRIDE_H_

#include <vector>
#include "dali/core/common.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

template <typename Backend>
DLL_PUBLIC void CopyWithStride(void *output, const void *input,
                               const Index *in_strides,
                               const Index *shape,
                               int ndim,
                               size_t item_size,
                               cudaStream_t stream = 0);

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_COPY_WITH_STRIDE_H_
