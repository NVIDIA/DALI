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

#ifndef DALI_OPERATORS_READER_LOADER_UTILS_READSLICE_H_
#define DALI_OPERATORS_READER_LOADER_UTILS_READSLICE_H_

#include "dali/core/common.h"
#include "dali/pipeline/data/types.h"

namespace dali {

// some defines
#define READSLICE_ALLOWED_DIMS (1, 2, 3, 4, 5)

#define READSLICE_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
  double)

void CopySlice(Tensor<CPUBackend>& output,
               const Tensor<CPUBackend>& input,
               const TensorShape<>& anchor,
               const TensorShape<>& shape);

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_UTILS_READSLICE_H_
