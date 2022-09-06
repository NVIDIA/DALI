// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
#define DALI_PIPELINE_DATA_TENSOR_VECTOR_H_

#include "dali/pipeline/data/tensor_list.h"

namespace dali {

/**
 * @brief Deprecated alias of one of the DALI batch containers, kept for backward compatibility.
 *
 * All usages should be moved to TensorList. The API of TensorList and TensorVector should be
 * compatible, TensorList supports both the contiguous and noncontiguous allocations
 * for sample-based construction.
 *
 * The original TensorVector was extended with TensorList API, and than the class names were
 * swapped.
 */
template <typename Backend>
using TensorVector
    [[deprecated("The API of DALI's batch containers: TensorVector and TensorList was unified, and "
                 "the class name TensorVector is no longer used. Use TensorList instead.")]] =
        TensorList<Backend>;

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_VECTOR_H_
