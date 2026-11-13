// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_EXECUTOR_SOURCE_INFO_PROPAGATION_H_
#define DALI_PIPELINE_EXECUTOR_SOURCE_INFO_PROPAGATION_H_

#include "dali/core/api_helper.h"

namespace dali {

class Workspace;

void DLL_PUBLIC ClearOutputSourceInfo(Workspace &ws);

/**
 * @brief Propagates the source info and decoded image cache key from input(s) to the outputs
 *
 * This function propagates the source info and cache key metadata from the source tensors to the destination
 * tensors under the following conditions:
 * - there's at least one input and at least one output
 * - none of the outputs has a SourceInfo already set
 * - all the input and output batch sizes are equal
 * - the source info for all inputs must be equal or empty (see below)
 * - the cache keys for all inputs must be equal or empty
 *
 * The cache key follows the source info so a reader and decoder agree on cached samples
 * even when an intermediate operator relies on automatic metadata propagation.
 *
 * Consistent source info
 *      inp0:    inp1:    inp2:
 *      'a.jpg', 'a.jpg', <empty>
 *      'b.jpg', 'b.jpg', <empty>
 *
 * Inconsistent source info
 *      inp0:    inp1:
 *      'a.jpg', 'a_mask.png'
 *      'b.jpg', 'b_mask.png'
 *
 * @return true,  if the propagation was successfully performed, false otherwise
 */
bool DLL_PUBLIC PropagateSourceInfo(Workspace &ws);

}  // namespace dali

#endif  // DALI_PIPELINE_EXECUTOR_SOURCE_INFO_PROPAGATION_H_
