// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef  DALI_KERNELS_KERNEL_REQ_H_
#define  DALI_KERNELS_KERNEL_REQ_H_

#include <array>
#include <vector>
#include <algorithm>
#include "dali/kernels/context.h"
#include "dali/core/tuple_helpers.h"
#include "dali/core/util.h"

namespace dali {
namespace kernels {

/**
 * @brief Represents requirements for kernel to do its job for given inputs and arguments.
 */
struct KernelRequirements {
  std::vector<TensorListShape<DynamicDimensions>> output_shapes;

  std::array<size_t, (size_t)AllocType::Count> scratch_sizes = {};

  /**
   * @param reuse_scratch  - if true, scratch size is taken to be maximum from that for
   *                         all input sets, otherwise it's the sum
   * @param new_req        - requirements for the new input set, to be merged with this one
   * @return               - *this, for chaining
   */
  KernelRequirements &AddInputSet(const KernelRequirements &new_req, bool reuse_scratch) {
    auto &r = new_req;

    append(output_shapes, r.output_shapes);

    for (size_t i = 0; i < scratch_sizes.size(); i++) {
      if (reuse_scratch)
        scratch_sizes[i] = std::max(scratch_sizes[i], r.scratch_sizes[i]);
      else
        scratch_sizes[i] += r.scratch_sizes[i];
    }
    return *this;
  }
};

/**
 * @brief A utility class for adding scratchpad requirements with proper alignment,
 *        assuming bump allocation.
 */
struct ScratchpadEstimator {
  /**
   * @brief Adds a new memory requirement for count instances of T
   *
   * The method includes padding, assuming the add function is called in order of allocations.
   * The resulting allocation size is equal to size of a structure which contains all allocated
   * objects assuming natural alignment.
   * The estimator assumes that scratch buffer implementation will provide memory block based at
   * largest possible alignment boundary.
   *
   * @return Total number of bytes required for given allocation method,
   *         including this allocation.
   */
  template <typename T>
  size_t add(AllocType alloc_type, size_t count, size_t alignment = alignof(T)) {
    size_t offset = align_up(sizes[(size_t)alloc_type], alignment);
    if (count) {
      sizes[(size_t)alloc_type] = offset + count * sizeof(T);
    }
    return sizes[(size_t)alloc_type];
  }

  std::array<size_t, (size_t)AllocType::Count> sizes = {};
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_KERNEL_REQ_H_
