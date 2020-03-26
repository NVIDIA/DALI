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

#ifndef DALI_KERNELS_REDUCE_ONLINE_REDUCER_H_
#define DALI_KERNELS_REDUCE_ONLINE_REDUCER_H_

#include "dali/kernels/reduce/reduce.h"

namespace dali {
namespace kernels {

/**
 * @brief Implements online reduction
 *
 * The default variant implements a tree reduction.
 *
 * This class does the reduction without prior knowledge of how many reduced values
 * there are going to be.
 *
 * The number of elements which can be reduced using this class is:
 * `max_seq * 2^capacity - 1`.
 */
template <typename Acc, typename Reduction, uint16_t max_seq, unsigned capacity = 32,
          bool use_tree = !reductions::is_accurate<Reduction>::value>
struct OnlineReducer {
  Acc tmp[capacity];  // NOLINT
  Acc current;
  uint32_t num;
  uint16_t seq;
  uint8_t ofs;

  DALI_HOST_DEV DALI_FORCEINLINE
  void reset(Reduction r = {}) {
    current = r.template neutral<Acc>();
    num = 0;
    seq = 0;
    ofs = 0;
  }

  /**
   * This function is much more efficient than direct indexing when idx is warp-uniform
   */
  DALI_HOST_DEV DALI_FORCEINLINE
  Acc &uniform_val(int idx) {
    switch (idx) {
      #define GET_UNIFORM_REF(index) case index: if (index < capacity) return tmp[index]
      GET_UNIFORM_REF(0);
      GET_UNIFORM_REF(1);
      GET_UNIFORM_REF(2);
      GET_UNIFORM_REF(3);
      GET_UNIFORM_REF(4);
      GET_UNIFORM_REF(5);
      GET_UNIFORM_REF(6);
      GET_UNIFORM_REF(7);
      GET_UNIFORM_REF(8);
      GET_UNIFORM_REF(9);
      GET_UNIFORM_REF(10);
      GET_UNIFORM_REF(11);
      GET_UNIFORM_REF(12);
      GET_UNIFORM_REF(13);
      GET_UNIFORM_REF(14);
      GET_UNIFORM_REF(15);
      GET_UNIFORM_REF(16);
      GET_UNIFORM_REF(17);
      GET_UNIFORM_REF(18);
      GET_UNIFORM_REF(19);
      GET_UNIFORM_REF(20);
      GET_UNIFORM_REF(21);
      GET_UNIFORM_REF(22);
      GET_UNIFORM_REF(23);
      GET_UNIFORM_REF(24);
      GET_UNIFORM_REF(25);
      GET_UNIFORM_REF(26);
      GET_UNIFORM_REF(27);
      GET_UNIFORM_REF(28);
      GET_UNIFORM_REF(29);
      GET_UNIFORM_REF(30);
      GET_UNIFORM_REF(31);
    }
    return current;
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  const Acc &uniform_val(int idx) const {
    return const_cast<OnlineReducer*>(this)->uniform_val(idx);
  }

  /**
   * @brief Store one value and tree-reduce the contents, as necessary
   */
  DALI_HOST_DEV DALI_FORCEINLINE
  void add(Acc value, Reduction r = {}) {
    // The algorithm:
    // First, reduce sequentially in the `current` bin until max_seq is reached...
    r(current, value);
    if (++seq == max_seq) {
      uniform_val(ofs) = current;  // store current bin in the tmp buffer
      seq = 0;
      // reset current bin so it can accept new sequential reductions
      current = r.template neutral<Acc>();

      num++;
      // ...now, for clarity let's forget about the sequential stage.
      // So far the table `tmp` contains `num` accumulated values.
      // If two bins contain the same number of accumulated values, they can be folded into one
      // - this frees a bin.
      // We'll use the bit pattern of number of accumulated samples to perform the folding.
      //
      // Example:
      // num (bin)    # values in bins -> folding
      // 1            1
      // 10           1 1 -> 2
      // 11           2 1
      // 100          2 1 1 -> 2 2 -> 4
      // 101          4 1
      // 110          4 1 1 -> 4 2
      // 111          4 2 1
      // 1000         4 2 1 1 -> 4 2 2 -> 4 4 -> 8
      //
      // The example shows that the number of folds is equal to the number of trailing zeros
      // in `num`. It also illustrates that required size of `tmp` is at most log2(num) + 1
      for (auto mask = num; (mask & 1) == 0; mask >>= 1) {
        r(uniform_val(ofs-1), uniform_val(ofs));
        ofs--;  // free the bin
      }
      ofs++;
    }
  }

  /**
   * @brief Reduce all stored values
   */
  DALI_HOST_DEV DALI_FORCEINLINE
  Acc result(Reduction r = {}) const {
    Acc acc = current;
    for (int i = ofs - 1; i >= 0; i--) {
      r(acc, uniform_val(i));
    }
    return acc;
  }
};

/**
 * @brief Implements trivial online reduction
 *
 * This class is used when the reduction does not accumulate any error - we only
 * need one accumulator to which we sequentially add values.
 */
template <typename Acc, typename Reduction, uint16_t max_seq, unsigned capacity>
struct OnlineReducer<Acc, Reduction, max_seq, capacity, false> {
  Acc current;

  DALI_HOST_DEV DALI_FORCEINLINE
  void reset(Reduction r = {}) {
    current = r.template neutral<Acc>();
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  void add(Acc value, Reduction r = {}) {
    r(current, value);
  }

  DALI_HOST_DEV DALI_FORCEINLINE
  Acc result(Reduction r = {}) const {
    return current;
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_REDUCE_ONLINE_REDUCER_H_
