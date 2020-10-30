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

#ifndef DALI_CORE_RANDOM_H_
#define DALI_CORE_RANDOM_H_

#include <algorithm>
#include <random>
#include <type_traits>
#include <utility>
#include "dali/core/util.h"

namespace dali {

/**
 * @brief Generates a random permutation of numbers in range [0..size(out)-1]
 */
template <typename Collection, typename RNG>
void random_permutation(Collection &out, RNG &rng) {
  std::iota(dali::begin(out), dali::end(out), 0);
  std::shuffle(dali::begin(out), dali::end(out), rng);
}

/**
 * @brief Generates random derangement, i.e. permutation without fixed points.
 */
template <typename Collection, typename RNG>
void random_derangement(Collection &out, RNG &rng) {
  int N = size(out);
  std::iota(dali::begin(out), dali::end(out), 0);
  for (int i = 0; i < N-1; i++) {
    std::uniform_int_distribution<int> dist(i+1, N-1);
    int j = dist(rng);
    std::swap(out[i], out[j]);
  }
}

/**
 * @brief Generates a sequence of random integers in range [lo..hi-1]
 */
template <typename Collection, typename RNG, typename T>
std::enable_if_t<std::is_integral<T>::value>
random_sequence(Collection &out, T lo, T hi, RNG &rng) {
  std::uniform_int_distribution<T> dist(lo, hi-1);
  for (auto &x : out)
    x = dist(rng);
}

/**
 * @brief Generates a sequence of random numbers in range [lo..hi)
 */
template <typename Collection, typename RNG, typename T>
std::enable_if_t<std::is_floating_point<T>::value>
random_sequence(Collection &out, T lo, T hi, RNG &rng) {
  std::uniform_real_distribution<T> dist(lo, hi);
  for (auto &x : out)
    x = dist(rng);
}

/**
 * @brief Generates a sequence of random integers in range [lo..hi-1] where out[i] != i
 *
 * If it's impossible to generate the sequence using given lo/hi values and sequence length,
 * the result is undefined.
 */
template <typename Collection, typename RNG>
void random_sequence_no_fixed_points(Collection &out, int lo, int hi, RNG &rng) {
  int N = size(out);
  std::uniform_int_distribution<int> dist1(lo, hi-1);
  int i = 0;
  // when index is below lo, no fixed points possible
  for (; i < std::min(lo, N); i++) {
    out[i] = dist1(rng);
  }
  if (i < N && hi > 0) {
    // this is the part where we need to care about fixed points
    std::uniform_int_distribution<int> dist2(lo, hi-2);
    for (; i < std::min(hi, N); i++) {
        int x = dist2(rng);  // use a smaller distribution - fixed points removed
        x += (x >= i);  // if we're at or above fixed point, move up
        out[i] = x;
    }
  }
  // we're above hi now - no fixed points possible
  for (; i < N; i++) {
    out[i] = dist1(rng);
  }
}

}  // namespace dali

#endif  // DALI_CORE_RANDOM_H_

