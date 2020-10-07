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

#ifndef DALI_CORE_BOUNDARY_H_
#define DALI_CORE_BOUNDARY_H_

#include "dali/core/host_dev.h"
#include "dali/core/force_inline.h"
#include "dali/core/geom/vec.h"

namespace dali {

/// Out-of-bounds index handling
namespace boundary {

/**
 * @brief Reflects out-of-range indices until fits in range.
 *
 * @param idx The index to clamp
 * @param lo Low (inclusive) bound
 * @param hi High (exclusive) bound
 *
 * This reflect flavor does not repeat the first and last element:
 * ```
 * lo--v     v--hi
 *     ABCDEF         is padded as
 * EDCBABCDEFEDCBABCD
 * ```
 */
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_101(T idx, T lo, T hi) {
  if (hi - lo < 2)
    return hi - 1;  // make it obviously wrong if hi <= lo
  for (;;) {
    if (idx < lo)
      idx = 2 * lo - idx;
    else if (idx >= hi)
      idx = 2 * hi - 2 - idx;
    else
      break;
  }
  return idx;
}

/// @brief Equivalent to `idx_reflect_101(idx, 0, size)`
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_101(T idx, T size) {
  return idx_reflect_101(idx, T(0), size);
}


/**
 * @brief Reflects out-of-range indices until fits in range.
 *
 * @param idx The index to clamp
 * @param lo Low (inclusive) bound
 * @param hi High (exclusive) bound
 *
 * This reflect flavor repeats the first and last element:
 * ```
 * lo--v     v--hi
 *     ABCDEF           is padded as
 * DCBAABCDEFFEDCBAABCD
 * ```
 */
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_1001(T idx, T lo, T hi) {
  if (hi - lo < 1)
    return hi - 1;  // make it obviously wrong if hi <= lo
  for (;;) {
    if (idx < lo)
      idx = 2 * lo - 1 - idx;
    else if (idx >= hi)
      idx = 2 * hi - 1 - idx;
    else
      break;
  }
  return idx;
}

/// @brief Equivalent to `idx_reflect_1001(idx, 0, size)`
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_1001(T idx, T size) {
  return idx_reflect_1001(idx, T(0), size);
}

/**
 * @brief Clamps out of range coordinates to [lo, hi)
 *
 * @param idx The index to clamp
 * @param lo Low (inclusive) bound
 * @param hi High (exclusive) bound
 */
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<T>::value, T> idx_clamp(T idx, T lo, T hi) {
  return clamp(idx, lo, hi - 1);
}

/// @brief Equivalent to `idx_clamp(idx, 0, size)`
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<T>::value, T> idx_clamp(T idx, T size) {
  return idx_clamp(idx, 0, size);
}

/// @brief Wraps out-of-range indices modulo `size`
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, T>
idx_wrap(T idx, T size) {
  idx %= size;
  return idx < 0 ? idx + size : idx;
}

/// @brief Wraps out-of-range indices modulo `size`
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
std::enable_if_t<std::is_unsigned<T>::value, T> idx_wrap(T idx, T size) {
  return idx % size;
}

// vector variants

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
ivec<n> idx_clamp(ivec<n> idx, ivec<n> lo, ivec<n> hi) {
  return clamp(idx, lo, hi - 1);
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
ivec<n> idx_clamp(ivec<n> idx, ivec<n> size) {
  return clamp(idx, ivec<n>(), size - 1);
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
ivec<n> idx_reflect_101(ivec<n> idx, ivec<n> lo, ivec<n> hi) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_reflect_101(idx[i], lo[i], hi[i]);
  return out;
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
ivec<n> idx_reflect_101(ivec<n> idx, ivec<n> size) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_reflect_101(idx[i], size[i]);
  return out;
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
ivec<n> idx_reflect_1001(ivec<n> idx, ivec<n> lo, ivec<n> hi) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_reflect_1001(idx[i], lo[i], hi[i]);
  return out;
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
ivec<n> idx_reflect_1001(ivec<n> idx, ivec<n> size) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_reflect_1001(idx[i], size[i]);
  return out;
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
ivec<n> idx_wrap(ivec<n> idx, ivec<n> size) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_wrap(idx[i], size[i]);
  return out;
}


}  // namespace boundary
}  // namespace dali

#endif  // DALI_CORE_BOUNDARY_H_
