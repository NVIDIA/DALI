// Copyright (c) 2019, 2022, 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <string_view>
#include "dali/core/force_inline.h"
#include "dali/core/format.h"
#include "dali/core/geom/vec.h"
#include "dali/core/host_dev.h"

namespace dali {

/// Out-of-bounds index handling
namespace boundary {

/**
 * Specifies, how to handle pixels on a border of an image.
 *
 * ---------------------------------------------------------------|
 * | CONSTANT     | iiiiii|abcdefgh|iiiiiii with some specified i |
 * ---------------------------------------------------------------|
 * | CLAMP        | aaaaaa|abcdefgh|hhhhhhh                       |
 * ---------------------------------------------------------------|
 * | REFLECT_1001 | fedcba|abcdefgh|hgfedcb                       |
 * ---------------------------------------------------------------|
 * | REFLECT_101  | gfedcb|abcdefgh|gfedcba                       |
 * ---------------------------------------------------------------|
 * | WRAP         | cdefgh|abcdefgh|abcdefg                       |
 * ---------------------------------------------------------------|
 * | TRANSPARENT  | uvwxyz|abcdefgh|ijklmno                       |
 * ---------------------------------------------------------------|
 * | ISOLATED     | do not look outside of ROI                    |
 * ---------------------------------------------------------------|
 */
enum class BoundaryType {
  CONSTANT,
  CLAMP,
  REFLECT_1001,
  REFLECT_101,
  WRAP,
  TRANSPARENT,
  ISOLATED
};

template <typename T>
struct Boundary {
  BoundaryType type = BoundaryType::REFLECT_101;
  T value = {};
};

inline std::string to_string(const BoundaryType& type) {
  switch (type) {
    case BoundaryType::CONSTANT:
      return "constant";
    case BoundaryType::CLAMP:
      return "clamp";
    case BoundaryType::REFLECT_1001:
      return "reflect_1001";
    case BoundaryType::REFLECT_101:
      return "reflect_101";
    case BoundaryType::WRAP:
      return "wrap";
    case BoundaryType::TRANSPARENT:
      return "transparent";
    case BoundaryType::ISOLATED:
      return "isolated";
    default:
      return "<unknown>";
  }
}

inline bool TryParse(BoundaryType &type, std::string_view type_str) {
  std::string s(type_str);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](auto c) { return std::tolower(c); });
  if (s == "constant" || s == "const" || s == "fill" || s == "pad") {
    type = BoundaryType::CONSTANT;
    return true;
  }
  if (s == "clamp" || s == "edge") {
    type = BoundaryType::CLAMP;
    return true;
  }
  if (s == "reflect" || s == "reflect_1001" || s == "reflect1001" || s == "1001") {
    type = BoundaryType::REFLECT_1001;
    return true;
  }
  if (s == "reflect_101" || s == "reflect101" || s == "101") {
    type = BoundaryType::REFLECT_101;
    return true;
  }
  if (s == "wrap") {
    type = BoundaryType::WRAP;
    return true;
  }
  if (s == "transparent") {
    type = BoundaryType::TRANSPARENT;
    return true;
  }
  if (s == "isolated") {
    type = BoundaryType::ISOLATED;
    return true;
  }
  return false;
}

inline BoundaryType parse(std::string_view type) {
  BoundaryType result;
  if (!TryParse(result, type))
    return result;
  throw std::invalid_argument(make_string("Unknown boundary type was specified: ``", type, "``."));
}

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
constexpr std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_101(T idx, T lo, T hi) {
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
constexpr std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_101(T idx, T size) {
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
constexpr std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_1001(T idx, T lo, T hi) {
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
constexpr std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_1001(T idx, T size) {
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
constexpr std::enable_if_t<std::is_integral<T>::value, T> idx_clamp(T idx, T lo, T hi) {
  return clamp(idx, lo, hi - 1);
}

/// @brief Equivalent to `idx_clamp(idx, 0, size)`
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr std::enable_if_t<std::is_integral<T>::value, T> idx_clamp(T idx, T size) {
  return idx_clamp(idx, 0, size);
}

/// @brief Wraps out-of-range indices modulo `size`
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, T>
idx_wrap(T idx, T size) {
  idx %= size;
  return idx < 0 ? idx + size : idx;
}

/// @brief Wraps out-of-range indices modulo `size`
template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr std::enable_if_t<std::is_unsigned<T>::value, T> idx_wrap(T idx, T size) {
  return idx % size;
}

// vector variants

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr ivec<n> idx_clamp(ivec<n> idx, ivec<n> lo, ivec<n> hi) {
  return clamp(idx, lo, hi - 1);
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr ivec<n> idx_clamp(ivec<n> idx, ivec<n> size) {
  return clamp(idx, ivec<n>(), size - 1);
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr ivec<n> idx_reflect_101(ivec<n> idx, ivec<n> lo, ivec<n> hi) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_reflect_101(idx[i], lo[i], hi[i]);
  return out;
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr ivec<n> idx_reflect_101(ivec<n> idx, ivec<n> size) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_reflect_101(idx[i], size[i]);
  return out;
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr ivec<n> idx_reflect_1001(ivec<n> idx, ivec<n> lo, ivec<n> hi) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_reflect_1001(idx[i], lo[i], hi[i]);
  return out;
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr ivec<n> idx_reflect_1001(ivec<n> idx, ivec<n> size) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_reflect_1001(idx[i], size[i]);
  return out;
}

template <int n>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr ivec<n> idx_wrap(ivec<n> idx, ivec<n> size) {
  ivec<n> out;
  for (int i = 0; i < n; i++)
    out[i] = idx_wrap(idx[i], size[i]);
  return out;
}


template <typename T>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr T handle_bounds(T idx, T data_extent, BoundaryType border_type) {
  if (border_type == BoundaryType::CLAMP) {
    return idx_clamp(idx, T(0), data_extent);
  } else if (border_type == BoundaryType::REFLECT_101) {
    return idx_reflect_101(idx, T(0), data_extent);
  } else if (border_type == BoundaryType::REFLECT_1001) {
    return idx_reflect_1001(idx, T(0), data_extent);
  } else if (border_type == BoundaryType::WRAP) {
    return idx_wrap(idx, data_extent);
  } else {
    return idx;
  }
}


}  // namespace boundary
}  // namespace dali

#endif  // DALI_CORE_BOUNDARY_H_
