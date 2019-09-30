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

#ifndef DALI_CORE_PARSE_UTILS_H_
#define DALI_CORE_PARSE_UTILS_H_

namespace dali {

namespace detail {

template <typename T, int nbytes, bool is_little_endian>
T ReadValueImpl(const uint8_t* data) {
  static_assert(std::is_integral<T>::value, "T must be an integral type");
  static_assert(std::is_unsigned<T>::value || sizeof(T) == nbytes,
    "T must be an unsigned type or nbytes == sizeof(T)");
  static_assert(sizeof(T) >= nbytes, "T can't hold the requested number of bytes");
  T value = 0;
  for (int i = 0; i < nbytes; i++) {
    unsigned shift = is_little_endian ? (i*8) : (nbytes-1-i)*8;
    value |= data[i] << shift;
  }
  return value;
}

}  // namespace detail


/**
 * @brief Reads value of size `nbytes` from a stream of bytes (little-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueLE(const uint8_t* data) {
  return detail::ReadValueImpl<T, nbytes, true>(data);
}

/**
 * @brief Reads value of size `nbytes` from a stream of bytes (big-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueBE(const uint8_t* data) {
  return detail::ReadValueImpl<T, nbytes, false>(data);
}

}  // namespace dali

#endif  // DALI_CORE_PARSE_UTILS_H_
