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

#ifndef DALI_CORE_BYTE_IO_H_
#define DALI_CORE_BYTE_IO_H_

namespace dali {

namespace detail {

template <int nbytes, bool is_little_endian, typename T>
void ReadValueImpl(T &value, const uint8_t* data) {
  static_assert(std::is_integral<T>::value, "T must be an integral type");
  static_assert(sizeof(T) >= nbytes, "T can't hold the requested number of bytes");
  value = 0;
  constexpr unsigned pad = (sizeof(T) - nbytes) * 8;  // handle sign when nbytes < sizeof(T)
  for (int i = 0; i < nbytes; i++) {
    unsigned shift = is_little_endian ? (i*8) + pad: (sizeof(T)-1-i)*8;
    value |= data[i] << shift;
  }
  value >>= pad;
}

template <int nbytes, bool is_little_endian>
void ReadValueImpl(float &value, const uint8_t* data) {
  static_assert(nbytes == sizeof(float),
    "nbytes is expected to be the same as sizeof(float)");
  uint32_t tmp;
  ReadValueImpl<nbytes, is_little_endian>(tmp, data);
  memcpy(&value, &tmp, sizeof(float));
}

}  // namespace detail


/**
 * @brief Reads value of size `nbytes` from a stream of bytes (little-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueLE(const uint8_t* data) {
  T ret;
  detail::ReadValueImpl<nbytes, true>(ret, data);
  return ret;
}

/**
 * @brief Reads value of size `nbytes` from a stream of bytes (big-endian)
 */
template <typename T, int nbytes = sizeof(T)>
T ReadValueBE(const uint8_t* data) {
  T ret;
  detail::ReadValueImpl<nbytes, false>(ret, data);
  return ret;
}

}  // namespace dali

#endif  // DALI_CORE_BYTE_IO_H_
