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

#ifndef DALI_CORE_TENSOR_LAYOUT_H_
#define DALI_CORE_TENSOR_LAYOUT_H_

#include <string>
#include <cstring>

namespace dali {

/** @brief Provides a domain-agnostic, flexible description of a tensor layout
 *
 * The object is essentially a string with storage optimized for short sequences.
 */
struct TensorLayout {
  constexpr TensorLayout() = default;

  /** @brief Constructs a TensorLayout from a C-style string */
  constexpr TensorLayout(const char *str) : TensorLayout(str, strlen(str)) {  // NOLINT
  }

  /** @brief Constructs a TensorLayout from a C-style string of known length */
  constexpr TensorLayout(const char *str, size_t n) {  // NOLINT
    assert(n < sizeof(data_));
    if (n >= sizeof(data_))
      n = sizeof(data_) - 1;
    for (size_t i = 0; i < n; i++)
      data_[i] = str[i];
    data_[n] = 0;
    size_ = n;
  }

  /** @brief Constructs a TensorLayout from a char array of known length, e.g. a string literal */
  template <size_t N>
  constexpr TensorLayout(const char (&s)[N])  // NOLINT
  : TensorLayout(s, N && s[N-1] == '\0' ? N - 1 : N) {
  }

  /** @brief Constructs a TensorLayout from std::string */
  TensorLayout(const std::string &s) : TensorLayout(s.data(), s.length()) {  // NOLINT
  }

  constexpr char &operator[](int d) noexcept {
    return data_[d];
  }
  constexpr const char &operator[](int d) const noexcept {
    return data_[d];
  }

  /** @brief Returns a pointer to the internal representation of the layout */
  constexpr const char *c_str() const noexcept {
    return data_;
  }
  /** @brief Returns a pointer to the internal representation of the layout */
  constexpr const char *data() const noexcept {
    return data_;
  }
  /** @brief Copies the contents to std::string */
  std::string str() const { return c_str(); }

  constexpr int find(char dim_name) const noexcept {
    for (int i = 0; i < ndim(); i++) {
      if (data_[i] == dim_name)
        return i;
    }
    return -1;
  }

  /** @brief Checks if the string contains a character */
  constexpr bool contains(char dim_name) const noexcept {
    return find(dim_name) >= 0;
  }

  /** @brief Provides a three-way comparison against another TensorLayout */
  int compare(const TensorLayout &tl) const noexcept {
    return std::strcmp(c_str(), tl.c_str());
  }

  bool operator<(const TensorLayout &tl) const noexcept {
    return compare(tl) < 0;
  }
  bool operator>(const TensorLayout &tl) const noexcept {
    return compare(tl) > 0;
  }
  bool operator<=(const TensorLayout &tl) const noexcept {
    return compare(tl) <= 0;
  }
  bool operator>=(const TensorLayout &tl) const noexcept {
    return compare(tl) >= 0;
  }
  bool operator==(const TensorLayout &tl) const noexcept {
    return size_ == tl.size_ && compare(tl) == 0;
  }
  bool operator!=(const TensorLayout &tl) const noexcept {
    return !(*this == tl);
  }

  /** @brief Number of characters, excluding the (always present) null terminator */
  constexpr uint8_t size() const noexcept { return size_; }
  /** @brief Number of dimensions described by this object; same value as size() */
  constexpr int ndim() const noexcept { return size(); }
  /** @brief Returns true if size() == 0 */
  constexpr bool empty() const noexcept { return size() == 0; }

  using iterator = char*;
  using const_iterator = const char*;

  constexpr iterator begin() noexcept               { return data_; }
  constexpr iterator end() noexcept                 { return data_ + size_; }
  constexpr auto begin() const noexcept             { return cbegin(); }
  constexpr auto end() const noexcept               { return cend(); }
  constexpr const_iterator cbegin() const noexcept  { return data_; }
  constexpr const_iterator cend() const noexcept    { return data_ + size_; }

  TensorLayout sub(int start_dim, int n) const noexcept {
    assert(start_dim + n <= ndim());
    return TensorLayout(begin() + start_dim, n);
  }

  TensorLayout first(int n) const noexcept {
    assert(n <= ndim());
    return TensorLayout(begin(), n);
  }

  TensorLayout last(int n) const noexcept {
    assert(n <= ndim());
    return TensorLayout(end() - n, n);
  }

  /** @brief Stores the dimension descriptions as a null-terminated string */
  uint8_t size_ = 0;
  char data_[15] = { 0 };
};

static_assert(sizeof(TensorLayout) == 16, "Tensor layout size should be exactly 16B");

#define DEFINE_TENSOR_LAYOUT_COMPARISON(op)\
bool operator op (const TensorLayout &tl, const std::string &s) {\
  return std::strcmp(tl.c_str(), s.c_str()) op 0;\
}\
bool operator op (const TensorLayout &tl, const char *s) {\
  return std::strcmp(tl.c_str(), s) op 0;\
}\
bool operator op (const std::string &s, const TensorLayout &tl) {\
  return std::strcmp(s.c_str(), tl.c_str()) op 0;\
}\
bool operator op (const char *s, const TensorLayout &tl) {\
  return std::strcmp(s, tl.c_str()) op 0;\
}

DEFINE_TENSOR_LAYOUT_COMPARISON(<)  // NOLINT
DEFINE_TENSOR_LAYOUT_COMPARISON(>)  // NOLINT
DEFINE_TENSOR_LAYOUT_COMPARISON(<=)
DEFINE_TENSOR_LAYOUT_COMPARISON(>=)
DEFINE_TENSOR_LAYOUT_COMPARISON(==)
DEFINE_TENSOR_LAYOUT_COMPARISON(!=)

/** Provides basic functions for querying TensorLayout properties */
struct LayoutInfo {
  static int HasSampleDim(const TensorLayout &tl) {
    // sample dim may only appear at outermost level
    return !tl.empty() && tl[0] == 'N';
  }
  static int DimIndex(const TensorLayout &tl, char dim_symbol) {
    return tl.find(dim_symbol);
  }
};

/** @brief Provides functions for querying TensorLayout properties,
 *         assuming that the layout describes an image
 */
struct ImageLayoutInfo : LayoutInfo {
  static int NumSpatialDims(const TensorLayout &tl) {
    int s = 0;
    for (int i = 0; i < tl.ndim(); i++) {
      switch (tl[i]) {
      case 'D':
      case 'H':
      case 'W':
        s++;
        break;
      default:
        break;
      }
    }
    return s;
  }
  static int ChannelDimIndex(const TensorLayout &tl) {
    return DimIndex(tl, 'C');
  }
  static int HasChannel(const TensorLayout &tl) {
    return ChannelDimIndex(tl) >= 0;
  }
  static int Is2D(const TensorLayout &tl) {
    return NumSpatialDims(tl) == 2;
  }
  static int Is3D(const TensorLayout &tl) {
    return NumSpatialDims(tl) == 3;
  }
  static bool IsChannelFirst(const TensorLayout &tl) {
    return tl[0] == 'C' || (tl[0] == 'N' && tl[1] == 'C');
  }
  static bool IsChannelLast(const TensorLayout &tl) {
    return !tl.empty() && tl[tl.ndim()-1] == 'C';
  }
  static bool IsImage(const TensorLayout &tl) {
    return NumSpatialDims(tl) >= 2;
  }
};

/** @brief Provides functions for querying TensorLayout properties,
 *         assuming that the layout describes a video
 */
struct VideoLayoutInfo : ImageLayoutInfo {
  /** @brief Returns the index of the dimension referring to frames */
  static int FrameDim(const TensorLayout &tl) {
    return DimIndex(tl, 'F');
  }
  static bool IsChannelFirst(const TensorLayout &tl) {
    return tl[FrameDim(tl)+1] == 'C';
  }
  static bool IsSequence(const TensorLayout &tl) {
    return tl.contains('F');
  }
  static bool IsVideo(const TensorLayout &tl) {
    return IsSequence(tl) && IsImage(tl);
  }
  static bool IsStillImage(const TensorLayout &tl) {
    return !IsSequence(tl) && IsImage(tl);
  }
};

}  // namespace dali

#endif  // DALI_CORE_TENSOR_LAYOUT_H_
