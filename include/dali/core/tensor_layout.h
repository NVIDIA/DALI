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

#include <cassert>
#include <cstring>
#include <array>
#include <ostream>
#include <string>
#include <stdexcept>
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/host_dev.h"
#include "dali/core/small_vector.h"

namespace dali {

/**
 * @brief Provides a domain-agnostic, flexible description of a tensor layout
 *
 * The object is essentially a string with storage optimized for short sequences.
 */
class TensorLayout {
 public:
  DALI_HOST_DEV
  constexpr TensorLayout() { set_size(0); }

  /** @brief Constructs a TensorLayout from a C-style string */
  DALI_HOST_DEV
  constexpr TensorLayout(const char *str) {  // NOLINT
    int n = 0;
    for (; str[n] && n < max_ndim; n++)
      data_[n] = str[n];
    assert(!str[n] && "Input string too long!");
    set_size(n);
  }

  /**
   * @brief Constructs a TensorLayout from a C-style string of known length.
   *
   * @param str - pointer to the beginning of the string; it shall not contain '\0' except as
   *              an optional null terminator
   * @param n   - number of characters in str
   */
  DALI_HOST_DEV
  constexpr TensorLayout(const char *str, size_t n) {  // NOLINT
    assert(n < sizeof(data_) && "Input string too long");
    if (n >= sizeof(data_))
      n = sizeof(data_) - 1;
    for (size_t i = 0; i < n; i++)
      data_[i] = str[i];
    set_size(n);
  }

  /**
   * @brief Constructs a TensorLayout from a char array of known length, e.g. a string literal
   *
   * Converts a character array or a string literal to a TensorLayout. If the literal contains
   * null characters in the middle, the result is undefined.
   */
  template <size_t N>
  DALI_HOST_DEV
  constexpr TensorLayout(const char (&s)[N])  // NOLINT
  : TensorLayout(s, N > 0 && s[N-1] == '\0' ? N - 1 : N) {
  }

  /** @brief Constructs a TensorLayout from std::string */
  TensorLayout(const std::string &s) : TensorLayout(s.data(), s.length()) {  // NOLINT
  }

  /**
   * @brief Returns a reference to the d-th dimension in the layout
   * @remarks Value at ndim() is always '\0'.
   *          Any of the following leads to undefined behavior:
   *            - accessing values at d > ndim()
   *            - changing '\0' terminator to any other value
   *            - replacing any meaningful character with '\0'
   */
  DALI_HOST_DEV
  constexpr char &operator[](int d) noexcept {
    return data_[d];
  }

  /**
   * @brief Returns a reference to the d-th dimension in the layout
   * @remarks Value at ndim() is always '\0', accessing values beyond ndim() is forbidden.
   */
  DALI_HOST_DEV
  constexpr const char &operator[](int d) const noexcept {
    return data_[d];
  }

  /** @brief Returns a pointer to the internal representation of the layout */
  DALI_HOST_DEV
  constexpr const char *c_str() const noexcept {
    return data_;
  }
  /** @brief Returns a pointer to the internal representation of the layout */
  DALI_HOST_DEV
  constexpr const char *data() const noexcept {
    return data_;
  }
  /** @brief Copies the contents to std::string */
  std::string str() const { return c_str(); }

  /** @brief Copies the contents to std::string */
  explicit operator std::string() const { return c_str(); }

  /** @brief Searches for the first occurrence of dim_name, starting at index start */
  DALI_HOST_DEV
  constexpr int find(char dim_name, int start = 0) const noexcept {
    for (int i = start; i < ndim(); i++) {
      if (data_[i] == dim_name)
        return i;
    }
    return -1;
  }

  /** @brief Checks if the string contains a character */
  DALI_HOST_DEV
  constexpr bool contains(char dim_name) const noexcept {
    return find(dim_name) >= 0;
  }

  /**
   * @brief Returns a layout without the dimension specified in dim_name
   *
   * @return Layout without the first occurrence of given dimension.
   *         When repetitions are present, only the first occurrence is removed.
   *         When given dim is not found, the function returns unchanged layout.
   */
  DALI_HOST_DEV
  constexpr TensorLayout skip(char dim_name) const noexcept;

  /** @brief Provides a three-way comparison against another TensorLayout */
  DALI_HOST_DEV
  constexpr int compare(const TensorLayout &tl) const noexcept {
    int n = ndim() < tl.ndim() ? ndim() : tl.ndim();
    // <= to include the null terminator
    for (int i = 0; i <= n; i++) {
      int d = data_[i] - tl.data_[i];
      if (d)
        return d;
    }
    return 0;
  }

  DALI_HOST_DEV
  bool operator<(const TensorLayout &tl) const noexcept {
    return compare(tl) < 0;
  }
  DALI_HOST_DEV
  bool operator>(const TensorLayout &tl) const noexcept {
    return compare(tl) > 0;
  }
  DALI_HOST_DEV
  bool operator<=(const TensorLayout &tl) const noexcept {
    return compare(tl) <= 0;
  }
  DALI_HOST_DEV
  bool operator>=(const TensorLayout &tl) const noexcept {
    return compare(tl) >= 0;
  }
  DALI_HOST_DEV
  bool operator==(const TensorLayout &tl) const noexcept {
    return data_[max_ndim] == tl.data_[max_ndim] && compare(tl) == 0;
  }
  DALI_HOST_DEV
  bool operator!=(const TensorLayout &tl) const noexcept {
    return !(*this == tl);
  }

  /** @brief Number of characters, excluding the (always present) null terminator */
  DALI_HOST_DEV
  constexpr uint8_t size() const noexcept { return max_ndim - data_[max_ndim]; }
  /** @brief Number of dimensions described by this object; same value as size() */
  DALI_HOST_DEV
  constexpr int ndim() const noexcept { return size(); }
  /** @brief Returns true if size() == 0 */
  DALI_HOST_DEV
  constexpr bool empty() const noexcept { return size() == 0; }

  void resize(size_t new_size, char fill_value = '?') noexcept {
    assert(new_size < max_ndim);
    auto prev_size = size();
    set_size(new_size);
    for (size_t i = prev_size; i < new_size; i++) {
      data_[i] = fill_value;
    }
  }

  DALI_HOST_DEV
  bool is_permutation_of(TensorLayout b) const {
    // argument passed by value, because we'll reorder it to match *this
    if (ndim() != b.ndim())
      return false;

    int n = ndim();
    // not the nicest O(n^2) algorithm, but who cares with ndim() <= 15
    for (int i = 0; i < n; i++) {
      char c = data_[i];
      if (c == b[i])
          continue;
      int j;
      for (j = i + 1; j < n; j++) {
        if (b[j] == c)
          break;
      }
      if (j == n)
        return false;
      char tmp = b[i];
      b[i] = b[j];
      b[j] = tmp;
    }
    return true;
  }

  using iterator = char*;
  using const_iterator = const char*;
  using value_type = char;
  using pointer = char*;
  using const_pointer = char*;
  using reference = char &;
  using const_reference = const char &;

  DALI_HOST_DEV
  constexpr iterator begin() noexcept               { return data_; }
  DALI_HOST_DEV
  constexpr iterator end() noexcept                 { return data_ + ndim(); }
  DALI_HOST_DEV
  constexpr auto begin() const noexcept             { return cbegin(); }
  DALI_HOST_DEV
  constexpr auto end() const noexcept               { return cend(); }
  DALI_HOST_DEV
  constexpr const_iterator cbegin() const noexcept  { return data_; }
  DALI_HOST_DEV
  constexpr const_iterator cend() const noexcept    { return data_ + ndim(); }

  DALI_HOST_DEV
  TensorLayout sample_layout() const {
    if (empty())
      return {};
#ifdef __CUDA_ARCH__
    assert(data_[0] == 'N');
#else
    if (data_[0] != 'N')
      throw std::logic_error("This is not a multi-sample layout: \"" + str() + "\"");
#endif
    return sub(1);
  }

  DALI_HOST_DEV
  constexpr TensorLayout sub(int start, int n = -1) const noexcept {
    assert(start >= 0 && start <= ndim());
    if (n < 0)
      n = ndim() - start;
    assert(start + n <= ndim());
    return TensorLayout(begin() + start, n);
  }

  DALI_HOST_DEV
  constexpr TensorLayout first(int n) const noexcept {
    assert(n <= ndim());
    return TensorLayout(begin(), n);
  }

  DALI_HOST_DEV
  constexpr TensorLayout last(int n) const noexcept {
    assert(n <= ndim());
    return TensorLayout(end() - n, n);
  }

  DALI_HOST_DEV TensorLayout& operator+=(const TensorLayout &oth) noexcept {
    *this = *this + oth;
    return *this;
  }

  DALI_HOST_DEV TensorLayout& operator+=(char oth) noexcept {
    *this = *this + oth;
    return *this;
  }

  DALI_HOST_DEV
  void erase(int index) noexcept {
    assert(index >= 0 && index < ndim());
    *this = first(index) + sub(index + 1);
  }

  static constexpr int max_ndim = 16-1;

 private:
  /**
   * @brief Stores the dimension descriptions as a null-terminated string
   *
   * Last character in the string is aliased with max_ndim - length.
   * If length == max_ndim, then 0 is stored and aliased as '\0' terminator,
   * allowing 1 extra character to be stored.
   */
  char data_[max_ndim + 1] = { 0 };
  DALI_HOST_DEV
  void set_size(int n) {
    assert(n >= 0 && n <= max_ndim);
    data_[max_ndim] = max_ndim - n;
  }

  DALI_HOST_DEV
  friend constexpr TensorLayout operator+(const TensorLayout &a, const TensorLayout &b);
  DALI_HOST_DEV
  friend constexpr TensorLayout operator+(const TensorLayout &a, char b);
};

static_assert(sizeof(TensorLayout) == 16, "Tensor layout size should be exactly 16B");

/** @brief Appends a single element to the layout string */
DALI_HOST_DEV
constexpr TensorLayout operator+(const TensorLayout &a, char b) {
  assert(a.size() + 1 < TensorLayout::max_ndim);
  TensorLayout result = a;
  int i = result.ndim();
  result[i++] = b;
  result[i] = '\0';

  /* Cannot use
   *   result.set_size(i);
   * because it's not constexpr
   */
  result.data_[TensorLayout::max_ndim] = TensorLayout::max_ndim - i;
  return result;
}

/** @brief Concatenates the layout strings */
DALI_HOST_DEV
constexpr TensorLayout operator+(const TensorLayout &a, const TensorLayout &b) {
  assert(a.size() + b.size() < TensorLayout::max_ndim);
  TensorLayout result = a;
  int i = 0, j = 0;
  for (i = result.ndim(), j = 0; i < TensorLayout::max_ndim && j < b.ndim(); i++, j++)
    result[i] = b[j];
  result[i] = '\0';

  /* Cannot use
   *   result.set_size(i);
   * because it's not constexpr
   */
  result.data_[TensorLayout::max_ndim] = TensorLayout::max_ndim - i;
  return result;
}


DALI_HOST_DEV
constexpr TensorLayout TensorLayout::skip(char dim_name) const noexcept {
  int i = find(dim_name);
  if (i < 0)
    return *this;
  return first(i) + sub(i+1);
}

#define DEFINE_TENSOR_LAYOUT_COMPARISON(op)                             \
inline bool operator op(const TensorLayout &tl, const std::string &s) { \
  return std::strcmp(tl.c_str(), s.c_str()) op 0;                       \
}                                                                       \
DALI_HOST_DEV                                                           \
constexpr bool operator op(const TensorLayout &tl, const char *s) {     \
  return tl.compare(s) op 0;                                            \
}                                                                       \
inline bool operator op(const std::string &s, const TensorLayout &tl) { \
  return std::strcmp(s.c_str(), tl.c_str()) op 0;                       \
}                                                                       \
DALI_HOST_DEV                                                           \
constexpr bool operator op(const char *s, const TensorLayout &tl) {     \
  return TensorLayout(s).compare(tl) op 0;                              \
}

DEFINE_TENSOR_LAYOUT_COMPARISON(<)  // NOLINT
DEFINE_TENSOR_LAYOUT_COMPARISON(>)  // NOLINT
DEFINE_TENSOR_LAYOUT_COMPARISON(<=)
DEFINE_TENSOR_LAYOUT_COMPARISON(>=)
DEFINE_TENSOR_LAYOUT_COMPARISON(==)
DEFINE_TENSOR_LAYOUT_COMPARISON(!=)

/** Provides basic functions for querying TensorLayout properties */
struct LayoutInfo {
  DALI_HOST_DEV
  static bool HasSampleDim(const TensorLayout &tl) {
    // sample dim may only appear at outermost level
    return !tl.empty() && tl[0] == 'N';
  }
  DALI_HOST_DEV
  static int DimIndex(const TensorLayout &tl, char dim_symbol) {
    return tl.find(dim_symbol);
  }
};

/**
 * @brief Provides functions for querying TensorLayout properties,
 *        assuming that the layout describes an image
 */
struct ImageLayoutInfo : LayoutInfo {
  /**
   * @brief Returns true, if the dimension name describes a spatial extent.
   *
   * Spatial dimensions are: 'D'epth, 'H'eight and 'W'idth
   */
  DALI_HOST_DEV
  static bool IsSpatialDim(char dim_name) {
    switch (dim_name) {
      case 'D':
      case 'H':
      case 'W':
        return true;
      default:
        return false;
    }
  }

  /**
   * @brief Counts spatial dimensions in the layout.
   *
   * Spatial dimensions are: 'D'epth, 'H'eight and 'W'idth
   */
  DALI_HOST_DEV
  static int NumSpatialDims(const TensorLayout &tl) {
    int s = 0;
    for (int i = 0; i < tl.ndim(); i++) {
      s += IsSpatialDim(tl[i]) ? 1 :0;
    }
    return s;
  }

  DALI_HOST_DEV
  static bool Is2D(const TensorLayout &tl) {
    return NumSpatialDims(tl) == 2;
  }

  DALI_HOST_DEV
  static bool Is3D(const TensorLayout &tl) {
    return NumSpatialDims(tl) == 3;
  }

  /** @brief Returns the index at which 'C' dimension (channel) is present or -1 if not found */
  DALI_HOST_DEV
  static int ChannelDimIndex(const TensorLayout &tl) {
    return DimIndex(tl, 'C');
  }

  /** @brief Returns true if the layout contains 'C' dimension (channel) */
  DALI_HOST_DEV
  static bool HasChannel(const TensorLayout &tl) {
    return ChannelDimIndex(tl) >= 0;
  }

  DALI_HOST_DEV
  static bool IsChannelFirst(const TensorLayout &tl) {
    return tl[0] == 'C' || (tl[0] == 'N' && tl[1] == 'C');
  }

  DALI_HOST_DEV
  static bool IsChannelLast(const TensorLayout &tl) {
    return !tl.empty() && tl[tl.ndim()-1] == 'C';
  }

  /**
   * @brief Returns true if there are at least 2 spatial dimensions in the layout
   *
   * This function returns true for 2D and 3D images and videos.
   */
  DALI_HOST_DEV
  static bool IsImage(const TensorLayout &tl) {
    return NumSpatialDims(tl) >= 2;
  }
};

/**
 * @brief Provides functions for querying TensorLayout properties,
 *        assuming that the layout describes a video
 */
struct VideoLayoutInfo : ImageLayoutInfo {
  /** @brief Returns the index of the dimension referring to frames ('F') */
  DALI_HOST_DEV
  static int FrameDimIndex(const TensorLayout &tl) {
    return DimIndex(tl, 'F');
  }

  DALI_HOST_DEV
  static bool IsChannelFirst(const TensorLayout &tl) {
    return tl[FrameDimIndex(tl)+1] == 'C';
  }

  /** @brief Returns true, if 'F' (frame) dimension is first */
  DALI_HOST_DEV
  static bool IsSequence(const TensorLayout &tl) {
    return FrameDimIndex(tl) == 0 || (HasSampleDim(tl) && FrameDimIndex(tl) == 1);
  }

  /** @brief Returns true, if 'F' (frame) dimension is present */
  DALI_HOST_DEV
  static bool HasSequence(const TensorLayout &tl) {
    return tl.contains('F');
  }

  /** @brief Returns true if the layout describes an image with 'F' (frame) dimension */
  DALI_HOST_DEV
  static bool IsVideo(const TensorLayout &tl) {
    return IsSequence(tl) && IsImage(tl);
  }

  /** @brief Returns true if the layout describes an image, but does not contain 'F' (frame) */
  DALI_HOST_DEV
  static bool IsStillImage(const TensorLayout &tl) {
    return !IsSequence(tl) && IsImage(tl);
  }

  /** @brief Removes frame dimension ('F') from the layout */
  DALI_HOST_DEV
  static TensorLayout GetFrameLayout(const TensorLayout &tl) {
    return tl.skip('F');
  }

  /** @brief Adds 'F' to the layout at the beginning or after 'N' if not already present */
  DALI_HOST_DEV
  static TensorLayout GetSequenceLayout(const TensorLayout &tl) {
    if (tl.contains('F'))
      return tl;
    if (tl[0] == 'N')
      return "NF" + tl.sub(1);
    else
      return "F" + tl;
  }
};

/**
 * @brief Calculates mapping of dimensions from given output to input.
 *
 * Array element corresponding to given output dimension is the index of that
 * dimension in the input layout:
 * `input_layout[result[i]] == output_layout[i]` for all valid i.
 *
 * Example:
 * ```
 * input_layout = "NHWC"
 * output_layout = "NCHW"
 * result = { 0, 3, 1, 2 }
 * ```
 * Explanation:
 *   - 'N' appears at index 0 in both input_layout and output_layout.
 *   - 'C' appears at index 3 in input
 *   - 'H' appears at index 1 in input
 *   - 'W' appears at index 2 in input
 *
 * If there are repetitions, the repeated names are listed in increasing order, e.g.
 * ```
 *  input_layout = "aaabb"
 *  output_layout = "baaba"
 *  result = { 3, 0, 1, 4, 2 }
 * ```
 */
template <int Dims>
inline std::array<int, Dims> GetLayoutMapping(const TensorLayout &in_layout,
                                              const TensorLayout &out_layout) {
  std::array<int, Dims> dim_map;
  for (int d = 0; d < Dims; d++) {
    dim_map[d] = d;
  }

  if (in_layout.empty() || out_layout.empty() || in_layout == out_layout)
    return dim_map;

  DALI_ENFORCE(in_layout.ndim() == Dims && out_layout.ndim() == Dims,
    "Unexpected number of dimensions in layout description");

  const char *a = out_layout.data();
  TensorLayout b = in_layout;

  int min_j = 0;
  for (int i = 0; i < Dims; i++) {
    char c = a[i];
    while (b[min_j] == 0 && min_j < Dims)
      min_j++;
    int j;
    for (j = min_j; j < Dims; j++) {
      if (b[j] == c)
        break;
    }
    if (j == Dims)
      DALI_FAIL("\"" + out_layout.str() + "\" is not a permutation of \"" + in_layout.str() + "\"");
    b[j] = '\0';  // mark as used
    dim_map[i] = j;
  }

  return dim_map;
}

inline std::ostream &operator<<(std::ostream &os, const TensorLayout &tl) {
  return os << tl.c_str();
}

inline SmallVector<int, 6> GetDimIndices(const TensorLayout &layout,
                                         const TensorLayout &dim_names) {
  SmallVector<int, 6> dims;
  dims.reserve(dim_names.size());
  for (auto dim_name : dim_names) {
    int d = layout.find(dim_name);
    DALI_ENFORCE(d >= 0, make_string("Axis '", dim_name, "' is not present in the input layout"));
    dims.push_back(d);
  }
  return dims;
}

}  // namespace dali

#endif  // DALI_CORE_TENSOR_LAYOUT_H_
