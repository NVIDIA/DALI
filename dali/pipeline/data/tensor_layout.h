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

#ifndef DALI_PIPELINE_DATA_TENSOR_LAYOUT_H_
#define DALI_PIPELINE_DATA_TENSOR_LAYOUT_H_

#include <memory>
#include <string>
#include <cstring>
#include "dali/core/common.h"
#include "dali/core/small_vector.h"

namespace dali {

struct TensorLayout {
  TensorLayout() = default;

  TensorLayout(const char *str) {  // NOLINT
    data_.resize(std::strlen(str) + 1);
    for (size_t i = 0; i < data_.size(); i++)
      data_[i] = str[i];
  }

  TensorLayout(const char *str, size_t n) {  // NOLINT
    data_.resize(n + 1);
    for (size_t i = 0; i < n; i++)
      data_[i] = str[i];
    data_[n] = 0;
  }

  template <size_t N>
  TensorLayout(const char (&s)[N]) {  // NOLINT
    size_t n = N && s[N-1] == '\0' ? N - 1 : N;
    data_.resize(n + 1);
    for (size_t i = 0; i < n; i++)
      data_[i] = s[i];
    data_[n] = 0;
  }

  TensorLayout(const std::string &s) : TensorLayout(s.data(), s.length()) {  // NOLINT
  }

  char &operator[](int d) {
    return data_[d];
  }
  const char &operator[](int d) const {
    return data_[d];
  }

  const char *c_str() const noexcept {
    return data_.data();
  }
  const char *data() const noexcept {
    return data_.data();
  }
  std::string str() const { return c_str(); }

  int find(char dim_name) const noexcept {
    for (int i = 0; i < ndim(); i++) {
      if (data_[i] == dim_name)
        return i;
    }
    return -1;
  }

  bool contains(char dim_name) const noexcept {
    return find(dim_name) >= 0;
  }

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
    return data_ == tl.data_;
  }
  bool operator!=(const TensorLayout &tl) const noexcept {
    return data_ != tl.data_;
  }

  size_t size() const noexcept { return data_.size() - 1; }
  int ndim() const noexcept { return size(); }
  bool empty() const noexcept { return size() == 0; }

  auto begin() { return data_.begin(); }
  auto end() { return data_.end() - 1; }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end() - 1; }
  auto cbegin() { return data_.cbegin(); }
  auto cend() { return data_.cend() - 1; }

  // This is the size that the SmallVector would take anyway
  static constexpr size_t static_capacity = sizeof(char*)+sizeof(size_t);

  /// @brief Stores the dimension descriptions as a null-terminated string
  SmallVector<char, static_capacity> data_ = { '\0' };
};

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

struct LayoutInfo {
  static int HasSampleDim(const TensorLayout &tl) {
    // sample dim may only appear at outermost level
    return !tl.empty() && tl[0] == 'N';
  }
  static int DimIndex(const TensorLayout &tl, char dim_symbol) {
    return tl.find(dim_symbol);
  }
};

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
};

struct VideoLayoutInfo : ImageLayoutInfo {
  static int FrameDim(const TensorLayout &tl) {
    return DimIndex(tl, 'F');
  }
  static bool IsSequence(const TensorLayout &tl) {
    return tl.contains('F');
  }
};

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_LAYOUT_H_

