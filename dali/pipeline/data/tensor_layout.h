// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <cstring>
#include "dali/core/small_vector.h"

namespace dali {

struct TensorLayout {
  enum {
    Dim_Sample = 'N',
    Dim_Depth = 'D',
    Dim_Height = 'H',
    Dim_Width = 'W',
    Dim_Channel = 'C',
    Dim_Frame = 'F',
    Dim_Frequency = 'Q',
    Dim_Time = 'T',
  };

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

}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TENSOR_LAYOUT_H_

