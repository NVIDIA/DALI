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

#ifndef DALI_OPERATORS_SEGMENTATION_UTILS_SEARCHABLE_RLE_MASK_H_
#define DALI_OPERATORS_SEGMENTATION_UTILS_SEARCHABLE_RLE_MASK_H_

#include <algorithm>
#include <utility>
#include <vector>
#include "dali/core/tensor_view.h"
#include "dali/core/span.h"

namespace dali {

/**
 * @brief Encodes a segmentation mask in a way that is easy to access to
 *        know how many foreground pixels are there and get the flat index
 *        to an arbitrary i-th foreground pixel, as they appear in the mask.
 */
class SearchableRLEMask {
 public:
  struct Group {
    int64_t ith;    // The group begins with the i-th foreground pixel
    int64_t start;  // The group starts at this flat index
  };

  struct is_positive {
    template <typename T>
    bool operator()(const T &value) const { return value > 0; }
  };

  void Clear() {
    groups_.clear();
    count_ = 0;
  }

  /**
   * @brief Construct a searchable RLE mask. ``predicate`` is used to
   *        determine the mask values that are considered foreground
   */
  template <typename T, typename Predicate = is_positive>
  void Init(span<const T> mask_view, Predicate &&is_foreground = {}) {
    Clear();
    int64_t idx = 0;
    int64_t sz = mask_view.size();
    while (idx < sz) {
      while (idx < sz && !is_foreground(mask_view[idx])) {
        idx++;
      }
      if (idx < sz) {  // found a foreground pixel
        groups_.push_back({count_++, idx++});
        while (idx < sz && is_foreground(mask_view[idx])) {
          idx++;
          count_++;
        }
      }
    }
  }

  template <typename T, typename Predicate = is_positive>
  void Init(TensorView<StorageCPU, T> mask_view, Predicate &&is_foreground = {}) {
    Init(span<const T>{mask_view.data, volume(mask_view.shape)},
         std::forward<Predicate>(is_foreground));
  }

  /**
   * @brief Returns the position of the i-th foreground pixel.
   *        If ith is an invalid index, -1 is returned
   */
  int64_t find(int64_t ith) {
    if (ith < 0 || ith >= count_) {
      return -1;
    }
    auto it = std::upper_bound(groups_.begin(), groups_.end(), ith,
                               [](int64_t x, const Group &g) { return x < g.ith; }) - 1;
    return it->start + (ith - it->ith);
  }

  /**
   * @brief returns the number of foreground pixels in the mask
   */
  int64_t count() const {
    return count_;
  }

  /** 
   * @brief returns the internal RLE representation of the mask
   */
  span<const Group> encoded() const {
    return make_cspan(groups_);
  }

 private:
  int64_t count_ = 0;
  std::vector<Group> groups_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEGMENTATION_UTILS_SEARCHABLE_RLE_MASK_H_
