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

#include <gtest/gtest.h>
#include "dali/operators/segmentation/utils/searchable_rle_mask.h"

namespace dali {

TEST(SearchableRLEMask, tests) {
  uint8_t mask[] = {
    0, 0, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0};
  TensorView<StorageCPU, uint8_t> mask_view(mask, TensorShape<>{6, 5});
  SearchableRLEMask search_mask(mask_view);
  ASSERT_EQ(7, search_mask.count());

  auto rle = search_mask.encoded();
  ASSERT_EQ(4, rle.size());  // 4 groups of foreground pixels

  // First group has 1 pixel
  ASSERT_EQ(0, rle[0].ith);    // First group starts with the 0-th foreground pixel
  ASSERT_EQ(2, rle[0].start);  // Starting at position 2

  // Second group has 2 pixels
  ASSERT_EQ(1, rle[1].ith);    // Second group starts with the 1-th foreground pixel
  ASSERT_EQ(8, rle[1].start);  // Starting at position 8

  // Third group has 2 pixels
  ASSERT_EQ(3,  rle[2].ith);    // First group starts with the 3-th foreground pixel
  ASSERT_EQ(13, rle[2].start);  // Starting at position 13

  // Fourth group has 2 pixels
  ASSERT_EQ(5,  rle[3].ith);    // First group starts with the 5-th foreground pixel
  ASSERT_EQ(20, rle[3].start);  // Starting at position 20

  ASSERT_EQ(-1, search_mask.find(7));  // doesn't exist
  ASSERT_EQ(-1, search_mask.find(-1));  // doesn't exist
  ASSERT_EQ(2, search_mask.find(0));  // first of first group
  ASSERT_EQ(8, search_mask.find(1));  // first of second group
  ASSERT_EQ(9, search_mask.find(2));  // second of second group
  ASSERT_EQ(14, search_mask.find(4));  // 4-th pixel is at position 14

  std::vector<float> all_bg(10, 0.0f);
  SearchableRLEMask all_bg_mask(make_cspan(all_bg), 0.0f);
  ASSERT_EQ(0, all_bg_mask.count());
  ASSERT_EQ(-1, all_bg_mask.find(0));

  std::vector<float> all_fg(10, 1.0f);
  SearchableRLEMask all_fg_mask(make_cspan(all_fg), 0.0f);
  ASSERT_EQ(all_fg.size(), all_fg_mask.count());
  for (size_t i = 0; i < all_fg.size(); i++)
    ASSERT_EQ(i, all_fg_mask.find(i));

  std::vector<float> pattern(10, 0.0f);
  for (size_t i = 1; i < pattern.size(); i+=2)
    pattern[i] = 1.0f;
  SearchableRLEMask pattern_mask(make_cspan(pattern), 0.0f);
  ASSERT_EQ(pattern.size() / 2, pattern_mask.count());
  for (int i = 0; i < pattern_mask.count(); i++)
    ASSERT_EQ(2 * i + 1, pattern_mask.find(i));
}

}  // namespace dali
