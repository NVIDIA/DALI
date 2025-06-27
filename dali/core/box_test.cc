// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/geom/box.h"

namespace dali {

namespace test {

namespace {

using BoxCoordinateType = int;
using Box_t = Box<2, BoxCoordinateType>;
static_assert(is_pod_v<Box_t>, "Box has to be POD.");

Box_t reference_box = {{3,  3},
                       {20, 15}};
Box_t empty_box = {{0, 0},
                   {0, 0}};

std::vector<Box_t> boxes = {
        {{0,  0},  {0,   0}},
        {{0,  0},  {100, 100}},
        {{4,  21}, {9,   25}},
        {{3,  3},  {20,  15}},
        {{4,  5},  {10,  9}},
        {{20, 7},  {24,  9}},
        {{17, 10}, {20,  15}},
        {{15, 12}, {22,  18}},
        {{9,  15}, {11,  19}}
};

std::vector<Box_t::corner_t> extents = {
        {0,   0},
        {100, 100},
        {5,   4},
        {17,  12},
        {6,   4},
        {4,   2},
        {3,   5},
        {7,   6},
        {2,   4}
};

std::vector<bool> contains_box = {
        false, false, false, true, true, false, true, false, false
};

std::vector<bool> contains_corner = {
        false, false, false, true, true, false, true, true, false
};

std::vector<bool> overlaps = {
        false, true, false, true, true, false, true, true, false
};

std::vector<BoxCoordinateType> volumes = {
        0, 10000, 20, 204, 24, 8, 15, 42, 8
};

std::vector<Box_t> intersections = {
        {{0,  0},  {0,  0}},
        {{3,  3},  {20, 15}},
        {{0,  0},  {0,  0}},
        {{3,  3},  {20, 15}},
        {{4,  5},  {10, 9}},
        {{0,  0},  {0,  0}},
        {{17, 10}, {20, 15}},
        {{15, 12}, {20, 15}},
        {{0,  0},  {0,  0}}
};

}  // namespace


TEST(BoxSimpleTest, operator_equal_test) {
  EXPECT_TRUE(reference_box == reference_box);
  EXPECT_FALSE(reference_box == empty_box);
}


TEST(BoxSimpleTest, empty_test) {
  EXPECT_FALSE(reference_box.empty());
  EXPECT_TRUE(empty_box.empty());
}


TEST(BoxSimpleTest, extent_test) {
  ASSERT_EQ(boxes.size(), extents.size()) << "Bad testing data";
  for (size_t i = 0; i < boxes.size(); i++) {
    EXPECT_TRUE(boxes[i].extent() == extents[i]) << "Failed at index: " << i;
  }
}


TEST(BoxSimpleTest, contains_box_test) {
  ASSERT_EQ(boxes.size(), contains_box.size()) << "Bad testing data";
  for (size_t i = 0; i < boxes.size(); i++) {
    EXPECT_EQ(reference_box.contains(boxes[i]), contains_box[i]) << "Failed at index: " << i;
  }
}


TEST(BoxSimpleTest, contains_corner_test) {
  ASSERT_EQ(boxes.size(), contains_corner.size()) << "Bad testing data";
  for (size_t i = 0; i < boxes.size(); i++) {
    EXPECT_EQ(reference_box.contains(boxes[i].lo), contains_corner[i]) << "Failed at index: " << i;
  }
}


TEST(BoxSimpleTest, overlaps_test) {
  ASSERT_EQ(boxes.size(), overlaps.size()) << "Bad testing data";
  for (size_t i = 0; i < boxes.size(); i++) {
    EXPECT_EQ(reference_box.overlaps(boxes[i]), overlaps[i]) << "Failed at index: " << i;
  }
}


TEST(BoxSimpleTest, volume_test) {
  ASSERT_EQ(boxes.size(), volumes.size()) << "Bad testing data";
  for (size_t i = 0; i < boxes.size(); i++) {
    EXPECT_EQ(volume(boxes[i]), volumes[i]) << "Failed at index: " << i;
  }
}


TEST(BoxSimpleTest, intersection_test) {
  ASSERT_EQ(boxes.size(), intersections.size()) << "Bad testing data";
  for (size_t i = 0; i < boxes.size(); i++) {
    EXPECT_EQ(intersection(reference_box, boxes[i]), intersections[i]) << "Failed at index: " << i;
  }
}

}  // namespace test


}  // namespace dali
