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

#include <gtest/gtest.h>
#include <list>
#include "dali/kernels/util.h"
#include "dali/kernels/tensor_shape.h"

namespace dali {

static_assert(std::is_same<volume_t<uint8_t>, int>::value, "Expected promotion to `int`");
static_assert(std::is_same<volume_t<int16_t>, int>::value, "Expected promotion to `int`");
static_assert(std::is_same<volume_t<int32_t>, int64_t>::value,
  "Expected promotion to `int64_t`");
static_assert(std::is_same<volume_t<uint32_t>, uint64_t>::value,
  "Expected promotion to `uint64_t`");
static_assert(std::is_same<volume_t<float>, float>::value,
  "Floating-point volume should keep extent type");
static_assert(std::is_same<volume_t<double>, double>::value,
  "Floating-point volume should keep extent type");

using kernels::TensorShape;

TEST(Volume, Scalar) {
  EXPECT_EQ(volume(5), 5);
  EXPECT_EQ(volume(12.25f), 12.25f);
  EXPECT_EQ(volume(0.625), 0.625);
}

TEST(Volume, Collections) {
  int c_array[] = { 5, 7, 4 };
  std::array<int, 4> cpp_array = { 2, 4, 6, 8 };
  EXPECT_EQ(volume(c_array), 5*7*4);
  EXPECT_EQ(volume(cpp_array), 2*4*6*8);
  std::list<uint8_t> list = { 3, 5, 7, 9 };
  EXPECT_EQ(volume(list), 3*5*7*9);
  TensorShape<3> ts = { 11, 22, 33 };
  EXPECT_EQ(volume(ts), 11*22*33);
  // initializer list
  EXPECT_EQ(volume({4, 9, 2, 3}), 4*9*2*3);
}

TEST(Volume, Ranges) {
  int c_array[] = { 5, 7, 4 };
  std::array<int, 4> cpp_array = { 2, 4, 6, 8 };
  EXPECT_EQ(volume(c_array+1, c_array+3), 7*4);
  EXPECT_EQ(volume(cpp_array.begin()+1, cpp_array.end()-1), 4*6);
  std::list<uint8_t> list = { 3, 5, 7, 9 };
  EXPECT_EQ(volume(std::next(list.begin()), std::prev(list.end())), 5*7);

  std::initializer_list<int> il = {1, 2, 4, 9, 2, 3, 9, 10};
  auto b = il.begin();
  auto e = il.end();
  EXPECT_EQ(volume(b + 2, e - 2), 4*9*2*3);
}

}  // namespace dali
