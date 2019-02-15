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
#include <array>
#include "dali/kernels/dev_array.h"

namespace dali {
namespace kernels {

static_assert(sizeof(DeviceArray<int, 41>) == sizeof(int[41]),
  "Size of DeviceArray must match exactly the size of equally-sized C array");
static_assert(DeviceArray<float, 123>().size() == 123,
  "DeviceArray<T, N>::size() must return N");

TEST(DeviceArray, StdArray) {
  const int N = 42;

  std::array<int, N> arr;
  for (int i = 0; i < N; i++)
    arr[i] = 3*i + 5;

  DeviceArray<int, N> devarr = arr;
  std::array<int , N> arr2 = devarr;
  EXPECT_EQ(arr2, arr);
}

TEST(DeviceArray, Indexing) {
  const size_t N = 42;

  std::array<int, N> arr;
  DeviceArray<int, N> devarr;
  const auto &cdevarr = devarr;
  for (size_t i = 0; i < N; i++)
    arr[i] = devarr[i] = 5*i + 3;

  EXPECT_EQ(devarr.size(), N);

  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(devarr[i], arr[i]);
    EXPECT_EQ(cdevarr[i], arr[i]);
  }
}

TEST(DeviceArray, Iteration) {
  const int N = 42;

  std::array<int, N> arr;
  DeviceArray<int, N> devarr;
  const auto &cdevarr = devarr;
  for (int i = 0; i < N; i++)
    arr[i] = devarr[i] = 5*i + 3;

  EXPECT_EQ(devarr.begin(), cdevarr.begin());
  EXPECT_EQ(devarr.cbegin(), cdevarr.begin());
  EXPECT_EQ(devarr.end(), cdevarr.end());
  EXPECT_EQ(devarr.cend(), cdevarr.end());
  EXPECT_EQ(devarr.end() - devarr.begin(), N);
  EXPECT_EQ(&devarr[0], &*devarr.begin());
  EXPECT_EQ(&devarr[0], devarr.data());

  int i = 0;
  for (auto &x : devarr) {
    EXPECT_EQ(x, arr[i++]);
  }

  i = 0;
  for (auto &x : cdevarr) {
    EXPECT_EQ(x, arr[i++]);
  }
}

TEST(DeviceArray, Comparison) {
  const int N = 31;
  DeviceArray<float, N> arr1, arr2;
  for (int i = 0; i < N; i++)
    arr1[i] = 0.5f*i + 3.25f;
  arr2 = arr1;

  EXPECT_TRUE(arr1 == arr2);
  EXPECT_FALSE(arr1 != arr2);

  for (int i = 0; i < N; i++) {
    arr2[i] = 0;
    EXPECT_FALSE(arr1 == arr2);
    EXPECT_TRUE(arr1 != arr2);
    arr2[i] = arr1[i];
    EXPECT_TRUE(arr1 == arr2);
  }
}

}  // namespace kernels
}  // namespace dali
