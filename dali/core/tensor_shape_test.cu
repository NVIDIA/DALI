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

#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/test/device_test.h"

namespace dali {
template <int N>
__device__ DeviceString dev_to_string(const dali::TensorShape<N> &s) {
  DeviceString result;
  for (int i = 0; i < N; i++) {
    if (i)
      result += "x";
    result += dev_to_string(s[i]);
  }
  return result;
}
}  // namespace dali

DEVICE_TEST(TensorShapeDev, Construct, 1, 1) {
  dali::TensorShape<3> a = { 4, 5, 6 };
  DEV_EXPECT_EQ(a[0], 4);
  DEV_EXPECT_EQ(a[1], 5);
  DEV_EXPECT_EQ(a[2], 6);
  DEV_EXPECT_EQ(dali::volume(a), 4*5*6);

  dali::TensorShape<4> b;
  DEV_EXPECT_EQ(b[0], 0);
  DEV_EXPECT_EQ(b[1], 0);
  DEV_EXPECT_EQ(b[2], 0);
  DEV_EXPECT_EQ(b[3], 0);
}


DEVICE_TEST(TensorShapeDev, Copy, 1, 1) {
  dali::TensorShape<3> a = { 4, 5, 6 };
  dali::TensorShape<3> b = a, c;
  c = b;
  DEV_EXPECT_EQ(a, b);
  DEV_EXPECT_EQ(a, c);
}

DEVICE_TEST(TensorShapeDev, Move, 1, 1) {
  dali::TensorShape<3> a = { 4, 5, 6 };
  dali::TensorShape<3> ref = a;
  dali::TensorShape<3> c = dali::cuda_move(a), d;
  DEV_EXPECT_EQ(c, ref);
  d = dali::cuda_move(c);
  DEV_EXPECT_EQ(d, ref);
}

DEVICE_TEST(TensorShapeDev, ForEach, 1, 1) {
  dali::TensorShape<3> a = { 4, 5, 6 };
  int64_t arr[3];
  int i = 0;
  for (auto v : a) {
    DEV_ASSERT_LT(i, 3);
    arr[i++] = v;
  }
  DEV_EXPECT_EQ(i, 3);
  DEV_EXPECT_EQ(arr[0], 4);
  DEV_EXPECT_EQ(arr[1], 5);
  DEV_EXPECT_EQ(arr[2], 6);
}

DEVICE_TEST(TensorShapeDev, FirstLast, 1, 1) {
  dali::TensorShape<5> a = { 11, 22, 33, 44, 55 };
  auto first = a.first<3>();
  auto last = a.last<4>();
  static_assert(std::is_same<decltype(first), dali::TensorShape<3>>::value,
    "Wrong type inferred for first()");
  static_assert(std::is_same<decltype(last), dali::TensorShape<4>>::value,
    "Wrong type inferred for last()");
  DEV_EXPECT_EQ(first[0], a[0]);
  DEV_EXPECT_EQ(first[1], a[1]);
  DEV_EXPECT_EQ(first[2], a[2]);

  DEV_EXPECT_EQ(last[0], a[1]);
  DEV_EXPECT_EQ(last[1], a[2]);
  DEV_EXPECT_EQ(last[2], a[3]);
  DEV_EXPECT_EQ(last[3], a[4]);
}

DEFINE_TEST_KERNEL(TensorShapeDev, KernelArg, dali::TensorShape<4> a) {
  DEV_EXPECT_EQ(a[threadIdx.x], threadIdx.x + 2);
}

TEST(TensorShapeDev, KernelArg) {
  dali::TensorShape<4> a = { 2, 3, 4, 5 };
  DEVICE_TEST_CASE_BODY(TensorShapeDev, KernelArg, 1, 4, a);
}


TEST(TensorShapeDev, ConstructorFromIntegerCollectionHost) {
  using dali::TensorShape;
  using dali::DeviceArray;
  TensorShape<5> ref{1, 2, 3, 4, 5};
  DeviceArray<int, 5> vec_i{1, 2, 3, 4, 5};

  EXPECT_EQ(ref, TensorShape<5>(vec_i));
  EXPECT_EQ(ref, TensorShape<5>(vec_i.begin(), vec_i.end()));
  TensorShape<5> sh;
  sh = vec_i;
  EXPECT_EQ(ref, sh);
}

DEVICE_TEST(TensorShapeDev, ConstructorFromIntegerCollectionDevice, 1, 1) {
  using dali::TensorShape;
  using dali::DeviceArray;
  TensorShape<5> ref{1, 2, 3, 4, 5};
  DeviceArray<int, 5> vec_i{1, 2, 3, 4, 5};

  DEV_EXPECT_EQ(ref, TensorShape<5>(vec_i));
  DEV_EXPECT_EQ(ref, TensorShape<5>(vec_i.begin(), vec_i.end()));
  TensorShape<5> sh;
  sh = vec_i;
  DEV_EXPECT_EQ(ref, sh);
}
