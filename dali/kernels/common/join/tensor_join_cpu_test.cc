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
#include <vector>
#include "dali/kernels/common/join/tensor_join_cpu.h"

namespace dali {
namespace kernels {
namespace test {

TEST(TensorJoinCpuTest, JoinedShapeStack) {
  std::vector<TensorShape<>> shin = {{4, 5, 7, 8},
                                     {4, 5, 7, 8}};
  TensorShape<> sh0 = {2, 4, 5, 7, 8};
  TensorShape<> sh1 = {4, 2, 5, 7, 8};
  TensorShape<> sh2 = {4, 5, 2, 7, 8};
  TensorShape<> sh3 = {4, 5, 7, 2, 8};
  TensorShape<> sh4 = {4, 5, 7, 8, 2};
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 0, true), sh0);
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 1, true), sh1);
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 2, true), sh2);
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 3, true), sh3);
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 4, true), sh4);
}


TEST(TensorJoinCpuTest, JoinedShapeConcatConstantExtent) {
  std::vector<TensorShape<>> shin = {{4, 5, 7, 8},
                                     {4, 5, 7, 8}};
  TensorShape<> sh0 = {8, 5, 7, 8};
  TensorShape<> sh1 = {4, 10, 7, 8};
  TensorShape<> sh2 = {4, 5, 14, 8};
  TensorShape<> sh3 = {4, 5, 7, 16};
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 0, false), sh0);
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 1, false), sh1);
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 2, false), sh2);
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 3, false), sh3);
}


TEST(TensorJoinCpuTest, JoinedShapeConcatVaryingExtent) {
  std::vector<TensorShape<>> shin = {{4, 5, 7, 8},
                                     {4, 5, 3, 8}};
  TensorShape<> shout = {4, 5, 10, 8};
  EXPECT_EQ(tensor_join::JoinedShape(make_span(shin), 2, false), shout);
}


TEST(TensorJoinCpuTest, ConcatenateTensorsTest) {
  using std::vector;

  vector<vector<int>> arr = {{6, 8, 5, 1,
                              3, 5, 1, 6,
                              8, 3, 7, 5},
                             {4, 5, 1, 8,
                              4, 4, 1, 4,
                              1, 7, 6, 6}};
  vector<TensorShape<>> sh = {{3, 4},
                              {3, 4}};
  vector<TensorView<StorageCPU, const int>> in;
  for (size_t i = 0; i < arr.size(); i++) {
    in.emplace_back(arr[i].data(), sh[i]);
  }

  // Output shape for this buffer: {2, 3, 4} (STACK) or {6, 4} (CONCAT)
  vector<int> arr0 = {6, 8, 5, 1,
                      3, 5, 1, 6,
                      8, 3, 7, 5,
                      // join
                      4, 5, 1, 8,
                      4, 4, 1, 4,
                      1, 7, 6, 6};

  // Output shape for this buffer: {3, 2, 4} (STACK) or {3, 8} (CONCAT)
  vector<int> arr1 = {6, 8, 5, 1,  // join
                      4, 5, 1, 8,

                      3, 5, 1, 6,  // join
                      4, 4, 1, 4,

                      8, 3, 7, 5,  // join
                      1, 7, 6, 6};

  // Output shape for this buffer: {3, 4, 2} (STACK), CONCAT unavailable

  //                    v------v------v------v---- join
  vector<int> arr2 = {6, 4,  8, 5,  5, 1,  1, 8,
                      3, 4,  5, 4,  1, 1,  6, 4,
                      8, 1,  3, 7,  7, 6,  5, 6};

  vector<vector<int>> ref_arr;
  ref_arr.emplace_back(arr0);
  ref_arr.emplace_back(arr1);
  ref_arr.emplace_back(arr2);

  for (size_t ax = 0; ax < ref_arr.size(); ax++) {
    auto outsh = tensor_join::JoinedShape(make_span(sh), ax, true);
    vector<int> outbuf(volume(outsh));
    ASSERT_EQ(volume(outsh), ref_arr[ax].size());
    tensor_join::ConcatenateTensors({outbuf.data(), outsh}, make_cspan(in), ax);
    EXPECT_EQ(outbuf, ref_arr[ax]);
  }

  for (size_t ax = 0; ax < ref_arr.size() - 1; ax++) {
    auto outsh = tensor_join::JoinedShape(make_span(sh), ax, false);
    vector<int> outbuf(volume(outsh));
    ASSERT_EQ(volume(outsh), ref_arr[ax].size());
    tensor_join::ConcatenateTensors({outbuf.data(), outsh}, make_cspan(in), ax);
    EXPECT_EQ(outbuf, ref_arr[ax]);
  }
}


template<template<typename, int = -1> class Kernel, typename T>
void
KernelRunAndVerify(TensorView<StorageCPU, T> ref, span<const TensorView<StorageCPU, const T>> in,
                   int axis) {
  Kernel<T> kernel;
  KernelContext c;
  auto kr = kernel.Setup(c, in, axis);
  auto outsh = kr.output_shapes[0].tensor_shape(0);
  std::vector<T> outbuf(volume(outsh));
  TensorView<StorageCPU, T> out{outbuf.data(), outsh};
  kernel.Run(c, out, in);
  ASSERT_EQ(out.shape, ref.shape);
  if (std::is_integral<T>::value) {
    for (int i = 0; i < volume(ref.shape); i++) {
      EXPECT_EQ(out.data[i], ref.data[i]);
    }
  } else {
    for (int i = 0; i < volume(ref.shape); i++) {
      EXPECT_FLOAT_EQ(out.data[i], ref.data[i]);
    }
  }
}


TEST(TensorJoinCpuTest, StackKernelTest) {
  using std::vector;
  vector<vector<int>> arr = {{100, 101, 102,
                              110, 111, 112,
                              120, 121, 122,
                              130, 131, 132},
                             {200, 201, 202,
                              210, 211, 212,
                              220, 221, 222,
                              230, 231, 232}};
  vector<TensorShape<>> sh = {{4, 3},
                              {4, 3}};
  vector<TensorView<StorageCPU, const int>> in;
  for (size_t i = 0; i < arr.size(); i++) {
    in.emplace_back(arr[i].data(), sh[i]);
  }

  vector<int> arr_st0 = {100, 101, 102,
                         110, 111, 112,
                         120, 121, 122,
                         130, 131, 132,
                         // join
                         200, 201, 202,
                         210, 211, 212,
                         220, 221, 222,
                         230, 231, 232};
  TensorShape<> sh_st0 = {2, 4, 3};

  vector<int> arr_st1 = {100, 101, 102,  // join
                         200, 201, 202,

                         110, 111, 112,  // join
                         210, 211, 212,

                         120, 121, 122,  // join
                         220, 221, 222,

                         130, 131, 132,  // join
                         230, 231, 232};
  TensorShape<> sh_st1 = {4, 2, 3};

  //                         v-----------v-----------v------ join
  vector<int> arr_st2 = {100, 200,   101, 201,   102, 202,
                         110, 210,   111, 211,   112, 212,
                         120, 220,   121, 221,   122, 222,
                         130, 230,   131, 231,   132, 232};
  TensorShape<> sh_st2 = {4, 3, 2};

  KernelRunAndVerify<TensorStackCPU>({arr_st0.data(), sh_st0}, make_cspan(in), 0);
  KernelRunAndVerify<TensorStackCPU>({arr_st1.data(), sh_st1}, make_cspan(in), 1);
  KernelRunAndVerify<TensorStackCPU>({arr_st2.data(), sh_st2}, make_cspan(in), 2);
  ASSERT_ANY_THROW(KernelRunAndVerify<TensorStackCPU>({arr_st2.data(), sh_st2}, make_cspan(in), 3));
}


TEST(TensorJoinCpuTest, ConcatKernelTest) {
  using std::vector;
  vector<vector<int>> arr = {{100, 101, 102,
                              110, 111, 112,
                              120, 121, 122,
                              130, 131, 132},
                             {200, 201, 202,
                              210, 211, 212,
                              220, 221, 222,
                              230, 231, 232}};
  vector<TensorShape<>> sh = {{4, 3},
                              {4, 3}};
  vector<TensorView<StorageCPU, const int>> in;
  for (size_t i = 0; i < arr.size(); i++) {
    in.emplace_back(arr[i].data(), sh[i]);
  }

  vector<int> arr_cat0 = {100, 101, 102,
                          110, 111, 112,
                          120, 121, 122,
                          130, 131, 132,
                          // join
                          200, 201, 202,
                          210, 211, 212,
                          220, 221, 222,
                          230, 231, 232};
  TensorShape<> sh_cat0 = {8, 3};

  //                                    v--- join
  vector<int> arr_cat1 = {100, 101, 102,  200, 201, 202,
                          110, 111, 112,  210, 211, 212,
                          120, 121, 122,  220, 221, 222,
                          130, 131, 132,  230, 231, 232};
  TensorShape<> sh_cat1 = {4, 6};

  KernelRunAndVerify<TensorConcatCPU>({arr_cat0.data(), sh_cat0}, make_cspan(in), 0);
  KernelRunAndVerify<TensorConcatCPU>({arr_cat1.data(), sh_cat1}, make_cspan(in), 1);
  ASSERT_ANY_THROW(
          KernelRunAndVerify<TensorConcatCPU>({arr_cat0.data(), sh_cat0}, make_cspan(in), 2));
}


TEST(TensorJoinCpuTest, ConcatKernelDifferentExtentTest) {
  using std::vector;
  vector<vector<int>> arr = {{100, 101, 102,
                              110, 111, 112,
                              120, 121, 122,
                              130, 131, 132},
                             {200, 201, 202,
                              210, 211, 212,
                              220, 221, 222}};
  vector<TensorShape<>> sh = {{4, 3},
                              {3, 3}};
  vector<TensorView<StorageCPU, const int>> in;
  for (size_t i = 0; i < arr.size(); i++) {
    in.emplace_back(arr[i].data(), sh[i]);
  }

  vector<int> arr_cat0 = {100, 101, 102,
                          110, 111, 112,
                          120, 121, 122,
                          130, 131, 132,
                          // join
                          200, 201, 202,
                          210, 211, 212,
                          220, 221, 222};
  TensorShape<> sh_cat0 = {7, 3};

  KernelRunAndVerify<TensorConcatCPU>({arr_cat0.data(), sh_cat0}, make_cspan(in), 0);
  ASSERT_ANY_THROW(
          KernelRunAndVerify<TensorConcatCPU>({arr_cat0.data(), sh_cat0}, make_cspan(in), 1));
}


}  // namespace test
}  // namespace kernels
}  // namespace dali
