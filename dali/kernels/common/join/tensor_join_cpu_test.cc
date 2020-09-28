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

TEST(TensorJoinCpuTest, DetermineShapeStack) {
  std::vector<TensorShape<>> shin = {{4, 5, 7, 8},
                                     {4, 5, 7, 8}};
  TensorShape<> sh0 = {2, 4, 5, 7, 8};
  TensorShape<> sh1 = {4, 2, 5, 7, 8};
  TensorShape<> sh2 = {4, 5, 2, 7, 8};
  TensorShape<> sh3 = {4, 5, 7, 2, 8};
  TensorShape<> sh4 = {4, 5, 7, 8, 2};
  EXPECT_EQ(impl::DetermineShape<true>(make_span(shin), 0), sh0);
  EXPECT_EQ(impl::DetermineShape<true>(make_span(shin), 1), sh1);
  EXPECT_EQ(impl::DetermineShape<true>(make_span(shin), 2), sh2);
  EXPECT_EQ(impl::DetermineShape<true>(make_span(shin), 3), sh3);
  EXPECT_EQ(impl::DetermineShape<true>(make_span(shin), 4), sh4);
}


TEST(TensorJoinCpuTest, DetermineShapeConcatConstantExtent) {
  std::vector<TensorShape<>> shin = {{4, 5, 7, 8},
                                     {4, 5, 7, 8}};
  TensorShape<> sh0 = {8, 5, 7, 8};
  TensorShape<> sh1 = {4, 10, 7, 8};
  TensorShape<> sh2 = {4, 5, 14, 8};
  TensorShape<> sh3 = {4, 5, 7, 16};
  EXPECT_EQ(impl::DetermineShape<false>(make_span(shin), 0), sh0);
  EXPECT_EQ(impl::DetermineShape<false>(make_span(shin), 1), sh1);
  EXPECT_EQ(impl::DetermineShape<false>(make_span(shin), 2), sh2);
  EXPECT_EQ(impl::DetermineShape<false>(make_span(shin), 3), sh3);
}


TEST(TensorJoinCpuTest, DetermineShapeConcatVaryingExtent) {
  std::vector<TensorShape<>> shin = {{4, 5, 7, 8},
                                     {4, 5, 3, 8}};
  TensorShape<> shout = {4, 5, 10, 8};
  EXPECT_EQ(impl::DetermineShape<false>(make_span(shin), 2), shout);
}


TEST(TensorJoinCpuTest, TransferBufferTest) {
  using namespace std;  // NOLINT

  vector<vector<int>> arr = {{6, 8, 5, 1, 3, 5, 1, 6, 8, 3, 7, 5},
                             {4, 5, 1, 8, 4, 4, 1, 4, 1, 7, 6, 6}};
  vector<TensorShape<>> sh = {{3, 4},
                              {3, 4}};
  vector<TensorView<StorageCPU, const int>> in;
  for (size_t i = 0; i < arr.size(); i++) {
    in.emplace_back(arr[i].data(), sh[i]);
  }

  // Output shape for this buffer: {2, 3, 4} (STACK) or {6, 4} (CONCAT)
  vector<int> arr0 = {6, 8, 5, 1, 3, 5, 1, 6, 8, 3, 7, 5, 4, 5, 1, 8, 4, 4, 1, 4, 1, 7, 6, 6};

  // Output shape for this buffer: {3, 2, 4} (STACK) or {3, 8} (CONCAT)
  vector<int> arr1 = {6, 8, 5, 1, 4, 5, 1, 8, 3, 5, 1, 6, 4, 4, 1, 4, 8, 3, 7, 5, 1, 7, 6, 6};

  // Output shape for this buffer: {4, 3, 2} (STACK), CONCAT unavailable
  vector<int> arr2 = {6, 4, 8, 5, 5, 1, 1, 8, 3, 4, 5, 4, 1, 1, 6, 4, 8, 1, 3, 7, 7, 6, 5, 6};

  vector<vector<int>> ref_arr;
  ref_arr.emplace_back(arr0);
  ref_arr.emplace_back(arr1);
  ref_arr.emplace_back(arr2);

  for (size_t ax = 0; ax < ref_arr.size(); ax++) {
    auto outsh = impl::DetermineShape<true>(make_span(sh), ax);
    vector<int> outbuf(volume(outsh));
    impl::ConcatenateTensors({outbuf.data(), outsh}, make_span(in), ax);
    EXPECT_EQ(outbuf, ref_arr[ax]);
  }

  for (size_t ax = 0; ax < ref_arr.size() - 1; ax++) {
    auto outsh = impl::DetermineShape<false>(make_span(sh), ax);
    vector<int> outbuf(volume(outsh));
    impl::ConcatenateTensors({outbuf.data(), outsh}, make_span(in), ax);
    EXPECT_EQ(outbuf, ref_arr[ax]);
  }
}


}  // namespace test
}  // namespace kernels
}  // namespace dali
