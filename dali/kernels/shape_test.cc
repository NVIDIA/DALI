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

#include <gtest/gtest.h>

#include "dali/kernels/shape.h"
#include "dali/kernels/tensor_view.h"

namespace dali {

TEST(TensorShapeTest, Assignement) {
  TensorShape<2> a;
  TensorShape<3> b;
  TensorShape<3> c;
  // b = a;
  // a = b;
  c = b;
  b = c;
  TensorShape<-1> d(std::vector<int64_t>{2, 42, 1});
  // d = static_cast<TensorShape<-1>>(b);
  b = TensorShape<3>(d);
  TensorShape<5> t5(d);
  std::array<int64_t, 4> x = {1, 2, 3, 4};
  TensorShape<4> f(1, 2, 3, 4);
  TensorShape<4> g(x);

}

void foo(TensorView<EmptyBackendTag, int, 4> &) {}

TEST(TensorViewTest, Assignement) {
  int tab[20];
  int *p = tab;
  TensorView<EmptyBackendTag, int, 4> int_4{p, {1, 2, 3, 4}};
  TensorView<EmptyBackendTag, int, 3> int_3{p, {1, 2, 3}};

  TensorView<EmptyBackendTag, int, 4> int_4_2{int_4};
  TensorView<EmptyBackendTag, int, -1> int_4_3{int_4};
  TensorView<EmptyBackendTag, int, 4> int_4_4{int_4_3};
  auto x = TensorView<EmptyBackendTag, int, 4>(TensorView<EmptyBackendTag, int, -1>(int_3));
  foo(x);
  // int_4_4 = int_4_3;
  // TensorView<EmptyBackendTag, int, DynamicTensorShape> int_dyn;
  // // int_4 = int_dyn;
  // int_dyn = int_4;
}

}  // namespace dali
