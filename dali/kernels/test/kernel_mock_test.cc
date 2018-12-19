// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <dali/kernels/kernel.h>
#include <dali/kernels/test/test_tensors.h>
#include <gtest/gtest.h>

using std::cout;
using std::cerr;
using std::endl;

namespace dali {
namespace kernels {


template <int dim>
std::ostream &operator<<(std::ostream &os, const TensorShape<dim> &shape) {
  for (int i = 0; i < shape.size(); i++) {
    if (i) os << "x";
    os << shape[i];
  }
  return os;
}

template <int dim>
std::ostream &operator<<(std::ostream &os, const TensorListShape<dim> &shape) {
  os << "{";
  for (int i = 0; i < shape.num_samples(); i++) {
    if (i) os << ",\n ";
    os << shape[i];
  }
  os << "}";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << v.size() << "{";
  for (size_t i = 0; i < v.size(); i++) {
    if (i) os << ", ";
    os << v[i];
  }
  os << "}";
  return os;
}


template <typename Input1, typename Input2, typename Output>
struct MockKernel {
  static KernelRequirements GetRequirements(KernelContext &context, const InListCPU<Input1, 3> &i1, const InListCPU<Input2, 3> &i2, float A) {
    KernelRequirements req;
    cout << i1.shape << "\n";
    req.output_shapes = { i1.shape };
    return req;
  }

  static void Run(KernelContext &context, const OutListCPU<Output, 3>&o, const InListCPU<Input1, 3> &i1, const InListCPU<Input2, 3> &i2, float A) {
    int samples = i1.num_samples();
    cout << "Samples: " << samples << endl;
    for (int n = 0; n < samples; n++) {
      auto tshape = i1.shape.tensor_shape_span(n);
      int d = tshape[0];
      int h = tshape[1];
      int w = tshape[2];
      cout << "sample #" << n << " is " << d << "x" << h << "x" << w << endl;

      auto t1 = i1[n];
      auto t2 = i2[n];
      auto to = o[n];

      for (int z = 0; z < d; z++) {
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            *to(z, y, z) = *t1(z, y, z) * A + *t2(z, y, x);
          }
        }
      }
    }
  }
};

template <typename Input1, typename Input2, typename Output>
void RunTest()
{
  MockKernel<Input1, Input2, Output> K;

  InListCPU<Input1, 3> i1;
  InListCPU<Input2, 3> i2;
  OutListCPU<Output, 3> o;

  KernelContext ctx;

  TestTensorList<Input1> tl1;
  TestTensorList<Input2> tl2;
  TestTensorList<Output> tlo;

  std::vector<TensorShape<3>> shapes;
  for (int i=0; i<3; i++)
    shapes.push_back({ rand()%3+1, rand()%512+512, rand()%512+512});
  TensorListShape<3> list_shape(shapes);
  cout << list_shape << endl;

  tl1.reshape(list_shape);
  tl2.reshape(list_shape);

  cout << i1.shape << " " << i2.shape << endl;

  i1 = tl1.template cpu<3>();
  i2 = tl2.template cpu<3>();

  cout << "GetReq" << endl;
  auto req = K.GetRequirements(ctx, i1, i2, 0.5f);
  ASSERT_EQ(req.output_shapes.size(), 1);
  tlo.reshape(req.output_shapes[0]);
  o = tlo.template cpu<3>();

  cout << "Run" << endl;
  K.Run(ctx, o, i1, i2, 0.5f);

}

TEST(KernelMock, Test1) {
  RunTest<float, float, float>();
}

}  // namespace kernels
}  // namespace dali
