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

#include <gtest/gtest.h>
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

using std::cout;

namespace dali {
namespace kernels {

TEST(TestUtil, RandomFillTensor) {
  const int W = 100;
  const int H = 100;
  const int N = W*H;
  std::vector<float> memory(N);
  TensorView<StorageCPU, float, 3> view(memory.data(), { 1, W, H });

  std::mt19937_64 rng;
  UniformRandomFill(view, rng, 0, 1);
  float sum = 0;
  int64_t n = view.num_elements();
  EXPECT_EQ(n, N);
  for (int64_t i = 0; i < n; i++) {
    sum += memory[i];
  }
  sum /= n;
  EXPECT_LE(std::abs(sum - 0.5f), 0.01f) << "Mean should be close to 0.5; actual " << sum;
}

TEST(TestUtil, RandomFillList) {
  const int D1 = 2;
  const int W1 = 20;
  const int H1 = 30;
  const int D2 = 3;
  const int W2 = 40;
  const int H2 = 33;
  const int N = D1*W1*H1 + D2*W2*H2;
  std::vector<float> memory(N);
  TensorListView<StorageCPU, float, 3> view(memory.data(), { { D1, W1, H1 }, { D2, W2, H2 } });

  std::mt19937_64 rng;
  UniformRandomFill(view, rng, 0, 1);
  float sum = 0;
  int64_t n = view.num_elements();
  EXPECT_EQ(n, N);
  for (int64_t i = 0; i < n; i++) {
    sum += memory[i];
  }
  sum /= n;
  EXPECT_LE(std::abs(sum - 0.5f), 0.01f) << "Mean should be close to 0.5; actual " << sum;
}

TEST(TestUtil, StatefulGenerator) {
  int mem[1024];
  auto view = make_tensor_cpu<2>(mem, { 32, 32 });
  int num = 1;
  auto seq_gen = [&]() { return num++; };
  Fill(view, seq_gen);
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(mem[i], i+1);
  }
}

TEST(TestTensorList, Transfer) {
  TestTensorList<int, 1> ttl;
  ttl.reshape({{{1024}}});

  int num = 1;
  auto seq_gen = [&]() { return num++; };
  Fill(ttl.cpu(0), seq_gen);

  auto *ptr = ttl.cpu(0).data[0];

  auto gpu = ttl.gpu(0);
  EXPECT_FALSE(gpu.empty());

  ttl.invalidate_cpu();

  auto cpu = ttl.cpu(0);
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(cpu.data[0][i], i+1);
  }
}

}  // namespace kernels
}  // namespace dali
