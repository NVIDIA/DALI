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
#include "dali/core/span.h"
#include "dali/kernels/alloc.h"
#include "dali/core/util.h"

namespace dali {
namespace kernels {
namespace {

template <typename T, span_extent_t extent>
__global__ void TestSpanKernel(span<T, extent> span) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  if (x < span.size()) {
    span[x] += 1 + x;
  }
}

}  // namespace

inline void Validate(span<const int> s) {
  int i = 1;
  for (auto a : s)
    EXPECT_EQ(a, i++);
  EXPECT_EQ(i, static_cast<int>(s.size() + 1));
}

TEST(Span, Convert) {
  int A[10];
  auto s = make_span(A);
  int i = 1;
  for (auto &a : s)
    a = i++;
  Validate(s);
}

TEST(TestGPUSpan, Test1) {
  const int N = 1000;
  int array[N], out[N];
  for (int i = 0; i < N; i++) {
    array[i] = i + 1;
  }

  dim3 block(32);
  dim3 grid = div_ceil(N, block.x);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  auto gpumem = memory::alloc_unique<int>(AllocType::GPU, N);
  cudaMemcpy(gpumem.get(), array, sizeof(array), cudaMemcpyHostToDevice);
  span<int> dyn_span = { gpumem.get(), N };
  TestSpanKernel<<<grid, block>>>(dyn_span);
  cudaMemcpy(out, gpumem.get(), sizeof(array), cudaMemcpyDeviceToHost);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(out[i], array[i] + 1 + i);
  }

  cudaMemcpy(gpumem.get(), array, sizeof(array), cudaMemcpyHostToDevice);
  span<int, N> static_span = { gpumem.get() };
  TestSpanKernel<<<grid, block>>>(static_span);
  cudaMemcpy(out, gpumem.get(), sizeof(array), cudaMemcpyDeviceToHost);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(out[i], array[i] + 1 + i);
  }
}

}  // namespace kernels
}  // namespace dali
