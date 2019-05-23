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

#include "dali/test/device_test.h"
#include "dali/core/util.h"
#include "dali/core/dev_array.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"

namespace dali {

DEVICE_TEST(CoreUtilsDev, Volume, 1, 1) {
  int a0[] = { 42 };
  DEV_EXPECT_EQ(volume(a0), 42);
  int a1[] = { 2, 3, 4 };
  DEV_EXPECT_EQ(volume(a1), 2*3*4);
  DeviceArray<int, 2> b = { 10000000, 10000000 };
  DEV_EXPECT_EQ(volume(b), 100000000000000LL);
}

DEVICE_TEST(CoreUtilsDev, Size, 1, 1) {
  int a0[] = { 42 };
  DEV_EXPECT_EQ(size(a0), 1);
  int a1[] = { 2, 3, 4 };
  DEV_EXPECT_EQ(size(a1), 3);

  SmallVector<int, 5> v;
  v.resize(10);
  DEV_EXPECT_EQ(v.size(), 10);
  DEV_EXPECT_EQ(size(v), 10);
}

DEFINE_TEST_KERNEL(CoreUtilsDev, Span, span<float> data) {
  DEV_ASSERT_EQ(data.size(), 1000);
  int x = threadIdx.x + blockDim.x*blockIdx.x;
  if (x < data.size())
    data[x] = x + 5;

  __syncthreads();

  auto blk = make_span(data.data() + blockDim.x*blockIdx.x, blockDim.x);

  x = blockDim.x*blockIdx.x;
  for (auto v : blk) {
    if (x < data.size()) {
      DEV_EXPECT_EQ(v, x + 5);
    }
    x++;
  }
}

TEST(CoreUtilsDev, Span) {
  using T = float;
  const int N = 1000;
  T *dev_data;
  CUDA_CALL(cudaMalloc(&dev_data, sizeof(T)*N));
  DEVICE_TEST_CASE_BODY(CoreUtilsDev, Span, div_ceil(N, 256), 256, make_span(dev_data, N));
  T host_data[N];
  CUDA_CALL(cudaMemcpy(host_data, dev_data, sizeof(host_data), cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++)
    EXPECT_EQ(host_data[i], i + 5);
  CUDA_CALL(cudaFree(dev_data));
}

}  // namespace dali
