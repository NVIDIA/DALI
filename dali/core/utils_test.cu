// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/dev_buffer.h"

namespace dali {
namespace test {

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
  DEV_EXPECT_EQ(dali::size(a0), 1u);
  int a1[] = { 2, 3, 4 };
  DEV_EXPECT_EQ(dali::size(a1), 3u);

  SmallVector<int, 5> v;
  v.resize(10);
  DEV_EXPECT_EQ(v.size(), 10u);
  DEV_EXPECT_EQ(dali::size(v), 10u);
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
  DeviceBuffer<T> dev_data;
  dev_data.resize(N);
  DEVICE_TEST_CASE_BODY(CoreUtilsDev, Span, div_ceil(N, 256), 256, make_span(dev_data.data(), N));
  T host_data[N];
  CUDA_CALL(cudaMemcpy(host_data, dev_data, sizeof(host_data), cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++)
    EXPECT_EQ(host_data[i], i + 5);
}

TEST(CoreUtils, SpanFlatten) {
  std::array<std::array<std::array<int, 4>, 3>, 2> arr;
  for (int i = 0, n = 1; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++, n++) {
        arr[i][j][k] = n;
      }
    }
  }

  span<int, 24> flat = flatten(make_span(arr));
  for (int i = 0; i < 24; i++)
    EXPECT_EQ(flat[i], i+1);

  span<const int> cflat = flatten(make_cspan(&arr[0], 2));
  for (int i = 0; i < 24; i++)
    EXPECT_EQ(cflat[i], i+1);
}

TEST(CoreUtils, CTZ) {
  int32_t i32 = 0;
  int64_t i64 = 0;
  EXPECT_EQ(ctz(i32), 32);
  EXPECT_EQ(ctz(i64), 64);
  i32 = 1;
  i64 = 1;
  EXPECT_EQ(ctz(i32), 0);
  EXPECT_EQ(ctz(i64), 0);
  i32 = 0b11010010101;
  i64 = 0b10010110111;
  EXPECT_EQ(ctz(i32), 0);
  EXPECT_EQ(ctz(i64), 0);

  i32 = 0b110100101010;
  i64 = 0b100101101110;
  EXPECT_EQ(ctz(i32), 1);
  EXPECT_EQ(ctz(i64), 1);

  i32 = 0b11010010101000;
  i64 = 0b10010110111000;
  EXPECT_EQ(ctz(i32), 3);
  EXPECT_EQ(ctz(i64), 3);

  i32 = -1;
  i64 = -1;
  uint32_t u32 = i32;
  uint64_t u64 = i64;
  for (int s = 0; s <= 32; s++) {
    EXPECT_EQ(ctz(i32), s);
    EXPECT_EQ(ctz(u32), s);
    u32 <<= 1;
    i32 = u32;
  }

  for (int s = 0; s <= 64; s++) {
    EXPECT_EQ(ctz(i64), s);
    EXPECT_EQ(ctz(u64), s);
    u64 <<= 1;
    i64 = u64;
  }
}

}  // namespace test
}  // namespace dali
