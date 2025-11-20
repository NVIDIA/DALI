// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/util/random_dist.h"
#include "dali/test/device_test.h"

namespace dali {
namespace test {

template <typename T>
__device__ vec<2, T> CurendUniformDistTest() {
  curand_uniform_dist<T> dist(1, 2);
  const int N = 4096;
  // Generate a bunch of random values in shared memory...
  __shared__ T values[N];
  curandStatePhilox4_32_10_t state;
  curand_init(12345, blockIdx.x, threadIdx.x * 4, &state);
    for (int i = threadIdx.x * 4; i < N; i += blockDim.x * 4) {
    vec<4, T> val = dist.get4(state);
    values[i + 0] = val[0];
    values[i + 1] = val[1];
    values[i + 2] = val[2];
    values[i + 3] = val[3];
  }
  __syncthreads();
  // ..and check the distribution.

  // Not efficient, but it's only a test.
  T mean = 0;
  T var = 0;
  if (threadIdx.x == 0) {
    for (int i = 0; i < N; i++)
      mean += values[i];
    mean /= N;
    for (int i = 0; i < N; i++)
      var += (values[i] - mean) * (values[i] - mean);
    var /= N;
  }
  return { mean, var };
}

DEVICE_TEST(CurandUniformDistTest, Float, 1024, 256) {
  vec<2, float> result = CurendUniformDistTest<float>();
  if (threadIdx.x == 0) {
    const float expected_mean = 1.5f;
    const float expected_var = 1.0f / 12;
    float mean = result[0];
    float var = result[1];
    DEV_EXPECT_LT(abs(mean - expected_mean), 3e-2f);
    DEV_EXPECT_LT(abs(var - expected_var), 1e-2f);
  }
}

}  // namespace test
}  // namespace dali
