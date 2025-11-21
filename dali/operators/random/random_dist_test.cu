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

#include "dali/operators/random/random_dist.h"
#include <gtest/gtest.h>
#include <curand_kernel.h>
#include <vector>
#include "dali/core/cuda_stream.h"
#include "dali/core/dev_buffer.h"
#include "dali/operators/random/philox.h"

namespace dali {
namespace random {
namespace test {

template <typename CurandState>
struct CurandGenerator {
    CurandState &state;
    __device__ explicit CurandGenerator(CurandState &s) : state(s) {}
    __device__ inline uint32_t operator()() const {
        return curand(&state);
    }
};

/** Fill the array with the data distributed according to `dist` using Philox RNG.
 *
 * The output array is filled in blocks of 4 to amortize the cost of Philox evaluation.
 * The generator is configured as if it started at offset defined by the position in the array.
 * The offset is further multiplied by 16 to avoid significant overlap between adjacent groups
 * of 4 elements. This is useful when the distribution needs to get multiple words from the RNG.
 */
template <typename T, typename Dist>
__global__ void GetGPUDistOutput(T *output, int n, Dist d, uint64_t seed, uint64_t seq) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int base_idx = tid * 4;
  if (base_idx >= n)
    return;
  curandStatePhilox4_32_10_t curand_state = {};
  curand_init(seed, seq, base_idx * 16, &curand_state);
  Dist dist = d;
  CurandGenerator gen(curand_state);
  // Go in blocks of 4 to amortize the cost of Philox evaluation
  for (int i = base_idx; i < base_idx + 4 && i < n; i++) {
    T value = dist(gen);
    output[i] = value;
  }
}

template <typename T, typename Dist>
void GetCPUDistOutput(T *output, int n, Dist dist, uint64_t seed, uint64_t seq) {
  for (int base = 0; base < n; base += 4) {
    Philox4x32_10 philox{};
    philox.init(seed, seq, base * 16);
    // Go in blocks of 4 to amortize the cost of Philox evaluation
    for (int i = base; i < base + 4 && i < n; i++) {
        output[i] = dist(philox);
    }
  }
}


template <typename T, typename Dist>
void CompareDist(Dist dist, int length, double tolerance, uint64_t seed = 12345, uint64_t seq = 0) {
  std::vector<T> gpu_output(length);
  std::vector<T> cpu_output(length);
  DeviceBuffer<T> dev_buf;;
  dev_buf.resize(length);
  CUDAStream s = CUDAStream::Create(true);

  CUDA_CALL(cudaMemsetAsync(dev_buf.data(), 0xFE, length * sizeof(T), s));

  GetGPUDistOutput<<<div_ceil(length, 4 * 256), 256, 0, s.get()>>>(
      dev_buf.data(), length, dist, seed, seq);

  copyD2H(gpu_output.data(), dev_buf.data(), length, s.get());

  CUDA_CALL(cudaStreamSynchronize(s));

  GetCPUDistOutput(cpu_output.data(), length, dist, seed, seq);
  for (int i = 0; i < length; i++) {
    ASSERT_NEAR(gpu_output[i], cpu_output[i], tolerance) << " at " << i;
  }
}

template <typename T>
class GPURandomDistFPTest : public ::testing::Test {};

template <typename T>
class GPURandomDistIntTest : public ::testing::Test {};

using FPTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GPURandomDistFPTest, FPTypes);

TYPED_TEST(GPURandomDistFPTest, UniformReal) {
  using T = TypeParam;
  dali::random::uniform_real_dist<T> dist(1, 2);
  double eps = std::is_same_v<T, float> ? 1e-6f : 1e-15;
  CompareDist<T>(dist, 10001, eps);
}

TYPED_TEST(GPURandomDistFPTest, Normal) {
  using T = TypeParam;
  dali::random::normal_dist<T> dist(2, 3);
  double eps = std::is_same_v<T, float> ? 1e-5f : 1e-14;
  CompareDist<T>(dist, 10001, eps);
}

TEST(GPURandomDistTest, UniformIntUnsigned) {
  using T = uint32_t;
  dali::random::uniform_int_dist<T> dist(0, 0xffffffffu);
  CompareDist<T>(dist, 10002, 0);
  dist = dali::random::uniform_int_dist<T>(1234, 31337, true);
  CompareDist<T>(dist, 10003, 0);
}

TEST(GPURandomDistTest, UniformIntSigned) {
  using T = int32_t;
  dali::random::uniform_int_dist<T> dist(-0x80000000, 0x7fffffff);
  CompareDist<T>(dist, 10001, 0);

  dist = dali::random::uniform_int_dist<T>(-100, 12345, true);
  CompareDist<T>(dist, 10002, 0);
}

TEST(GPURandomDistTest, Bernoulli) {
  using T = int;
  dali::random::bernoulli_dist dist(0.25f);
  CompareDist<T>(dist, 10003, 0);
}

}  // namespace test
}  // namespace random
}  // namespace dali
