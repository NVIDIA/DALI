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

#include <gtest/gtest.h>
#include <curand_kernel.h>
#include <vector>
#include "dali/operators/random/philox.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/cuda_stream.h"

namespace dali {
namespace test {

__global__ void GetCurandPhiloxOutput(
      uint32_t *output, int n, uint64_t key, uint64_t sequence, uint64_t offset) {
  curandStatePhilox4_32_10_t curand_state;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  curand_init(key, sequence, offset + tid, &curand_state);
  output[tid] = curand(&curand_state);
}

std::vector<uint32_t> GetReferencePhiloxOutput(
  int n, uint64_t key, uint64_t sequence, uint64_t offset) {
  std::vector<uint32_t> output(n);
  DeviceBuffer<uint32_t> output_buf;
  output_buf.resize(n);
  CUDA_CALL(cudaMemset(output_buf.data(), 0xFE, n * sizeof(uint32_t)));

  CUDAStream s = CUDAStream::Create(true);

  GetCurandPhiloxOutput<<<div_ceil(n, 256), 256, 0, s>>>(
      output_buf.data(), n, key, sequence, offset);

  copyD2H(output.data(), output_buf.data(), n, s.get());

  CUDA_CALL(cudaStreamSynchronize(s));
  return output;
}

TEST(TestPhilox, VersusCurand) {
  const int n = 1 << 20;

  // some arbitrary values
  uint64_t key = 0xCAFEBABEFEEDCAFE_u64;
  uint64_t seq = 0xDECAFBADDEADBEEF_u64;
  uint64_t ofs = 0x600DF00DF0CACC1A_u64;

  auto ref = GetReferencePhiloxOutput(n, key, seq, ofs);

  Philox4x32_10 philox;
  philox.init(key, seq, ofs);
  for (int i = 0; i < n; i++) {
    uint32_t ret = philox.next();
    uint32_t curand_ret = ref[i];
    ASSERT_EQ(ret, curand_ret) << " at " << i;
  }

  // Go backwards and reinitialize the state each time to check rewinding
  for (int i = n - 1; i >= 0; i--) {
    philox.init(key, seq, ofs + i);
    uint32_t ret = philox.next();
    uint32_t curand_ret = ref[i];
    ASSERT_EQ(ret, curand_ret) << " at " << i;
  }
}

}  // namespace test
}  // namespace dali
