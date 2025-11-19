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
#include <random>
#include <vector>
#include "dali/operators/random/philox.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/cuda_stream.h"

namespace dali {
namespace test {

namespace {

__global__ void GetCurandPhiloxOutput(
      uint32_t *output,
      int n,
      uint64_t key,
      uint64_t sequence,
      uint64_t offset,
      bool use_skipahead) {
  curandStatePhilox4_32_10_t curand_state{};
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;
  if (use_skipahead) {
    curand_init(key, 0, tid, &curand_state);
    skipahead(offset, &curand_state);
    skipahead_sequence(sequence, &curand_state);
    output[tid] = curand(&curand_state);
  } else {
    curand_init(key, sequence, offset, &curand_state);
    skipahead(tid, &curand_state);  // we need to skipahead to avoid overflow for large offsets
    output[tid] = curand(&curand_state);
  }
}

std::vector<uint32_t> GetReferencePhiloxOutput(
  int n, uint64_t key, uint64_t sequence, uint64_t offset, bool skipahead = false) {
  std::vector<uint32_t> output(n);
  DeviceBuffer<uint32_t> output_buf;
  output_buf.resize(n);

  CUDAStream s = CUDAStream::Create(true);
  CUDA_CALL(cudaMemsetAsync(output_buf.data(), 0xFE, n * sizeof(uint32_t), s));

  GetCurandPhiloxOutput<<<div_ceil(n, 256), 256, 0, s>>>(
      output_buf.data(), n, key, sequence, offset, skipahead);

  copyD2H(output.data(), output_buf.data(), n, s.get());

  CUDA_CALL(cudaStreamSynchronize(s));
  return output;
}

  // some arbitrary values
const uint64_t key = 0xCAFEBABEFEEDCAFE_u64;
const uint64_t seq = 0xDECAFBADDEADBEEF_u64;
const uint64_t ofs = 0xFFFFFFFFFFFFF000_u64;  // make sure we overflow

}  // namespace


TEST(TestPhilox, CurandSkipaheadSanityCheck) {
  const int n = 1 << 20;
  auto ref = GetReferencePhiloxOutput(n, key, seq, ofs, false);
  auto skipahead = GetReferencePhiloxOutput(n, key, seq, ofs, true);

  for (int i = 0; i < n; i++) {
    ASSERT_EQ(ref[i], skipahead[i]) << " at " << i << "cuRAND vs cuRAND skipahead mismatch";
  }
}


TEST(TestPhilox, VersusCurandSeq) {
  const int n = 1 << 20;
  auto ref = GetReferencePhiloxOutput(n, key, seq, ofs);

  Philox4x32_10 philox{};
  philox.init(key, seq, ofs);
  for (int i = 0; i < n; i++) {
    uint32_t ret = philox.next();
    uint32_t curand_ret = ref[i];
    ASSERT_EQ(ret, curand_ret) << " at " << i;
  }
}

TEST(TestPhilox, VersusCurandInitSeqOffset) {
  const int n = 1 << 20;
  auto ref = GetReferencePhiloxOutput(n, key, seq, ofs);

  Philox4x32_10 philox{};
  for (int i = n - 1; i >= 0; i--) {
    philox.init(key, seq, ofs);
    philox.skipahead(i);  // we can't just init becasue i + ofs overflows
    uint32_t ret = philox.next();
    uint32_t curand_ret = ref[i];
    ASSERT_EQ(ret, curand_ret) << " at " << i;
  }
}

TEST(TestPhilox, VersusCurandInitCtrPhase) {
  const int n = 1 << 20;
  auto ref = GetReferencePhiloxOutput(n, key, seq, ofs);

  Philox4x32_10 philox{};
  for (int i = n - 1; i >= 0; i--) {
    uint64_t ofs_lo = ofs + i;
    uint64_t ofs_hi = ofs_lo < ofs;

    philox.init(key, seq, (ofs_lo >> 2) | (ofs_hi << 62), ofs_lo & 3);
    uint32_t ret = philox.next();
    uint32_t curand_ret = ref[i];
    ASSERT_EQ(ret, curand_ret) << " at " << i;
  }
}

TEST(TestPhilox, VersusCurandRandomSkipahead) {
  const int n = 1 << 20;
  auto ref = GetReferencePhiloxOutput(n, key, seq, ofs);

  std::mt19937 mt(12345);
  Philox4x32_10 philox{};
  // Go backwards and reinitialize the state each time to check rewinding
  for (int i = n - 1; i >= 0; i--) {
    uint64_t seq_part = std::uniform_int_distribution<uint64_t>(0, seq)(mt);
    uint64_t ofs_part = std::uniform_int_distribution<uint64_t>(i, ofs)(mt);
    philox.init(key, seq_part, ofs_part);

    philox.skipahead_sequence(seq - seq_part);
    philox.skipahead(ofs - ofs_part + i);
    uint32_t ret = philox.next();
    uint32_t curand_ret = ref[i];
    ASSERT_EQ(ret, curand_ret) << " at " << i;
  }
}

}  // namespace test
}  // namespace dali
