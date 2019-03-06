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
#include "dali/kernels/common/copy.h"
#include "dali/kernels/alloc.h"
#include "dali/kernels/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

struct CopyTest : ::testing::Test {
  static const TensorShape<4> shape;
  static constexpr int kSize = 3 * 5 * 7 * 9;
  std::mt19937 rng;
};
const TensorShape<4> CopyTest::shape = { 3, 5, 7, 9 };

TEST_F(CopyTest, HostDevHost) {
  float data_src[kSize];
  float data_dst[kSize] = { 0 };
  auto data_gpu = memory::alloc_unique<float>(AllocType::GPU, kSize);
  auto src = make_tensor_cpu(data_src, shape);
  auto gpu = make_tensor_gpu(data_gpu.get(), shape);
  auto host = make_tensor_cpu(data_dst, shape);
  UniformRandomFill(src, rng, -1, 1);
  copy(gpu, src);
  copy(host, gpu);
  Check(host, src);
}

TEST_F(CopyTest, HostHost) {
  float data_src[kSize];
  float data_dst[kSize] = { 0 };
  auto src = make_tensor_cpu(data_src, shape);
  auto host = make_tensor_cpu(data_dst, shape);
  UniformRandomFill(src, rng, -1, 1);
  copy(host, src);
  Check(host, src);
}


TEST_F(CopyTest, HostDevDevHost) {
  float data_src[kSize];
  float data_dst[kSize] = { 0 };
  auto data_gpu1 = memory::alloc_unique<float>(AllocType::GPU, kSize);
  auto data_gpu2 = memory::alloc_unique<float>(AllocType::GPU, kSize);
  auto src = make_tensor_cpu(data_src, shape);
  auto gpu1 = make_tensor_gpu(data_gpu1.get(), shape);
  auto gpu2 = make_tensor_gpu(data_gpu2.get(), shape);
  auto host = make_tensor_cpu(data_dst, shape);
  UniformRandomFill(src, rng, -1, 1);
  copy(gpu1, src);
  copy(gpu2, gpu1);
  copy(host, gpu2);
  Check(host, src);
}


TEST_F(CopyTest, HostDevUnifiedHost) {
  float data_src[kSize];
  float data_dst[kSize] = { 0 };
  auto data_gpu = memory::alloc_unique<float>(AllocType::GPU, kSize);
  auto data_unified = memory::alloc_unique<float>(AllocType::Unified, kSize);
  auto src = make_tensor_cpu(data_src, shape);
  auto gpu = make_tensor_gpu(data_gpu.get(), shape);
  auto unified = make_tensor<StorageUnified>(data_unified.get(), shape);
  auto host = make_tensor_cpu(data_dst, shape);
  UniformRandomFill(src, rng, -1, 1);
  copy(gpu, src);
  copy(unified, gpu);
  copy(host, unified);
  Check(host, src);
}

TEST_F(CopyTest, HostUnifiedDevHost) {
  float data_src[kSize];
  float data_dst[kSize] = { 0 };
  auto data_gpu = memory::alloc_unique<float>(AllocType::GPU, kSize);
  auto data_unified = memory::alloc_unique<float>(AllocType::Unified, kSize);
  auto src = make_tensor_cpu(data_src, shape);
  auto gpu = make_tensor_gpu(data_gpu.get(), shape);
  auto unified = make_tensor<StorageUnified>(data_unified.get(), shape);
  auto host = make_tensor_cpu(data_dst, shape);
  UniformRandomFill(src, rng, -1, 1);
  copy(unified, src);
  copy(gpu, unified);
  copy(host, gpu);
  Check(host, src);
}

}  // namespace kernels
}  // namespace dali
