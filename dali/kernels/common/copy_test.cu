// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_error.h"
#include "dali/core/mm/memory.h"
#include "dali/test/tensor_test_utils.h"

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
  auto data_gpu = mm::alloc_raw_unique<float, mm::memory_kind::device>(kSize);
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
  auto data_gpu1 = mm::alloc_raw_unique<float, mm::memory_kind::device>(kSize);
  auto data_gpu2 = mm::alloc_raw_unique<float, mm::memory_kind::device>(kSize);
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

TEST_F(CopyTest, ListNonContiguous) {
  // contiguous blocks: 0-1, 2-3
  const int part1 = 26;
  const int part2 = 16;
  const int N = part1 + part2;

  TensorListShape<2> in_shape = {{
    { 2, 3 },  // 6 +
    { 4, 5 },  // 20 == 26
    { 3, 4 },  // 12 +
    { 2, 2 }   // 4  == 16
  }};

  TensorListShape<2> gpu_shape = {{
    { 4, 4 },  // 16 +
    { 2, 5 },  // 10 == 26
    { 4, 4 },  // 16
  }};

  // requires sample merging
  TensorListShape<1> out_shape = {{
    TensorShape<1>{ 10 },
    TensorShape<1>{ 22 },
    TensorShape<1>{ 10 }
  }};

  int in_data[N], ref_data[N], out_data[N];
  for (int i = 0; i < N; i++) {
    in_data[i] = i+1;
    out_data[i] = -1;
  }

  for (int i = 0; i < part1; i++)
    ref_data[i] = i + 1 + part2;
  for (int i = part1; i < N; i++)
    ref_data[i] = i + 1 - part1;

  auto gpu_data = mm::alloc_raw_unique<int, mm::memory_kind::device>(kSize);
  TensorListView<StorageCPU, int, 2> in;
  TensorListView<StorageGPU, int, 2> gpu;
  TensorListView<StorageCPU, int, 1> out, ref;

  in.resize(4);
  in.shape = in_shape;
  in.data[0] = in_data + part2;
  in.data[1] = in_data + part2 + 6;
  in.data[2] = in_data;
  in.data[3] = in_data + 12;

  gpu = make_tensor_list_gpu(gpu_data.get(), gpu_shape);

  out.resize(1);
  out = make_tensor_list_cpu(out_data, out_shape);

  ref = make_tensor_list_cpu(ref_data, out_shape);

  copy(gpu, in);
  copy(out, gpu);
  Check(out, ref);
}


TEST_F(CopyTest, HostDevUnifiedHost) {
  float data_src[kSize];
  float data_dst[kSize] = { 0 };
  mm::uptr<float> data_gpu, data_unified;
  try {
    data_gpu = mm::alloc_raw_unique<float, mm::memory_kind::device>(kSize);
    data_unified = mm::alloc_raw_unique<float, mm::memory_kind::managed>(kSize);
  } catch (const CUDAError &e) {
    if ((e.is_drv_api() && e.drv_error() == CUDA_ERROR_NOT_SUPPORTED) ||
        (e.is_rt_api() && e.rt_error() == cudaErrorNotSupported)) {
      GTEST_SKIP() << "Unified memory not supported on this platform";
    }
  }
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
  mm::uptr<float> data_gpu, data_unified;
  try {
    data_gpu = mm::alloc_raw_unique<float, mm::memory_kind::device>(kSize);
    data_unified = mm::alloc_raw_unique<float, mm::memory_kind::managed>(kSize);
  } catch (const CUDAError &e) {
    if ((e.is_drv_api() && e.drv_error() == CUDA_ERROR_NOT_SUPPORTED) ||
        (e.is_rt_api() && e.rt_error() == cudaErrorNotSupported)) {
      GTEST_SKIP() << "Unified memory not supported on this platform";
    }
  }
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


TEST_F(CopyTest, CopyReturnTensorViewTestUnified) {
  float data_src[kSize];
  auto tv = make_tensor_cpu(data_src, shape);
  UniformRandomFill(tv, rng, -1, 1);
  try {
    auto tvcpy = copy<mm::memory_kind::managed>(tv);
    CUDA_CALL(cudaDeviceSynchronize());
    Check(tv, tvcpy.first);
  } catch (const CUDAError &e) {
    if ((e.is_drv_api() && e.drv_error() == CUDA_ERROR_NOT_SUPPORTED) ||
        (e.is_rt_api() && e.rt_error() == cudaErrorNotSupported)) {
      GTEST_SKIP() << "Unified memory not supported on this platform";
    }
  }
}


TEST_F(CopyTest, CopyReturnTensorViewTestHost) {
  float data_src[kSize];
  auto tv = make_tensor_cpu(data_src, shape);
  UniformRandomFill(tv, rng, -1, 1);
  auto tvcpy = copy<mm::memory_kind::host>(tv);
  CUDA_CALL(cudaDeviceSynchronize());
  Check(tv, tvcpy.first);
}

}  // namespace kernels
}  // namespace dali
