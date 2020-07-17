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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// #include "dali/core/static_switch.h"
#include "dali/operators/math/expressions/expression_impl_gpu.cuh"
// #include "dali/operators/math/expressions/arithmetic.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/test/test_tensors.h"
#include "dali/core/cuda_stream.h"
#include "dali/core/cuda_event.h"


// #include "dali/pipeline/data/types.h"
// #include "dali/pipeline/pipeline.h"
#include "dali/test/dali_operator_test.h"
#include "dali/test/tensor_test_utils.h"



namespace dali {

template <ArithmeticOp op_ = ArithmeticOp::mul,
  typename Result_ = float,
  typename Left_ = float,
  typename Right_ = float,
  int IsLeftTensor_ = true,
  int IsRightTensor_ = false,
  int blocks_x_ = 128,
  int thread_num_ = 32,
  int batch_size_ = 256,
  int tile_size_ = 65536,
  int sample_size_ = 1024 *1024>
struct ArithmOpParams {
  static constexpr ArithmeticOp op = op_;
  using Result = Result_;
  using Left = Left_;
  using Right = Right_;
  static constexpr int IsLeftTensor = IsLeftTensor_;
  static constexpr int IsRightTensor = IsRightTensor_;
  static constexpr int blocks_x = blocks_x_;
  static constexpr int thread_num = thread_num_;
  static constexpr int batch_size = batch_size_;
  static constexpr int tile_size = tile_size_;
  static constexpr int sample_size = sample_size_;
  static constexpr int tiles_per_sample = sample_size / tile_size;
  static constexpr int num_tiles = batch_size *  tiles_per_sample;
};


template <typename TestConfig>
struct BinaryArithmeticOpGpuPerfTest : public ::testing::Test {
  void SetUp() override {
    stream = CUDAStream::Create(true);

    /// Fill tile descriptors (shapes)
    tile_descs.resize(TestConfig::num_tiles);
    for (int sample_id = 0; sample_id < TestConfig::batch_size; sample_id++) {
      for (int extent_id = 0; extent_id < TestConfig::tiles_per_sample; extent_id++) {
        int tile_id = sample_id * TestConfig::tiles_per_sample + extent_id;
        tile_descs[tile_id].sample_idx = sample_id;
        tile_descs[tile_id].extent_idx = extent_id;
        tile_descs[tile_id].tile_size = TestConfig::tile_size;
        tile_descs[tile_id].extent_size = TestConfig::tile_size;
      }
    }


  // Reshape memory for those tiles
    result.reshape(uniform_list_shape<1>(TestConfig::batch_size,
                                         {TestConfig::tile_size * TestConfig::tiles_per_sample}));
    if (TestConfig::IsLeftTensor) {
      left.reshape(uniform_list_shape<1>(TestConfig::batch_size,
                                         {TestConfig::tile_size * TestConfig::tiles_per_sample}));
    } else {
      left.reshape(uniform_list_shape<1>(TestConfig::batch_size, {1}));
    }
    if (TestConfig::IsRightTensor) {
      right.reshape(uniform_list_shape<1>(TestConfig::batch_size,
                                          {TestConfig::tile_size * TestConfig::tiles_per_sample}));
    } else {
      right.reshape(uniform_list_shape<1>(TestConfig::batch_size, {1}));
    }

    // Fill pointers for tiles
    tiles_data.reshape(uniform_list_shape<1>(1, {TestConfig::num_tiles}));
    auto tiles_cpu = tiles_data.cpu()[0];
    for (int sample_id = 0; sample_id < TestConfig::batch_size; sample_id++) {
      for (int extent_id = 0; extent_id < TestConfig::tiles_per_sample; extent_id++) {
        int tile_id = sample_id * TestConfig::tiles_per_sample + extent_id;
        tiles_cpu(tile_id)->desc = tile_descs[tile_id];
        tiles_cpu(tile_id)->output =
            result.gpu(stream)[sample_id].data + extent_id * TestConfig::tile_size;
        tiles_cpu(tile_id)->args[0] = left.gpu(stream)[sample_id].data +
            (TestConfig::IsLeftTensor ? extent_id * TestConfig::tile_size : 0);
        tiles_cpu(tile_id)->args[1] = right.gpu(stream)[sample_id].data +
            (TestConfig::IsRightTensor ? extent_id * TestConfig::tile_size : 0);
      }
    }

    tiles_gpu = tiles_data.gpu(stream)[0].data;
  }

  void MeasurePerf() {
    ExecuteTiledBinOp<TestConfig::op, Result, Left, Right, TestConfig::IsLeftTensor,
                      TestConfig::IsRightTensor><<<grid, block, 0, stream>>>(tiles_gpu);

    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);

    cudaEventRecord(start, stream);
    constexpr int kIters = 100;
    for (int i = 0; i < kIters; i++) {
      ExecuteTiledBinOp<TestConfig::op, Result, Left, Right, TestConfig::IsLeftTensor,
                        TestConfig::IsRightTensor><<<grid, block, 0, stream>>>(tiles_gpu);
    }
    cudaEventRecord(end, stream);
    cudaDeviceSynchronize();
    float time;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));

    time *= (1e+6f / kIters);  // convert to nanoseconds / 100 samples
    int64_t data_size = 0;
    data_size += TestConfig::num_tiles * TestConfig::tile_size * sizeof(Result);
    if (TestConfig::IsLeftTensor)
      data_size += TestConfig::num_tiles * TestConfig::tile_size * sizeof(Left);
    if (TestConfig::IsRightTensor)
      data_size += TestConfig::num_tiles * TestConfig::tile_size * sizeof(Right);
    std::cerr << "Throughput: " << data_size / time << " GB/s\n";
  }

  using Result = typename TestConfig::Result;
  using Left = typename TestConfig::Left;
  using Right = typename TestConfig::Right;

  // For kernel launch
  dim3 grid = dim3(TestConfig::blocks_x, TestConfig::num_tiles, 1);
  dim3 block = dim3(TestConfig::thread_num, 1, 1);

  // Tiles and data
  std::vector<TileDesc> tile_descs;
  kernels::TestTensorList<ExtendedTileDesc, 1> tiles_data;

  kernels::TestTensorList<Result, 1> result;
  kernels::TestTensorList<Left, 1> left;
  kernels::TestTensorList<Right, 1> right;

  CUDAStream stream;
  const ExtendedTileDesc *tiles_gpu;
};

TYPED_TEST_SUITE_P(BinaryArithmeticOpGpuPerfTest);


TYPED_TEST_P(BinaryArithmeticOpGpuPerfTest, Perf) {

  // TypeParam n = 0;
  this->MeasurePerf();

}

REGISTER_TYPED_TEST_SUITE_P(BinaryArithmeticOpGpuPerfTest, Perf);

using TestConfigs = ::testing::Types<ArithmOpParams<>>;

INSTANTIATE_TYPED_TEST_SUITE_P(BinaryArithmeticOpGpu, BinaryArithmeticOpGpuPerfTest, TestConfigs);

// TEST(BinaryArithmeticOpGpuPerfTest, Perf) {
//   int blocks_x = 128;
//   int thread_num = 32;
//   int batch_size = 256;
//   int tile_size = 65536;
//   int tiles_per_sample = 1024 * 1024 / tile_size; // 64K * 16 = 1MB
//   // int blocks_x = 128;
//   // int thread_num = 256;
//   // int batch_size = 256;
//   // int tile_size = 16358;
//   int num_tiles = batch_size *  tiles_per_sample;
//   dim3 grid = dim3(blocks_x, num_tiles, 1);
//   dim3 block = dim3(thread_num, 1, 1);


//   constexpr ArithmeticOp op = ArithmeticOp::mul;
//   using Result = float;
//   using Left = float;
//   using Right = float;
//   constexpr int IsLeftTensor = true;
//   constexpr int IsRightTensor = false;


//   /// Fill tile descriptors (shapes)
//   std::vector<TileDesc> tile_descs(num_tiles);
//   for (int sample_id = 0; sample_id < batch_size; sample_id++) {
//     for (int extent_id = 0; extent_id < tiles_per_sample; extent_id++) {
//       int tile_id = sample_id * tiles_per_sample + extent_id;
//       tile_descs[tile_id].sample_idx = sample_id;
//       tile_descs[tile_id].extent_idx = extent_id;
//       tile_descs[tile_id].tile_size = tile_size;
//       tile_descs[tile_id].extent_size = tile_size;
//     }
//   }

//   // Reshape memory for those tiles
//   kernels::TestTensorList<Result, 1> result;
//   kernels::TestTensorList<Left, 1> left;
//   kernels::TestTensorList<Right, 1> right;
//   result.reshape(uniform_list_shape<1>(batch_size, {tile_size * tiles_per_sample}));
//   if (IsLeftTensor) {
//     left.reshape(uniform_list_shape<1>(batch_size, {tile_size * tiles_per_sample}));
//   } else {
//     left.reshape(uniform_list_shape<1>(batch_size, {1}));
//   }
//   if (IsRightTensor) {
//     right.reshape(uniform_list_shape<1>(batch_size, {tile_size * tiles_per_sample}));
//   } else {
//     right.reshape(uniform_list_shape<1>(batch_size, {1}));
//   }

//   // Fill pointers for tiles
//   kernels::TestTensorList<ExtendedTileDesc, 1> tiles_data;
//   tiles_data.reshape(uniform_list_shape<1>(1, {num_tiles}));
//   auto tiles_cpu = tiles_data.cpu()[0];
//   for (int sample_id = 0; sample_id < batch_size; sample_id++) {
//     for (int extent_id = 0; extent_id < tiles_per_sample; extent_id++) {
//       int tile_id = sample_id * tiles_per_sample + extent_id;
//       tiles_cpu(tile_id)->desc = tile_descs[tile_id];
//       tiles_cpu(tile_id)->output = result.gpu(stream)[sample_id].data + extent_id * tile_size;
//       tiles_cpu(tile_id)->args[0] = left.gpu(stream)[sample_id].data + (IsLeftTensor ? extent_id * tile_size : 0);
//       tiles_cpu(tile_id)->args[1] = right.gpu(stream)[sample_id].data + (IsRightTensor ? extent_id * tile_size : 0);
//     }
//   }
//   auto *tiles_gpu = tiles_data.gpu(stream)[0].data;


//   ExecuteTiledBinOp<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
//         <<<grid, block, 0, stream>>>(tiles_gpu);

//   CUDAEvent start = CUDAEvent::CreateWithFlags(0);
//   CUDAEvent end = CUDAEvent::CreateWithFlags(0);

//   cudaEventRecord(start, stream);
//   constexpr int kIters = 100;
//   for (int i = 0; i < kIters; i++) {
//     ExecuteTiledBinOp<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
//         <<<grid, block, 0, stream>>>(tiles_gpu);
//   }
//   cudaEventRecord(end, stream);
//   cudaDeviceSynchronize();
//   float time;
//   CUDA_CALL(cudaEventElapsedTime(&time, start, end));

//   time *= (1e+6f / kIters);  // convert to nanoseconds / 100 samples
//   int64_t data_size = 0;
//   data_size += num_tiles * tile_size * sizeof(Result);
//   if (IsLeftTensor)
//     data_size += num_tiles * tile_size * sizeof(Left);
//   if (IsRightTensor)
//     data_size +=  num_tiles * tile_size * sizeof(Right);
//   std::cerr << "Throughput: " << data_size / time << " GB/s\n";
// }

}  // namespace dali
