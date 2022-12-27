// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_impl_gpu_binary.cuh"
#include "dali/test/dali_operator_test.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace expr {

template <ArithmeticOp op_ = ArithmeticOp::add, typename Result_ = float, typename Left_ = float,
          typename Right_ = float, int IsLeftTensor_ = true, int IsRightTensor_ = false,
          int blocks_x_ = 128, int thread_num_ = 32, int batch_size_ = 256, int tile_size_ = 65536,
          int sample_size_ = 1024 * 1024>
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
  static constexpr int num_tiles = batch_size * tiles_per_sample;
  static_assert(sample_size >= tile_size, "This test doesn't support samples smaller than tiles.");
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
        tile_descs[tile_id].offset = TestConfig::tile_size * extent_id;
        tile_descs[tile_id].size = TestConfig::tile_size;
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

    Left l{};
    Right r{};
    auto fill_left = [&l]() { return l += 1; };
    auto fill_right = [&r]() { return r += 1; };
    Fill(left.cpu(), fill_left);
    Fill(right.cpu(), fill_right);

    // Fill pointers for tiles
    samples_data.reshape(uniform_list_shape<1>(1, {TestConfig::batch_size}));
    tiles_data.reshape(uniform_list_shape<1>(1, {TestConfig::num_tiles}));
    auto samples_cpu = samples_data.cpu()[0];
    auto tiles_cpu = tiles_data.cpu()[0];
    // TestTensorList just allocates memory, this can leave SmallVector in weird state
    memset(samples_cpu.data, 0, TestConfig::batch_size * sizeof(SampleDescGPU<2>));
    memset(tiles_cpu.data, 0, TestConfig::num_tiles * sizeof(TileDesc));

    for (int sample_idx = 0; sample_idx < TestConfig::batch_size; sample_idx++) {
      auto &sample = samples_cpu.data[sample_idx];
      sample.ndim = 1;

      auto out_tv = left.gpu()[sample_idx];
      TensorShape<> out_strides;
      kernels::CalcStrides(out_strides, out_tv.shape);
      sample.output.data = out_tv.data;
      sample.output.dtype = type2id<Result>::value;
      sample.output.shape[0] = out_tv.shape[0];

      auto left_tv = left.gpu()[sample_idx];
      TensorShape<> left_strides;
      kernels::CalcStrides(left_strides, left_tv.shape);
      sample.args[0].data = left_tv.data;
      sample.args[0].dtype = type2id<Result>::value;
      sample.args[0].shape[0] = volume(left_tv.shape);

      auto right_tv = right.gpu()[sample_idx];
      TensorShape<> right_strides;
      kernels::CalcStrides(right_strides, right_tv.shape);
      sample.args[1].data = right_tv.data;
      sample.args[1].dtype = type2id<Result>::value;
      sample.args[1].shape[0] =  volume(right_tv.shape);

      for (int extent_idx = 0; extent_idx < TestConfig::tiles_per_sample; extent_idx++) {
        int tile_idx = sample_idx * TestConfig::tiles_per_sample + extent_idx;
        tiles_cpu.data[tile_idx] = tile_descs[tile_idx];
      }
    }

    tiles_gpu = tiles_data.gpu(stream)[0].data;
    samples_gpu = samples_data.gpu(stream)[0].data;
  }

  void MeasurePerf() {
    ExecuteTiledBinOpND<TestConfig::op, Result, Left, Right>
        <<<grid, block, 0, stream>>>(samples_gpu, tiles_gpu);

    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);

    CUDA_CALL(cudaEventRecord(start, stream));
    constexpr int kIters = 100;
    for (int i = 0; i < kIters; i++) {
      ExecuteTiledBinOpND<TestConfig::op, Result, Left, Right>
          <<<grid, block, 0, stream>>>(samples_gpu, tiles_gpu);
    }
    CUDA_CALL(cudaEventRecord(end, stream));
    CUDA_CALL(cudaDeviceSynchronize());
    float time;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));

    time *= (1e+6f / kIters);  // convert to nanoseconds / 100 samples
    int64_t data_size = 0;
    data_size +=
        static_cast<int64_t>(TestConfig::num_tiles) * TestConfig::tile_size * sizeof(Result);
    if (TestConfig::IsLeftTensor)
      data_size +=
          static_cast<int64_t>(TestConfig::num_tiles) * TestConfig::tile_size * sizeof(Left);
    if (TestConfig::IsRightTensor)
      data_size +=
          static_cast<int64_t>(TestConfig::num_tiles) * TestConfig::tile_size * sizeof(Right);
    std::cerr << "Throughput: " << data_size / time << " GB/s\n";
  }

  using Result = typename TestConfig::Result;
  using Left = typename TestConfig::Left;
  using Right = typename TestConfig::Right;

  // For kernel launch
  dim3 grid = dim3(TestConfig::blocks_x, TestConfig::num_tiles, 1);
  dim3 block = dim3(TestConfig::thread_num, 1, 1);

  // Samples, tiles and data
  std::vector<TileDesc> tile_descs;
  kernels::TestTensorList<TileDesc, 1> tiles_data;
  kernels::TestTensorList<SampleDescGPU<2>, 1> samples_data;

  kernels::TestTensorList<Result, 1> result;
  kernels::TestTensorList<Left, 1> left;
  kernels::TestTensorList<Right, 1> right;

  CUDAStream stream;
  const SampleDescGPU<2> *samples_gpu;
  const TileDesc *tiles_gpu;
};

TYPED_TEST_SUITE_P(BinaryArithmeticOpGpuPerfTest);

TYPED_TEST_P(BinaryArithmeticOpGpuPerfTest, Perf) {
  std::cerr << "Blocks_x: " << TypeParam::blocks_x << ", thread_num: " << TypeParam::thread_num
            << ", tile_size: " << TypeParam::tile_size / 1024.f
            << "KB, sample_size: " << TypeParam::sample_size / 1048576.f << "MB" << std::endl;
  // TypeParam n = 0;
  this->MeasurePerf();
}

REGISTER_TYPED_TEST_SUITE_P(BinaryArithmeticOpGpuPerfTest, Perf);

using TestConfigs = ::testing::Types<
    // op, Result, Left, Right, IsLeftTensor, IsRightTensor, blocks_x, thread_num, batch, tile,
    // sample Test Tensor op Constant
    ArithmOpParams<  // old config
        ArithmeticOp::add, float, float, float, true, false, 128, 256, 256, 16384, 1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 256, 256, 32768,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 256, 256, 65536,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 256, 256, 131072,
                   1024 * 1024>,
    // test small input data, forcing 1 tile per sample, a bit bigger batch,
    // to measure how performs with smaller inputs
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 256, 512, 16384,
                   16384>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 256, 512, 32768,
                   32768>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 256, 512, 65536,
                   65536>,

    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 64, 256, 256, 16384,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 64, 256, 256, 32768,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 64, 256, 256, 65536,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 64, 256, 256, 131072,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 64, 256, 512, 16384, 16384>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 64, 256, 512, 32768, 32768>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 64, 256, 512, 65536, 65536>,

    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 128, 256, 16384,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 128, 256, 32768,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 128, 256, 65536,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 128, 256, 131072,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 128, 512, 16384,
                   16384>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 128, 512, 32768,
                   32768>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, false, 128, 128, 512, 65536,
                   65536>,

    // Test Tensor op Tensor
    ArithmOpParams<  // old config
        ArithmeticOp::add, float, float, float, true, true, 128, 256, 256, 16384, 1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, true, 128, 256, 256, 65536,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, true, 128, 256, 512, 16384, 16384>,

    ArithmOpParams<ArithmeticOp::add, float, float, float, true, true, 64, 256, 256, 65536,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, true, 64, 256, 512, 16384, 16384>,

    ArithmOpParams<ArithmeticOp::add, float, float, float, true, true, 128, 128, 256, 65536,
                   1024 * 1024>,
    ArithmOpParams<ArithmeticOp::add, float, float, float, true, true, 128, 128, 512, 16384,
                   16384>>;

INSTANTIATE_TYPED_TEST_SUITE_P(BinaryArithmeticOpGpu, BinaryArithmeticOpGpuPerfTest, TestConfigs);

}  // namespace expr
}  // namespace dali
