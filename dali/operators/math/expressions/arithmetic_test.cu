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


struct StaticParams {};


TEST(ArithmeticOpTest, Perf) {
  int blocks_x = 128;
  int thread_num = 32;
  int batch_size = 256;
  int tile_size = 65536;
  // int blocks_x = 128;
  // int thread_num = 256;
  // int batch_size = 256;
  // int tile_size = 16358;
  int tiles_per_sample = 1024 * 1024 / tile_size; // 64K * 16 = 1MB
  int num_tiles = batch_size *  tiles_per_sample;
  dim3 grid = dim3(blocks_x, num_tiles, 1);
  dim3 block = dim3(thread_num, 1, 1);

  auto stream = CUDAStream::Create(true);

  constexpr ArithmeticOp op = ArithmeticOp::mul;
  using Result = float;
  using Left = float;
  using Right = float;
  constexpr int IsLeftTensor = true;
  constexpr int IsRightTensor = false;


  /// Fill tile descriptors (shapes)
  std::vector<TileDesc> tile_descs(num_tiles);
  for (int sample_id = 0; sample_id < batch_size; sample_id++) {
    for (int extent_id = 0; extent_id < tiles_per_sample; extent_id++) {
      int tile_id = sample_id * tiles_per_sample + extent_id;
      tile_descs[tile_id].sample_idx = sample_id;
      tile_descs[tile_id].extent_idx = extent_id;
      tile_descs[tile_id].tile_size = tile_size;
      tile_descs[tile_id].extent_size = tile_size;
    }
  }

  // Reshape memory for those tiles
  kernels::TestTensorList<Result, 1> result;
  kernels::TestTensorList<Left, 1> left;
  kernels::TestTensorList<Right, 1> right;
  result.reshape(uniform_list_shape<1>(batch_size, {tile_size * tiles_per_sample}));
  if (IsLeftTensor) {
    left.reshape(uniform_list_shape<1>(batch_size, {tile_size * tiles_per_sample}));
  } else {
    left.reshape(uniform_list_shape<1>(batch_size, {1}));
  }
  if (IsRightTensor) {
    right.reshape(uniform_list_shape<1>(batch_size, {tile_size * tiles_per_sample}));
  } else {
    right.reshape(uniform_list_shape<1>(batch_size, {1}));
  }

  // Fill pointers for tiles
  kernels::TestTensorList<ExtendedTileDesc, 1> tiles_data;
  tiles_data.reshape(uniform_list_shape<1>(1, {num_tiles}));
  auto tiles_cpu = tiles_data.cpu()[0];
  for (int sample_id = 0; sample_id < batch_size; sample_id++) {
    for (int extent_id = 0; extent_id < tiles_per_sample; extent_id++) {
      int tile_id = sample_id * tiles_per_sample + extent_id;
      tiles_cpu(tile_id)->desc = tile_descs[tile_id];
      tiles_cpu(tile_id)->output = result.gpu(stream)[sample_id].data + extent_id * tile_size;
      tiles_cpu(tile_id)->args[0] = left.gpu(stream)[sample_id].data + (IsLeftTensor ? extent_id * tile_size : 0);
      tiles_cpu(tile_id)->args[1] = right.gpu(stream)[sample_id].data + (IsRightTensor ? extent_id * tile_size : 0);
    }
  }
  auto *tiles_gpu = tiles_data.gpu(stream)[0].data;


  ExecuteTiledBinOp<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
        <<<grid, block, 0, stream>>>(tiles_gpu);

  CUDAEvent start = CUDAEvent::CreateWithFlags(0);
  CUDAEvent end = CUDAEvent::CreateWithFlags(0);

  cudaEventRecord(start, stream);
  constexpr int kIters = 100;
  for (int i = 0; i < kIters; i++) {
    ExecuteTiledBinOp<op, Result, Left, Right, IsLeftTensor, IsRightTensor>
        <<<grid, block, 0, stream>>>(tiles_gpu);
  }
  cudaEventRecord(end, stream);
  cudaDeviceSynchronize();
  float time;
  CUDA_CALL(cudaEventElapsedTime(&time, start, end));

  time *= (1e+6f / kIters);  // convert to nanoseconds / 100 samples
  int64_t data_size = 0;
  data_size += num_tiles * tile_size * sizeof(Result);
  if (IsLeftTensor)
    data_size += num_tiles * tile_size * sizeof(Left);
  if (IsRightTensor)
    data_size +=  num_tiles * tile_size * sizeof(Right);
  std::cerr << "Throughput: " << data_size / time << " GB/s\n";
}

}  // namespace dali
