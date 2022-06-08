// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/signal/moving_mean_square_gpu.h"
#include <gtest/gtest.h>
#include <vector>
#include "dali/core/cuda_event.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {
namespace signal {
namespace test {


template<class InputType>
class MovingMeanSquareGPU : public ::testing::Test {
 public:
  using In = InputType;
  using Out = float;

  TestTensorList<In> in_;
  TestTensorList<Out> out_;

  void SetUp() final {
    int nsamples = 4;

    TensorListShape<> sh = {{10000, }, {1000, }, {2049, }, {2027, }};
    in_.reshape(sh);
    out_.reshape(sh);

    std::mt19937 rng;
    UniformRandomFill(in_.cpu(), rng, 0.0f, 1.0f);
  }

  /**
   * @brief Naive calculation. Square and sum over a region
   */
  Out naive_moving_mean_square(In *start, In *pos, int window_size) {
    float factor = 1.0f / window_size;
    In *ptr = pos - window_size;
    if (ptr < start)
      ptr = start;
    acc_t<In> sum = 0;
    for (; ptr <= pos; ++ptr) {
      acc_t<In> x = *ptr;
      sum += (x * x);
    }
    return ConvertSat<Out>(factor * sum);
  }

  void RunTest() {
    KernelContext ctx;
    ctx.gpu.stream = 0;
    DynamicScratchpad dyn_scratchpad({}, AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;

    auto out_batch = out_.gpu().to_static<1>();
    auto in_batch = in_.gpu().template to_static<1>();
    int nsamples = in_batch.size();

    int window_size = 2048;
    MovingMeanSquareArgs args{window_size};
    MovingMeanSquareGpu<In> kernel;

    kernel.Run(ctx, out_batch, in_batch, args);
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));

    for (int s = 1; s < nsamples; s++) {
      auto in = in_.cpu()[s].data;
      auto out = out_.cpu()[s].data;
      int64_t len = in_.cpu()[s].shape.num_elements();
      assert(len == out_.cpu()[s].shape.num_elements());
      for (int64_t i = 0; i < len; i++) {
         ASSERT_NEAR(naive_moving_mean_square(&in[0], &in[i], window_size), out[i], 1e-5)
            << "Failed @ " << i << " ref " << naive_moving_mean_square(&in[0], &in[i], window_size)
            << " vs. " << out[i] << "\n";
      }
    }
  }
};

using TestTypes =
    ::testing::Types<int16_t, uint16_t, int8_t, uint8_t, int32_t, uint32_t, float, double>;
TYPED_TEST_SUITE(MovingMeanSquareGPU, TestTypes);

TYPED_TEST(MovingMeanSquareGPU, RunTest) {
  this->RunTest();
}


TEST(MovingMeanSquareGPU, DISABLED_Benchmark) {
  using In = float;
  using Out = float;
  int nsamples = 64;
  int n_iters = 1000;

  int window_size = 2048;
  MovingMeanSquareArgs args{window_size};
  MovingMeanSquareGpu<In> kernel;


  TensorListShape<> sh(nsamples, 1);
  for (int s = 0; s < nsamples; s++) {
    if (s % 4 == 0)
      sh.tensor_shape_span(s)[0] = 16000 * 60;
    else if (s % 4 == 1)
      sh.tensor_shape_span(s)[0] = 16000 * 120;
    else if (s % 4 == 2)
      sh.tensor_shape_span(s)[0] = 16000 * 30;
    else if (s % 4 == 3)
      sh.tensor_shape_span(s)[0] = 16000 * 90;
  }

  TestTensorList<In> in_data;
  in_data.reshape(sh);

  TestTensorList<Out> out_data;
  out_data.reshape(sh);

  std::mt19937 rng;
  UniformRandomFill(in_data.cpu(), rng, 0.0, 1.0);

  CUDAEvent start = CUDAEvent::CreateWithFlags(0);
  CUDAEvent end = CUDAEvent::CreateWithFlags(0);
  double total_time_ms = 0;
  int64_t in_elems = in_data.cpu().shape.num_elements();
  int64_t out_elems = out_data.cpu().shape.num_elements();
  int64_t in_bytes = in_elems * sizeof(In);
  int64_t out_bytes = out_elems * sizeof(Out);
  std::cout << "Resampling GPU Perf test.\n"
            << "Input contains " << in_elems << " elements.\n";

  KernelContext ctx;
  ctx.gpu.stream = 0;

  auto out = out_data.gpu().to_static<1>();
  auto in = in_data.gpu().to_static<1>();

  for (int i = 0; i < n_iters; ++i) {
    CUDA_CALL(cudaDeviceSynchronize());

    DynamicScratchpad dyn_scratchpad({}, AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;

    CUDA_CALL(cudaEventRecord(start));
    kernel.Run(ctx, out, in, args);
    CUDA_CALL(cudaEventRecord(end));
    CUDA_CALL(cudaDeviceSynchronize());
    float time_ms;
    CUDA_CALL(cudaEventElapsedTime(&time_ms, start, end));
    total_time_ms += time_ms;
  }
  std::cout << "Bandwidth: " << n_iters * (in_bytes + out_bytes) / (total_time_ms * 1e6)
            << " GBs/sec" << std::endl;
}

}  // namespace test
}  // namespace signal
}  // namespace kernels
}  // namespace dali
