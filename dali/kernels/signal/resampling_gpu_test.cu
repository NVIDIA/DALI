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

#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "dali/kernels/signal/resampling_gpu.h"
#include "dali/kernels/signal/resampling_test.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace kernels {
namespace signal {
namespace resampling {
namespace test {

class ResamplingGPUTest : public ResamplingTest {
 public:
  void RunResampling(span<const float> in_rates, span<const float> out_rates) override {
    ResamplerGPU<float> R;
    R.Initialize(16);

    KernelContext ctx;
    ctx.gpu.stream = 0;

    auto req = R.Setup(ctx, ttl_in_.gpu(), in_rates, out_rates);
    auto outref_sh = ttl_outref_.cpu().shape;
    auto in_batch_sh = ttl_in_.cpu().shape;
    for (int s = 0; s < outref_sh.size(); s++) {
      auto sh = req.output_shapes[0].tensor_shape_span(s);
      auto expected_sh = outref_sh.tensor_shape_span(s);
      ASSERT_EQ(sh, expected_sh);
    }

    R.Run(ctx, ttl_out_.gpu(), ttl_in_.gpu(), in_rates, out_rates);

    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
  }

  void RunPerfTest(int batch_size, int nchannels, int n_iters = 1000) {
    std::vector<float> in_rates_v(batch_size, 22050.0f);
    auto in_rates = make_cspan(in_rates_v);
    std::vector<float> out_rates_v(batch_size, 16000.0f);
    auto out_rates = make_cspan(out_rates_v);

    this->PrepareData(batch_size, nchannels, in_rates, out_rates);

    ResamplerGPU<float> R;
    R.Initialize(16);

    KernelContext ctx;
    ctx.gpu.stream = 0;

    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);
    double avg_time = 0;
    int64_t in_elems = ttl_in_.cpu().shape.num_elements();
    int64_t in_bytes = in_elems * sizeof(float);
    std::cout << "Resampling GPU Perf test.\n"
              << "\nInput contains " << in_elems << " floats.\n";

    auto req = R.Setup(ctx, ttl_in_.gpu(), in_rates, out_rates);
    ASSERT_EQ(ttl_out_.cpu().shape, req.output_shapes[0]);

    for (int i = 0; i < n_iters; ++i) {
      CUDA_CALL(cudaEventRecord(start));
      R.Run(ctx, ttl_out_.gpu(), ttl_in_.gpu(), in_rates, out_rates);
      CUDA_CALL(cudaEventRecord(end));
      CUDA_CALL(cudaDeviceSynchronize());
      float time;
      CUDA_CALL(cudaEventElapsedTime(&time, start, end));

      avg_time += time;
    }
    std::cout << "Processed " << in_bytes / avg_time << " bytes/sec" << std::endl;
  }
};

TEST_F(ResamplingGPUTest, SingleChannel) {
  this->RunTest(8, 1);
}

TEST_F(ResamplingGPUTest, TwoChannel) {
  this->RunTest(3, 2);
}

TEST_F(ResamplingGPUTest, EightChannel) {
  this->RunTest(3, 8);
}

TEST_F(ResamplingGPUTest, PerfTest) {
  this->RunPerfTest(64, 1, 1000);
}

}  // namespace test
}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali
