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
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace kernels {
namespace signal {
namespace resampling {
namespace test {

class ResamplingGPUTest : public ResamplingTest {
 public:
  ResamplingGPUTest() {
    this->nsamples_ = 8;
  }

  void RunResampling(span<const Args> args) override {
    ResamplerGPU<float> R;
    R.Initialize(16);

    KernelContext ctx;
    ctx.gpu.stream = 0;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;

    auto req = R.Setup(ctx, ttl_in_.gpu(), args);
    auto outref_sh = ttl_outref_.cpu().shape;
    auto in_batch_sh = ttl_in_.cpu().shape;
    for (int s = 0; s < outref_sh.size(); s++) {
      auto sh = req.output_shapes[0].tensor_shape_span(s);
      auto expected_sh = outref_sh.tensor_shape_span(s);
      ASSERT_EQ(sh, expected_sh);
    }

    R.Run(ctx, ttl_out_.gpu(), ttl_in_.gpu(), args);

    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
  }

  void RunPerfTest(int n_iters = 1000) {
    std::vector<Args> args_v(nsamples_, {22050.0f, 16000.0f});
    auto args = make_cspan(args_v);
    this->nsec_ = 30;
    this->PrepareData(args);

    ResamplerGPU<float> R;
    R.Initialize(16);

    KernelContext ctx;
    ctx.gpu.stream = 0;
    auto req = R.Setup(ctx, ttl_in_.gpu(), args);
    ASSERT_EQ(ttl_out_.cpu().shape, req.output_shapes[0]);

    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);
    double total_time_ms = 0;
    int64_t in_elems = ttl_in_.cpu().shape.num_elements();
    int64_t out_elems = ttl_out_.cpu().shape.num_elements();
    int64_t out_bytes = out_elems * sizeof(float);
    std::cout << "Resampling GPU Perf test.\n"
              << "Input contains " << in_elems << " floats.\n"
              << "Output contains " << out_elems << " floats.\n";

    for (int i = 0; i < n_iters; ++i) {
      CUDA_CALL(cudaDeviceSynchronize());

      DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
      ctx.scratchpad = &dyn_scratchpad;

      CUDA_CALL(cudaEventRecord(start));
      R.Run(ctx, ttl_out_.gpu(), ttl_in_.gpu(), args);
      CUDA_CALL(cudaEventRecord(end));
      CUDA_CALL(cudaDeviceSynchronize());
      float time_ms;
      CUDA_CALL(cudaEventElapsedTime(&time_ms, start, end));
      total_time_ms += time_ms;
    }
    std::cout << "Processed " << n_iters * out_bytes / (total_time_ms * 1e6) << " GBs/sec"
              << std::endl;
  }
};

TEST_F(ResamplingGPUTest, SingleChannel) {
  this->nchannels_ = 1;
  this->RunTest();
}

TEST_F(ResamplingGPUTest, TwoChannel) {
  this->nchannels_ = 2;
  this->RunTest();
}

TEST_F(ResamplingGPUTest, EightChannel) {
  this->nchannels_ = 8;
  this->RunTest();
}

TEST_F(ResamplingGPUTest, ThirtyChannel) {
  this->nchannels_ = 30;
  this->RunTest();
}

TEST_F(ResamplingGPUTest, OutBeginEnd) {
  this->roi_start_ = 100;
  this->roi_end_ = 8000;
  this->RunTest();
}

TEST_F(ResamplingGPUTest, EightChannelOutBeginEnd) {
  this->roi_start_ = 100;
  this->roi_end_ = 8000;
  this->nchannels_ = 8;
  this->RunTest();
}

TEST_F(ResamplingGPUTest, PerfTest) {
  this->RunPerfTest(1000);
}

TEST_F(ResamplingGPUTest, SingleChannelNeedHighPrecision) {
  this->default_freq_in_ = 0.49;
  this->nsec_ = 400;
  this->roi_start_ = 4000000;  // enough to look long into the signal
  this->roi_end_ = 4010000;
  this->RunTest();
}

TEST_F(ResamplingGPUTest, ThreeChannelNeedHighPrecision) {
  this->default_freq_in_ = 0.49;
  this->nsec_ = 400;
  this->nchannels_ = 3;
  this->roi_start_ = 4000000;  // enough to look long into the signal
  this->roi_end_ = 4010000;
  this->RunTest();
}

}  // namespace test
}  // namespace resampling
}  // namespace signal
}  // namespace kernels
}  // namespace dali
