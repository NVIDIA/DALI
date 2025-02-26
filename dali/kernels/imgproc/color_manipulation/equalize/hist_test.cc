// Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/common/utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/hist.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace hist {
namespace test {

constexpr cudaStream_t cuda_stream = 0;

class EqualizeHistGpuTest : public ::testing::Test {
 protected:
  static constexpr int range_size = 256;

  void Run() {
    PrepareBaseline();
    HistogramKernelGpu kernel;
    KernelContext ctx;
    ctx.gpu.stream = cuda_stream;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;
    auto out_view = out_.gpu(cuda_stream);
    auto in_view = in_.gpu(cuda_stream);
    kernel.Run(ctx, out_view, in_view);
    out_.invalidate_cpu();
    auto out_view_cpu = out_.cpu(cuda_stream);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    auto baseline_view = baseline_.cpu();
    Check(out_view_cpu, baseline_view);
  }

  void PrepareBaseline() {
    auto in_view = in_.cpu();
    int batch_size = in_view.shape.num_samples();
    TensorListShape<2> out_batch_shape(batch_size);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_channels = in_view.shape[sample_idx][1];
      out_batch_shape.set_tensor_shape(sample_idx, {num_channels, range_size});
    }
    baseline_.reshape(out_batch_shape);
    out_.reshape(out_batch_shape);
    auto out_view = out_.cpu();
    // fill output with non-zero value to make sure that the kernel does not rely on zeroed memory
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      for (int64_t i = 0; i < out_view[sample_idx].num_elements(); i++) {
        out_view[sample_idx].data[i] = 3;
      }
    }
    auto baseline_view = baseline_.cpu();
    for (int sample_idx = 0; sample_idx < baseline_view.shape.num_samples(); sample_idx++) {
      auto sample_baseline_view = baseline_view[sample_idx];
      auto sample_in_view = in_view[sample_idx];
      auto num_channels = sample_in_view.shape[1];
      ASSERT_EQ(num_channels, sample_baseline_view.shape[0]);
      for (int64_t idx = 0; idx < sample_baseline_view.num_elements(); idx++) {
        sample_baseline_view.data[idx] = 0;
      }
      for (int64_t pixel_idx = 0; pixel_idx < sample_in_view.shape[0]; pixel_idx++) {
        for (int64_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
          auto val = sample_in_view.data[pixel_idx * num_channels + channel_idx];
          ++sample_baseline_view.data[channel_idx * range_size + val];
        }
      }
    }
  }

  void PrepareUniformRandom(TensorListShape<2> batch_shape) {
    in_.reshape(batch_shape);
    UniformRandomFill(in_.cpu(), rng_, 0, 255);
  }

  TestTensorList<uint8_t, 2> in_;
  TestTensorList<uint64_t, 2> baseline_, out_;
  std::mt19937_64 rng_{12345};
};

TEST_F(EqualizeHistGpuTest, UniformRandChannel1Batch1) {
  this->PrepareUniformRandom(TensorListShape<2>{{{1024 * 1024 - 7, 1}}});
  this->Run();
}

TEST_F(EqualizeHistGpuTest, UniformRandChannel3Batch1) {
  this->PrepareUniformRandom(TensorListShape<2>{{{511 * 307, 3}}});
  this->Run();
}

TEST_F(EqualizeHistGpuTest, UniformRandVarChannels) {
  TensorListShape<2> tls{
      {{1, 1}, {101, 7}, {205, 4}, {511 * 702, 2}, {4096, 4}, {4096, 4}, {4096 - 5, 11}}};
  this->PrepareUniformRandom(tls);
  this->Run();
}

TEST_F(EqualizeHistGpuTest, SingleValue) {
  TensorListShape<2> batch_shape{{{1024 * 1024 * 64, 1},
                                  {1024 * 1024 * 64 - 1, 2},
                                  {1024 * 1024 * 64 - 2, 3},
                                  {1024 * 1024 * 64 - 3, 4},
                                  {1024 * 1024 * 64 - 4, 5},
                                  {1024 * 1024 * 64 - 4, 6}}};
  in_.reshape(batch_shape);
  auto in_view = in_.cpu();
  for (int sample_idx = 0; sample_idx < in_view.num_samples(); sample_idx++) {
    auto sample_in_view = in_view[sample_idx];
    for (int64_t idx = 0; idx < batch_shape[0].num_elements(); idx++) {
      sample_in_view.data[idx] = 51 * sample_idx;
    }
  }
  this->Run();
}

}  // namespace test
}  // namespace hist
}  // namespace equalize
}  // namespace kernels
}  // namespace dali
