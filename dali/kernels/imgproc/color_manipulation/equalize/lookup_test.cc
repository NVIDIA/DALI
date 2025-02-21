// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <random>

#include "dali/kernels/common/utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/lookup.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"


namespace dali {
namespace kernels {
namespace equalize {
namespace lookup {
namespace test {

constexpr cudaStream_t cuda_stream = 0;

class EqualizeLookupGpuTest : public ::testing::Test {
 protected:
  static constexpr int range_size = 256;

  void PrepareData(const TensorListShape<2> &batch_shape) {
    baseline_.reshape(batch_shape);
    out_.reshape(batch_shape);
    in_.reshape(batch_shape);
    lut_.reshape(GetLutShape(batch_shape));
    PrepareBaseline();
  }

  TensorListShape<2> GetLutShape(const TensorListShape<2> &batch_shape) {
    int batch_size = batch_shape.num_samples();
    TensorListShape<2> ret(batch_size);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_channels = batch_shape[sample_idx][1];
      ret.set_tensor_shape(sample_idx, {num_channels, range_size});
    }
    return ret;
  }

  void PrepareBaseline() {
    auto in_view = in_.cpu();
    auto lut_view = lut_.cpu();
    UniformRandomFill(in_view, rng_, 0, 255);
    UniformRandomFill(lut_view, rng_, 0, 255);
    auto baseline_view = baseline_.cpu();
    for (int sample_idx = 0; sample_idx < baseline_view.shape.num_samples(); sample_idx++) {
      auto sample_in = in_view[sample_idx];
      auto sample_lut = lut_view[sample_idx];
      auto sample_baseline = baseline_view[sample_idx];
      auto width = sample_in.shape[0];
      auto num_channels = sample_in.shape[1];
      for (int64_t pixel_idx = 0; pixel_idx < width; pixel_idx++) {
        for (int64_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
          int64_t idx = pixel_idx * num_channels + channel_idx;
          sample_baseline.data[idx] =
              sample_lut.data[channel_idx * range_size + sample_in.data[idx]];
        }
      }
    }
  }

  void Run(TensorListShape<2> batch_shape) {
    PrepareData(batch_shape);
    LookupKernelGpu kernel;
    KernelContext ctx;
    ctx.gpu.stream = cuda_stream;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;
    auto out_view = out_.gpu(cuda_stream);
    auto in_view = in_.gpu(cuda_stream);
    auto lut_view = lut_.gpu(cuda_stream);
    kernel.Run(ctx, out_view, in_view, lut_view);
    auto out_view_cpu = out_.cpu(cuda_stream);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    Check(out_view_cpu, baseline_.cpu());
  }

  TestTensorList<uint8_t, 2> in_;
  TestTensorList<uint8_t, 2> lut_;
  TestTensorList<uint8_t, 2> baseline_, out_;
  std::mt19937_64 rng_{12345};
};

TEST_F(EqualizeLookupGpuTest, Channel1Batch1) {
  this->Run(TensorListShape<2>{{{1024 * 1024 - 7, 1}}});
}

TEST_F(EqualizeLookupGpuTest, Channel3Batch1) {
  this->Run(TensorListShape<2>{{{511 * 307, 3}}});
}

TEST_F(EqualizeLookupGpuTest, Channel3Batch7) {
  TensorListShape<2> tls{
      {{1, 3}, {101, 3}, {205, 3}, {4096, 3}, {4096, 3}, {4096, 3}, {4096 - 5, 3}}};
  this->Run(tls);
}

TEST_F(EqualizeLookupGpuTest, VarChannels) {
  TensorListShape<2> tls{
      {{1, 1}, {101, 7}, {205, 4}, {511 * 702, 2}, {4096, 4}, {4096, 4}, {4096 - 5, 11}}};
  this->Run(tls);
}

}  // namespace test
}  // namespace lookup
}  // namespace equalize
}  // namespace kernels
}  // namespace dali
