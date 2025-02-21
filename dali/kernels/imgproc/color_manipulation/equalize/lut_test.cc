// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "dali/kernels/common/utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/color_manipulation/equalize/lut.h"
#include "dali/pipeline/data/sequence_utils.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {
namespace equalize {
namespace lut {
namespace test {
constexpr cudaStream_t cuda_stream = 0;

class EqualizeLutGpuTest : public ::testing::Test {
 protected:
  static constexpr int range_size = 256;

  void Run() {
    PrepareBaseline();
    LutKernelGpu kernel;
    KernelContext ctx;
    ctx.gpu.stream = cuda_stream;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;
    auto out_view = out_.gpu(cuda_stream);
    auto in_view = in_.gpu(cuda_stream);
    kernel.Run(ctx, out_view, in_view);
    auto out_view_cpu = out_.cpu(cuda_stream);
    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    auto baseline_view = baseline_.cpu();
    // there may be rounding discrepancies as computation is carried with doubles
    // then converted to uint8's, thus the `EqualEps(1)`
    Check(out_view_cpu, baseline_view, EqualEps(1));
  }

  void RandomUniformHist(std::vector<std::vector<int>> num_leading_zeros_per_channel) {
    int batch_size = num_leading_zeros_per_channel.size();
    TensorListShape<2> batch_shape(batch_size);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto& desc = num_leading_zeros_per_channel[sample_idx];
      int num_channels = desc.size();
      batch_shape.set_tensor_shape(sample_idx, {num_channels, range_size});
    }
    in_.reshape(batch_shape);
    auto in_view = in_.cpu();
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto& num_leading_zeros = num_leading_zeros_per_channel[sample_idx];
      int num_channels = num_leading_zeros.size();
      auto channels_range = sequence_utils::unfolded_view_range<1>(in_view[sample_idx]);
      ASSERT_EQ(channels_range.NumSlices(), num_channels);
      for (int channel_idx = 0; channel_idx < num_channels; channel_idx++) {
        auto channel_view = channels_range[channel_idx];
        int channel_leading_zeros = num_leading_zeros[channel_idx];
        for (int idx = 0; idx < channel_leading_zeros; idx++) {
          channel_view.data[idx] = 0;
        }
        if (channel_leading_zeros < range_size) {
          channel_view.data[channel_leading_zeros] = dist1_(rng_);
        }
        for (int idx = num_leading_zeros[channel_idx] + 1; idx < range_size; idx++) {
          channel_view.data[idx] = dist0_(rng_);
        }
      }
    }
  }

  void PrepareBaseline() {
    auto in_view = in_.cpu();
    baseline_.reshape(in_view.shape);
    intermediate_.reshape(in_view.shape);
    out_.reshape(in_view.shape);
    auto intermediate_view = intermediate_.cpu();
    auto baseline_view = baseline_.cpu();
    for (int sample_idx = 0; sample_idx < baseline_view.shape.num_samples(); sample_idx++) {
      auto channel_ranges = sequence_utils::unfolded_views_range<1>(
          baseline_view[sample_idx], intermediate_view[sample_idx], in_view[sample_idx]);
      for (auto&& [baseline, workspace, in] : channel_ranges) {
        workspace.data[0] = in.data[0];
        for (int i = 1; i < range_size; i++) {
          workspace.data[i] = in.data[i] + workspace.data[i - 1];
        }
        int first_non_zero = 0;
        for (; first_non_zero < range_size; first_non_zero++) {
          if (workspace.data[first_non_zero] > 0) {
            break;
          }
        }
        auto total = workspace.data[range_size - 1];
        if (first_non_zero == range_size || workspace.data[first_non_zero] == total) {
          std::iota(baseline.data, baseline.data + range_size, 0);
        } else {
          auto first_val = workspace.data[first_non_zero];
          for (int i = 0; i < first_non_zero; i++) {
            // those values are irrelevant and arbitrary - they are not present in the
            // data that histogram was computed on, so they won't be ever used in the lookup
            baseline.data[i] = 255;
          }
          double scale = (range_size - 1.) / (total - first_val);
          for (int i = first_non_zero; i < range_size; i++) {
            baseline.data[i] = ConvertSat<uint8_t>((workspace.data[i] - first_val) * scale);
          }
        }
      }
    }
  }

  TestTensorList<uint64_t, 2> in_, intermediate_;
  TestTensorList<uint8_t, 2> baseline_, out_;
  std::mt19937_64 rng_{12345};
  std::uniform_int_distribution<uint64_t> dist1_{1, 1024_i64 * 1024 * 1024 * 1024};
  std::uniform_int_distribution<uint64_t> dist0_{0, 1024_i64 * 1024 * 1024 * 1024};
};

TEST_F(EqualizeLutGpuTest, Channel3Batch5) {
  this->RandomUniformHist({{0, 0, 0}, {1, 0, 0}, {1, 41, 3}, {255, 256, 0}, {256, 1, 1}});
  this->Run();
}

TEST_F(EqualizeLutGpuTest, VarChannelsBatch7) {
  this->RandomUniformHist(
      {{0, 1, 2, 3}, {42, 0, 255}, {5, 4, 101, 0, 0, 0}, {5}, {0}, {7, 2}, {1}});
  this->Run();
}

TEST_F(EqualizeLutGpuTest, AllTheSame) {
  TensorListShape<2> batch_shape{
      {{4, range_size}, {3, range_size}, {2, range_size}, {1, range_size}}};
  in_.reshape(batch_shape);
  auto in_view = in_.cpu();
  for (int sample_idx = 0; sample_idx < batch_shape.num_samples(); sample_idx++) {
    for (int i = 0; i < batch_shape[sample_idx].num_elements(); i++) {
      in_view[sample_idx].data[i] = sample_idx * 51;
    }
  }
  this->Run();
}

TEST_F(EqualizeLutGpuTest, SinglePoint) {
  TensorListShape<2> batch_shape{
      {{2, range_size}, {5, range_size}, {4, range_size}, {1, range_size}}};
  in_.reshape(batch_shape);
  auto in_view = in_.cpu();
  for (int sample_idx = 0; sample_idx < batch_shape.num_samples(); sample_idx++) {
    for (int channel_idx = 0; channel_idx < batch_shape[sample_idx][0]; channel_idx++) {
      ASSERT_EQ(batch_shape[sample_idx][1], range_size);
      int nonzero_idx = 71 * sample_idx + channel_idx;
      ASSERT_LT(nonzero_idx, range_size);
      int nonzero_value = (sample_idx << 11) + channel_idx;
      for (int i = 0; i < range_size; i++) {
        in_view[sample_idx].data[channel_idx * range_size + i] =
            i == nonzero_idx ? nonzero_value : 0;
      }
    }
  }
  this->Run();
}

}  // namespace test
}  // namespace lut
}  // namespace equalize
}  // namespace kernels
}  // namespace dali
