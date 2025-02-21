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
#include <limits>
#include <random>
#include <vector>

#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/common/utils.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/imgproc/color_manipulation/debayer/debayer.h"
#include "dali/kernels/imgproc/color_manipulation/debayer/debayer_npp.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace kernels {
namespace debayer {
namespace test {

constexpr cudaStream_t cuda_stream = 0;

template <typename InOutT_, DALIDebayerAlgorithm alg_>
struct DebayerTestParams {
  using InOutT = InOutT_;
  static constexpr DALIDebayerAlgorithm alg = alg_;
};

template <typename DebayerTestParamsT>
class DebayerGpuTest : public ::testing::Test {
 protected:
  using InOutT = typename DebayerTestParamsT::InOutT;
  using Kernel = NppDebayerKernel<InOutT>;
  static constexpr int num_channels = 3;
  static_assert(DebayerTestParamsT::alg == DALIDebayerAlgorithm::DALI_DEBAYER_BILINEAR_NPP);

  void FillWithGradient(TensorListView<StorageCPU, InOutT, 3> rgb_batch) {
    int max_val = std::numeric_limits<InOutT>::max();
    static constexpr int num_channels = 3;
    const auto &batch_shape = rgb_batch.shape;
    for (int sample_idx = 0; sample_idx < rgb_batch.num_samples(); sample_idx++) {
      int height = batch_shape[sample_idx][0];
      int width = batch_shape[sample_idx][1];
      auto rgb_sample = rgb_batch[sample_idx];
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          rgb_sample.data[h * num_channels * width + w * num_channels] = max_val * (w + 1) / width;
        }
      }
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          rgb_sample.data[h * num_channels * width + w * num_channels + 1] =
              max_val * (h + 1) / height;
        }
      }
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          rgb_sample.data[h * num_channels * width + w * num_channels + 2] =
              max_val * (width - w) / width;
        }
      }
    }
  }

  void BayerSamples(TensorListView<StorageCPU, InOutT, 2> bayer_batch,
                    TensorListView<StorageCPU, const InOutT, 3> rgb_batch) {
    // Note that dali uses opncv's convention of naming the patterns, which
    // looks at the 2x2 tile starting at the second row and column of the sensors matrix.
    // We iterate over whole image, so the patterns are first transformed as if by looking
    // at the tile starting at first column and row.
    int pattern2channel[4][2][2] = {
        {{0, 1}, {1, 2}},  // bggr -> rggb -> 0112
        {{1, 0}, {2, 1}},  // gbrg -> grbg -> 1021
        {{1, 2}, {0, 1}},  // grbg -> gbrg -> 1201
        {{2, 1}, {1, 0}}   // rggb -> bggr -> 2110
    };
    int batch_size = rgb_batch.num_samples();
    const auto &batch_shape = rgb_batch.shape;
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      int height = batch_shape[sample_idx][0];
      int width = batch_shape[sample_idx][1];
      auto pattern = patterns_[sample_idx];
      auto rgb_sample = rgb_batch[sample_idx];
      auto bayer_sample = bayer_batch[sample_idx];
      ASSERT_EQ(height % 2, 0);
      ASSERT_EQ(width % 2, 0);
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int i = h & 1;
          int j = w & 1;
          int c = pattern2channel[static_cast<int>(pattern)][i][j];
          bayer_sample.data[h * width + w] =
              rgb_sample.data[h * width * num_channels + w * num_channels + c];
        }
      }
    }
  }

  void PrepareData(int batch_size, int min_extent, int max_extent) {
    std::uniform_int_distribution<> shape_dist{min_extent / 2, max_extent / 2};
    std::uniform_int_distribution<> pattern_dist{0, 3};
    TensorListShape<3> batch_shape(batch_size);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      TensorShape<3> sample_shape{2 * shape_dist(rng), 2 * shape_dist(rng), 3};
      batch_shape.set_tensor_shape(sample_idx, sample_shape);
      patterns_.push_back(static_cast<DALIBayerPattern>(pattern_dist(rng)));
    }
    baseline_.reshape(batch_shape);
    out_.reshape(batch_shape);
    in_.reshape(batch_shape.first<2>());
    auto baseline_view = baseline_.cpu();
    FillWithGradient(baseline_view);
    auto in_view = in_.cpu();
    BayerSamples(in_view, baseline_view);
  }

  void Run(int batch_size, int min_extent, int max_extent) {
    PrepareData(batch_size, min_extent, max_extent);
    Kernel kernel{0};
    KernelContext ctx;
    ctx.gpu.stream = cuda_stream;
    DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
    ctx.scratchpad = &dyn_scratchpad;
    auto in_view = in_.gpu(cuda_stream);
    auto out_view = out_.gpu(cuda_stream);
    kernel.Run(ctx, out_view, in_view, make_span(patterns_));
    auto out_view_cpu = out_.cpu(cuda_stream);
    int max_val = std::numeric_limits<InOutT>::max();
    int grad_step = (max_val + min_extent - 1) / min_extent;
    Check(out_view_cpu, baseline_.cpu(), EqualEps(grad_step));
  }

  std::vector<DALIBayerPattern> patterns_;
  TestTensorList<InOutT, 2> in_;
  TestTensorList<InOutT, 3> baseline_, out_;
  std::mt19937_64 rng{12345};
};

using TestParams =
    ::testing::Types<DebayerTestParams<uint8_t, DALIDebayerAlgorithm::DALI_DEBAYER_BILINEAR_NPP>,
                     DebayerTestParams<uint16_t, DALIDebayerAlgorithm::DALI_DEBAYER_BILINEAR_NPP>>;

TYPED_TEST_SUITE(DebayerGpuTest, TestParams);

TYPED_TEST(DebayerGpuTest, Gradient_1) {
  this->Run(1, 256, 400);
}

TYPED_TEST(DebayerGpuTest, Gradient_32) {
  this->Run(32, 256, 400);
}

TYPED_TEST(DebayerGpuTest, Gradient_200) {
  this->Run(200, 256, 300);
}

}  // namespace test
}  // namespace debayer
}  // namespace kernels
}  // namespace dali
