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
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include "dali/kernels/imgproc/resample/separable.h"
#include "dali/kernels/test/test_tensors.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/test/resampling_test/resampling_test_params.h"

using std::cout;
using std::endl;

namespace dali {
namespace kernels {
namespace resample_test {

namespace {
void RandomParams(
    TensorListShape<3> &tls,
    std::vector<ResamplingParams2D> &params,
    int num_samples) {
  std::mt19937_64 rng;
  auto size_dist    = uniform_distribution(128, 2048);
  auto channel_dist = uniform_distribution(1, 4);
  auto aspect_dist  = uniform_distribution(0.5f, 2.0f);

  tls.resize(num_samples);
  params.resize(num_samples);
  for (int i = 0; i < num_samples; i++) {
    auto ts = tls.tensor_shape_span(i);
    float aspect = sqrt(aspect_dist(rng));
    ts[0] = size_dist(rng) * aspect;
    ts[1] = size_dist(rng) / aspect;
    ts[2] = channel_dist(rng);
    aspect = sqrt(aspect_dist(rng));
    params[i][0].output_size = size_dist(rng) * aspect;
    params[i][1].output_size = size_dist(rng) / aspect;
    params[i][0].min_filter.type = ResamplingFilterType::Triangular;
    params[i][0].mag_filter.type = ResamplingFilterType::Linear;
    params[i][1].min_filter.type = ResamplingFilterType::Triangular;
    params[i][1].mag_filter.type = ResamplingFilterType::Linear;
  }
}
}  // namespace

TEST(SeparableImpl, Setup) {
  int N = 32;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  SeparableResamplingGPUImpl<uint8_t, uint8_t> resampling;
  TestTensorList<uint8_t, 3> input, output;

  TensorListShape<3> tls;
  std::vector<ResamplingParams2D> params;
  RandomParams(tls, params, N);

  input.reshape(tls);

  InListGPU<uint8_t, 3> in_tv = input.gpu();

  auto req = resampling.Setup(ctx, in_tv, make_span(params));
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  output.reshape(req.output_shapes[0].to_static<3>());
  OutListGPU<uint8_t, 3> out_tv = output.gpu();

  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      tls.tensor_shape_span(i)[2]
    };
    EXPECT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);

    EXPECT_EQ(out_tv.offsets[i], resampling.setup.sample_descs[i].offsets[2]);

    EXPECT_GE(resampling.setup.sample_descs[i].block_count.pass[0], 1);
    EXPECT_GE(resampling.setup.sample_descs[i].block_count.pass[1], 1);
  }
  EXPECT_GT(resampling.setup.total_blocks.pass[0], N);
  EXPECT_GT(resampling.setup.total_blocks.pass[1], N);
}

ResamplingTestBatch SingleImageBatch = {
  {
    "imgproc_test/alley.png", "imgproc_test/ref_out/alley_tri_300x300.png",
    { 300, 300 }, tri(), 5
  }
};

ResamplingTestBatch Batch1 = {
  {
    "imgproc_test/alley.png", "imgproc_test/ref_out/alley_tri_300x300.png",
    { 300, 300 }, tri(), 5
  },
  {
    "imgproc_test/score.png", "imgproc_test/ref_out/score_lanczos3.png",
    { 540, 250 }, lanczos(), 1
  },
  {
    "imgproc_test/score.png", "imgproc_test/ref_out/score_cubic.png",
    { 200, 93 }, cubic(), 1
  },
  {
    "imgproc_test/alley.png", "imgproc_test/ref_out/alley_blurred.png",
    { 681, 960 }, gauss(12), 2
  }
};

ResamplingTestBatch CropBatch = {
  {
    "imgproc_test/dots.png", "imgproc_test/ref_out/dots_crop_2x2.png",
    { 1.0f, 1.0f, 3.0f, 3.0f }, { 2, 2 }, nearest(), 0
  },
  {
    "imgproc_test/alley.png", "imgproc_test/ref_out/alley_cubic_crop.png",
    { 100.0f, 300.0f, 200.0f, 400.0f }, { 200, 200 }, cubic(), 1
  },
  {
    "imgproc_test/alley.png", "imgproc_test/ref_out/alley_cubic_crop_flip.png",
    { 200.0f, 300.0f, 100.0f, 400.0f }, { 200, 200 }, cubic(), 1
  },
  {
    "imgproc_test/alley.png", "imgproc_test/ref_out/alley_linear_crop.png",
    { 150.0f, 400.0f, 200.0f, 450.0f }, { 150, 200 }, lin(), 1
  },
  {
    "imgproc_test/alley.png", "imgproc_test/ref_out/alley_linear_crop_flip.png",
    { 150.0f, 450.0f, 200.0f, 400.0f }, { 150, 200 }, lin(), 1
  },
};


class BatchResamplingTest : public ::testing::Test,
                            public ::testing::WithParamInterface<ResamplingTestBatch> {
};

TEST_P(BatchResamplingTest, ResamplingImpl) {
  const ResamplingTestBatch &batch = GetParam();

  int N = batch.size();
  std::vector<cv::Mat> cv_img(N);
  std::vector<cv::Mat> cv_ref(N);
  std::vector<ResamplingParams2D> params(N);

  for (int i = 0; i < N; i++) {
    cv_img[i] = testing::data::image(batch[i].input.c_str());
    cv_ref[i] = testing::data::image(batch[i].reference.c_str());
    params[i] = batch[i].params;
  }

  ScratchpadAllocator scratch_alloc;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  SeparableResamplingGPUImpl<uint8_t, uint8_t> resampling;
  TestTensorList<uint8_t, 3> input, output;

  FilterDesc tri;
  tri.type = ResamplingFilterType::Triangular;
  FilterDesc lanczos;
  lanczos.type = ResamplingFilterType::Lanczos3;

  std::vector<TensorShape<3>> shapes;
  for (int i = 0; i < N; i++) {
    shapes.push_back(tensor_shape<3>(cv_img[i]));
  }
  input.reshape(shapes);
  OutListGPU<uint8_t, 3> in_tv = input.gpu();
  for (int i = 0; i < N; i++) {
    copy(in_tv[i], view_as_tensor<uint8_t, 3>(cv_img[i]));
  }

  auto req = resampling.Setup(ctx, in_tv, make_span(params));
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  scratch_alloc.Reserve(req.scratch_sizes);

  output.reshape(req.output_shapes[0].to_static<3>());
  OutListGPU<uint8_t, 3> out_tv = output.gpu();
  cudaMemset(out_tv.data, 0, out_tv.num_elements()*sizeof(*out_tv.data));

  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      in_tv.shape.tensor_shape_span(i)[2]
    };
    ASSERT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);

    if (!params[i][0].roi.use_roi || !params[i][1].roi.use_roi) {
      EXPECT_EQ(in_tv.offsets[i], resampling.setup.sample_descs[i].offsets[0]);
      EXPECT_EQ(resampling.intermediate.offsets[i], resampling.setup.sample_descs[i].offsets[1]);
    }
    EXPECT_EQ(out_tv.offsets[i], resampling.setup.sample_descs[i].offsets[2]);

    EXPECT_GE(resampling.setup.sample_descs[i].block_count.pass[0], 1);
    EXPECT_GE(resampling.setup.sample_descs[i].block_count.pass[1], 1);
  }

  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  resampling.Run(ctx, out_tv, in_tv, make_span(params));
  for (int i = 0; i < N; i++) {
    auto ref_tensor = view_as_tensor<uint8_t, 3>(cv_ref[i]);
    auto out_tensor = output.cpu()[i];
    ASSERT_NO_FATAL_FAILURE(Check(out_tensor, ref_tensor, EqualEps(batch[i].epsilon)))
    << [&]() {
      cv::Mat tmp(out_tensor.shape[0], out_tensor.shape[1], CV_8UC3, out_tensor.data);
      std::string inp_name = batch[i].input;
      int ext = inp_name.rfind('.');
#ifdef WINVER
      constexpr char sep = '\\';
#else
      constexpr char sep = '/';
#endif
      int start = inp_name.rfind(sep) + 1;
      std::string dif_name = inp_name.substr(start, ext - start) + "_dif"+inp_name.substr(ext);
      cv::imwrite(dif_name, 127 + tmp - cv_ref[i]);
      return "Diff written to " + dif_name;
    }();
  }
}

TEST_P(BatchResamplingTest, ResamplingKernelAPI) {
  const ResamplingTestBatch &batch = GetParam();

  int N = batch.size();
  std::vector<cv::Mat> cv_img(N);
  std::vector<cv::Mat> cv_ref(N);
  std::vector<ResamplingParams2D> params(N);

  for (int i = 0; i < N; i++) {
    cv_img[i] = testing::data::image(batch[i].input.c_str());
    cv_ref[i] = testing::data::image(batch[i].reference.c_str());
    params[i] = batch[i].params;
  }

  ScratchpadAllocator scratch_alloc;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  using Kernel = ResampleGPU<uint8_t, uint8_t>;
  TestTensorList<uint8_t, 3> input, output;

  FilterDesc tri;
  tri.type = ResamplingFilterType::Triangular;
  FilterDesc lanczos;
  lanczos.type = ResamplingFilterType::Lanczos3;

  std::vector<TensorShape<3>> shapes;
  for (int i = 0; i < N; i++) {
    shapes.push_back(tensor_shape<3>(cv_img[i]));
  }
  input.reshape(shapes);
  OutListGPU<uint8_t, 3> in_tv = input.gpu();
  for (int i = 0; i < N; i++) {
    copy(in_tv[i], view_as_tensor<uint8_t, 3>(cv_img[i]));
  }

  auto req = Kernel::GetRequirements(ctx, in_tv, make_span(params));
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  scratch_alloc.Reserve(req.scratch_sizes);

  output.reshape(req.output_shapes[0].to_static<3>());
  OutListGPU<uint8_t, 3> out_tv = output.gpu();
  cudaMemset(out_tv.data, 0, out_tv.num_elements()*sizeof(*out_tv.data));

  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      in_tv.shape.tensor_shape_span(i)[2]
    };
    ASSERT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);
  }

  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  Kernel::Run(ctx, out_tv, in_tv, make_span(params));
  for (int i = 0; i < N; i++) {
    auto ref_tensor = view_as_tensor<uint8_t, 3>(cv_ref[i]);
    auto out_tensor = output.cpu()[i];
    ASSERT_NO_FATAL_FAILURE(Check(out_tensor, ref_tensor, EqualEps(batch[i].epsilon)))
    << [&]() {
      cv::Mat tmp(out_tensor.shape[0], out_tensor.shape[1], CV_8UC3, out_tensor.data);
      std::string inp_name = batch[i].input;
      int ext = inp_name.rfind('.');
#ifdef WINVER
      constexpr char sep = '\\';
#else
      constexpr char sep = '/';
#endif
      int start = inp_name.rfind(sep) + 1;
      std::string dif_name = inp_name.substr(start, ext - start) + "_dif"+inp_name.substr(ext);
      std::string out_name = inp_name.substr(start, ext - start) + "_out"+inp_name.substr(ext);
      cv::imwrite(dif_name, 127 + tmp - cv_ref[i]);
      cv::imwrite(out_name, tmp);
      return "Diff written to " + dif_name;
    }();
  }
}

INSTANTIATE_TEST_SUITE_P(SingleImage, BatchResamplingTest, ::testing::Values(SingleImageBatch));
INSTANTIATE_TEST_SUITE_P(MultipleImages, BatchResamplingTest, ::testing::Values(Batch1));

INSTANTIATE_TEST_SUITE_P(Crop, BatchResamplingTest, ::testing::Values(CropBatch));

}  // namespace resample_test
}  // namespace kernels
}  // namespace dali
