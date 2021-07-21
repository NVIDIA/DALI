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
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
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
template <int spatial_ndim>
void RandomParams(
    TensorListShape<spatial_ndim+1> &tls,
    std::vector<ResamplingParamsND<spatial_ndim>> &params,
    int num_samples) {
  std::mt19937_64 rng;
  auto size_dist    = uniform_distribution(
                        powf(1e+4f, 1.0f/spatial_ndim),
                        powf(1e+7f, 1.0f/spatial_ndim));
  auto channel_dist = uniform_distribution(1, 4);
  auto aspect_dist  = uniform_distribution(0.5f, 2.0f);

  tls.resize(num_samples);
  params.resize(num_samples);
  for (int i = 0; i < num_samples; i++) {
    auto ts = tls.tensor_shape_span(i);
    float avg_in_size = size_dist(rng);
    float avg_out_size = size_dist(rng);
    vec<spatial_ndim> in_aspect, out_aspect;
    float in_aspect_norm = 1, out_aspect_norm = 1;
    for (int i = 0; i < spatial_ndim; i++) {
      in_aspect[i] = sqrt(aspect_dist(rng));
      out_aspect[i] = sqrt(aspect_dist(rng));
      in_aspect_norm *= in_aspect[i];
      out_aspect_norm *= out_aspect[i];
    }
    in_aspect /= in_aspect_norm;
    out_aspect /= out_aspect_norm;
    for (int i = 0; i < spatial_ndim; i++)
      ts[i] = avg_in_size * in_aspect[i];
    ts[spatial_ndim] = channel_dist(rng);
    for (int d = 0; d < spatial_ndim; d++) {
      params[i][d].output_size = avg_out_size * out_aspect[d];
      params[i][d].min_filter.type = ResamplingFilterType::Triangular;
      params[i][d].mag_filter.type = ResamplingFilterType::Linear;
    }
  }
}
}  // namespace

template <int spatial_ndim>
void TestSetup() {
  constexpr int tensor_ndim = spatial_ndim + 1;
  int N = 32;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  SeparableResamplingGPUImpl<uint8_t, uint8_t, spatial_ndim> resampling;
  TestTensorList<uint8_t, tensor_ndim> input, output;

  TensorListShape<tensor_ndim> tls;
  std::vector<ResamplingParamsND<spatial_ndim>> params;
  RandomParams<spatial_ndim>(tls, params, N);

  input.reshape(tls);

  InListGPU<uint8_t, tensor_ndim> in_tv = input.gpu();

  auto req = resampling.Setup(ctx, in_tv, make_span(params));
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  output.reshape(req.output_shapes[0].template to_static<tensor_ndim>());
  OutListGPU<uint8_t, tensor_ndim> out_tv = output.gpu();

  for (int i = 0; i < N; i++) {
    TensorShape<tensor_ndim> expected_shape;
    for (int d = 0; d < spatial_ndim; d++)
      expected_shape[d] = params[i][d].output_size;

    expected_shape[spatial_ndim] = tls.tensor_shape_span(i)[spatial_ndim];

    EXPECT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);

    for (int d = 0; d < spatial_ndim; d++) {
      for (int pass = 0; pass < spatial_ndim; pass++) {
        EXPECT_GE(resampling.setup.sample_descs[i].logical_block_shape[pass][d], 1);
      }
    }
  }
  for (int pass = 0; pass < spatial_ndim; pass++) {
    EXPECT_GE(resampling.setup.total_blocks[pass], N);
  }
}

TEST(SeparableImpl, Setup2D) {
  TestSetup<2>();
}

TEST(SeparableImpl, Setup3D) {
  TestSetup<3>();
}

ResamplingTestBatch SingleImageBatch = {
  {
    "imgproc/alley.png", "imgproc/ref/resampling/alley_tri_300x300.png",
    { 300, 300 }, tri(), 5
  }
};

ResamplingTestBatch Batch1 = {
  {
    "imgproc/alley.png", "imgproc/ref/resampling/alley_tri_300x300.png",
    { 300, 300 }, tri(), 5
  },
  {
    "imgproc/score.png", "imgproc/ref/resampling/score_lanczos3.png",
    { 540, 250 }, lanczos(), 1
  },
  {
    "imgproc/score.png", "imgproc/ref/resampling/score_cubic.png",
    { 200, 93 }, cubic(), 1
  },
  {
    "imgproc/alley.png", "imgproc/ref/resampling/alley_blurred.png",
    { 681, 960 }, gauss(12), 2
  }
};

ResamplingTestBatch CropBatch = {
  {
    "imgproc/dots.png", "imgproc/ref/resampling/dots_crop_2x2.png",
    { 1.0f, 1.0f, 3.0f, 3.0f }, { 2, 2 }, nearest(), 0
  },
  {
    "imgproc/alley.png", "imgproc/ref/resampling/alley_cubic_crop.png",
    { 100.0f, 300.0f, 200.0f, 400.0f }, { 200, 200 }, cubic(), 1
  },
  {
    "imgproc/alley.png", "imgproc/ref/resampling/alley_cubic_crop_flip.png",
    { 200.0f, 300.0f, 100.0f, 400.0f }, { 200, 200 }, cubic(), 1
  },
  {
    "imgproc/alley.png", "imgproc/ref/resampling/alley_linear_crop.png",
    { 150.0f, 400.0f, 200.0f, 450.0f }, { 150, 200 }, lin(), 1
  },
  {
    "imgproc/alley.png", "imgproc/ref/resampling/alley_linear_crop_flip.png",
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
  SeparableResamplingGPUImpl<uint8_t, uint8_t, 2> resampling;
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
  OutListGPU<uint8_t, 3> in_tlv = input.gpu();
  for (int i = 0; i < N; i++) {
    copy(in_tlv[i], view_as_tensor<uint8_t, 3>(cv_img[i]));
  }

  auto req = resampling.Setup(ctx, in_tlv, make_span(params));
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  scratch_alloc.Reserve(req.scratch_sizes);

  output.reshape(req.output_shapes[0].to_static<3>());
  OutListGPU<uint8_t, 3> out_tlv = output.gpu();
  for (int i = 0; i < out_tlv.num_samples(); i++) {
    auto tv = out_tlv[i];
    CUDA_CALL(cudaMemset(tv.data, 0, tv.num_elements()*sizeof(*tv.data)));
  }


  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      in_tlv.shape.tensor_shape_span(i)[2]
    };
    ASSERT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);
  }

  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  resampling.Run(ctx, out_tlv, in_tlv, make_span(params));
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
  Kernel kernel;
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
  OutListGPU<uint8_t, 3> in_tlv = input.gpu();
  for (int i = 0; i < N; i++) {
    copy(in_tlv[i], view_as_tensor<uint8_t, 3>(cv_img[i]));
  }

  auto req = kernel.Setup(ctx, in_tlv, make_span(params));
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  scratch_alloc.Reserve(req.scratch_sizes);

  output.reshape(req.output_shapes[0].to_static<3>());
  OutListGPU<uint8_t, 3> out_tlv = output.gpu();
  for (int i = 0; i < out_tlv.num_samples(); i++) {
    auto tv = out_tlv[i];
    CUDA_CALL(cudaMemset(tv.data, 0, tv.num_elements()*sizeof(*tv.data)));
  }

  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      in_tlv.shape.tensor_shape_span(i)[2]
    };
    ASSERT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);
  }

  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  kernel.Run(ctx, out_tlv, in_tlv, make_span(params));
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
