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
#include <opencv2/imgcodecs.hpp>
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/imgproc/resample/separable_cpu.h"
#include "dali/kernels/scratch.h"

namespace dali {
namespace kernels {

template <typename ElementType>
cv::Mat MatWithShape(TensorShape<3> shape) {
  using U = typename std::remove_const<ElementType>::type;
  int depth = cv::DataDepth<U>::value;
  return cv::Mat(shape[0], shape[1], CV_MAKETYPE(depth, shape[2]));
}


constexpr FilterDesc tri(float radius = 0) {
  return { ResamplingFilterType::Triangular, radius };
}

constexpr FilterDesc lin() {
  return { ResamplingFilterType::Linear, 0 };
}

constexpr FilterDesc lanczos() {
  return { ResamplingFilterType::Lanczos3, 0 };
}

constexpr FilterDesc gauss(float radius) {
  return { ResamplingFilterType::Gaussian, radius };
}

struct ResamplingTestEntry {
  ResamplingTestEntry(std::string input,
                      std::string reference,
                      std::array<int, 2> sizeWH,
                      FilterDesc filter,
                      double epsilon = 1)
    : ResamplingTestEntry(std::move(input)
    , std::move(reference), sizeWH, filter, filter, epsilon) {}

  ResamplingTestEntry(std::string input,
                      std::string reference,
                      std::array<int, 2> sizeWH,
                      FilterDesc fx,
                      FilterDesc fy,
                      double epsilon = 1)
    : input(std::move(input)), reference(std::move(reference)), epsilon(epsilon) {
    params[0].output_size = sizeWH[1];
    params[1].output_size = sizeWH[0];
    params[0].mag_filter = params[0].min_filter = fy;
    params[1].mag_filter = params[1].min_filter = fx;
  }

  std::string input, reference;
  ResamplingParams2D params;
  double epsilon = 1;
};

TEST(SeparableResampleCPU, NN) {
  auto img = testing::data::image("imgproc_test/blobs.png");
  auto ref = testing::data::image("imgproc_test/dots.png");
  auto in_tensor = view_as_tensor<const uint8_t, 3>(img);
  auto ref_tensor = view_as_tensor<const uint8_t, 3>(ref);

  SeparableResampleCPU<uint8_t, uint8_t> resample;
  KernelContext context;
  ScratchpadAllocator scratch_alloc;

  FilterDesc filter(ResamplingFilterType::Nearest);

  ResamplingParams2D params;
  params[0].min_filter = params[0].mag_filter = filter;
  params[1].min_filter = params[1].mag_filter = filter;
  params[0].output_size = 4;
  params[1].output_size = 4;

  auto req = resample.Setup(context, in_tensor, params);
  EXPECT_TRUE(resample.setup.desc.IsPureNN());
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  context.scratchpad = &scratchpad;

  auto out_mat = MatWithShape<uint8_t>(req.output_shapes[0].tensor_shape<3>(0));
  auto out_tensor = view_as_tensor<uint8_t, 3>(out_mat);

  resample.Run(context, out_tensor, in_tensor, params);

  Check(out_tensor, ref_tensor);
  // cv::imwrite("separable_NN_4x4.png", out_mat);
}


TEST(SeparableResampleCPU, Triangular) {
  auto img = testing::data::image("imgproc_test/alley.png");
  auto ref = testing::data::image("imgproc_test/ref_out/alley_tri_300x300.png");
  auto in_tensor = view_as_tensor<const uint8_t, 3>(img);
  auto ref_tensor = view_as_tensor<const uint8_t, 3>(ref);

  SeparableResampleCPU<uint8_t, uint8_t> resample;
  KernelContext context;
  ScratchpadAllocator scratch_alloc;

  FilterDesc filter(ResamplingFilterType::Triangular);

  ResamplingParams2D params;
  params[0].min_filter = params[0].mag_filter = filter;
  params[1].min_filter = params[1].mag_filter = filter;
  params[0].output_size = 300;
  params[1].output_size = 300;

  auto req = resample.Setup(context, in_tensor, params);
  EXPECT_FALSE(resample.setup.desc.IsPureNN());
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  context.scratchpad = &scratchpad;

  auto out_mat = MatWithShape<uint8_t>(req.output_shapes[0].tensor_shape<3>(0));
  auto out_tensor = view_as_tensor<uint8_t, 3>(out_mat);

  resample.Run(context, out_tensor, in_tensor, params);

  Check(out_tensor, ref_tensor, EqualEps(1));
}

TEST(SeparableResampleCPU, Linear) {
  auto img = testing::data::image("imgproc_test/dots.png");
  auto ref = testing::data::image("imgproc_test/blobs.png");
  auto in_tensor = view_as_tensor<const uint8_t, 3>(img);
  auto ref_tensor = view_as_tensor<const uint8_t, 3>(ref);

  SeparableResampleCPU<uint8_t, uint8_t> resample;
  KernelContext context;
  ScratchpadAllocator scratch_alloc;

  FilterDesc filter(ResamplingFilterType::Linear);

  ResamplingParams2D params;
  params[0].min_filter = params[0].mag_filter = filter;
  params[1].min_filter = params[1].mag_filter = filter;
  params[0].output_size = 300;
  params[1].output_size = 300;

  auto req = resample.Setup(context, in_tensor, params);
  EXPECT_FALSE(resample.setup.desc.IsPureNN());
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  context.scratchpad = &scratchpad;

  auto out_mat = MatWithShape<uint8_t>(req.output_shapes[0].tensor_shape<3>(0));
  auto out_tensor = view_as_tensor<uint8_t, 3>(out_mat);

  resample.Run(context, out_tensor, in_tensor, params);

  Check(out_tensor, ref_tensor, EqualEps(1));
}

TEST(SeparableResampleCPU, Lanczos3) {
  auto img = testing::data::image("imgproc_test/score.png");
  auto ref = testing::data::image("imgproc_test/ref_out/score_lanczos3.png");
  auto in_tensor = view_as_tensor<const uint8_t, 3>(img);
  auto ref_tensor = view_as_tensor<const uint8_t, 3>(ref);

  SeparableResampleCPU<uint8_t, uint8_t> resample;
  KernelContext context;
  ScratchpadAllocator scratch_alloc;

  FilterDesc filter(ResamplingFilterType::Lanczos3);

  ResamplingParams2D params;
  params[0].min_filter = params[0].mag_filter = filter;
  params[1].min_filter = params[1].mag_filter = filter;
  params[0].output_size = img.rows * 5;
  params[1].output_size = img.cols * 5;

  auto req = resample.Setup(context, in_tensor, params);
  EXPECT_FALSE(resample.setup.desc.IsPureNN());
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  context.scratchpad = &scratchpad;

  auto out_mat = MatWithShape<uint8_t>(req.output_shapes[0].tensor_shape<3>(0));
  auto out_tensor = view_as_tensor<uint8_t, 3>(out_mat);

  resample.Run(context, out_tensor, in_tensor, params);

  Check(out_tensor, ref_tensor, EqualEps(1));
}


}  // namespace kernels
}  // namespace dali
