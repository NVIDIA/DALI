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
#include "dali/kernels/test/resampling_test/resampling_test_params.h"
#include "dali/kernels/imgproc/resample/separable_cpu.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/kernels/scratch.h"

namespace dali {
namespace kernels {
namespace resample_test {

template <typename ElementType>
cv::Mat MatWithShape(TensorShape<3> shape) {
  using U = typename std::remove_const<ElementType>::type;
  int depth = cv::DataDepth<U>::value;
  return cv::Mat(shape[0], shape[1], CV_MAKETYPE(depth, shape[2]));
}

class ResamplingTestCPU : public ::testing::Test,
                          public ::testing::WithParamInterface<ResamplingTestEntry> {
};

TEST_P(ResamplingTestCPU, Impl) {
  const ResamplingTestEntry &param = GetParam();
  auto img = testing::data::image(param.input.c_str());
  auto ref = testing::data::image(param.reference.c_str());
  auto in_tensor = view_as_tensor<const uint8_t, 3>(img);
  auto ref_tensor = view_as_tensor<const uint8_t, 3>(ref);

  SeparableResampleCPU<uint8_t, uint8_t> resample;
  KernelContext context;
  ScratchpadAllocator scratch_alloc;

  FilterDesc filter(ResamplingFilterType::Nearest);

  auto req = resample.Setup(context, in_tensor, param.params);
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  context.scratchpad = &scratchpad;

  auto out_mat = MatWithShape<uint8_t>(req.output_shapes[0].tensor_shape<3>(0));
  auto out_tensor = view_as_tensor<uint8_t, 3>(out_mat);

  resample.Run(context, out_tensor, in_tensor, param.params);

  Check(out_tensor, ref_tensor, EqualEps(param.epsilon));
}

TEST_P(ResamplingTestCPU, KernelAPI) {
  const ResamplingTestEntry &param = GetParam();
  auto img = testing::data::image(param.input.c_str());
  auto ref = testing::data::image(param.reference.c_str());
  auto in_tensor = view_as_tensor<const uint8_t, 3>(img);
  auto ref_tensor = view_as_tensor<const uint8_t, 3>(ref);

  using Kernel = ResampleCPU<uint8_t, uint8_t>;
  KernelContext context;
  ScratchpadAllocator scratch_alloc;

  FilterDesc filter(ResamplingFilterType::Nearest);

  auto req = Kernel::GetRequirements(context, in_tensor, param.params);
  ASSERT_NE(any_cast<Kernel::Impl>(&context.kernel_data), nullptr);
  scratch_alloc.Reserve(req.scratch_sizes);
  auto scratchpad = scratch_alloc.GetScratchpad();
  context.scratchpad = &scratchpad;

  auto out_mat = MatWithShape<uint8_t>(req.output_shapes[0].tensor_shape<3>(0));
  auto out_tensor = view_as_tensor<uint8_t, 3>(out_mat);

  Kernel::Run(context, out_tensor, in_tensor, param.params);

  Check(out_tensor, ref_tensor, EqualEps(param.epsilon));

  if (HasFailure()) {
    cv::Mat mat(out_tensor.shape[0], out_tensor.shape[1], CV_8UC3, out_tensor.data);
    auto name = param.reference;
    name = name.substr(name.rfind('/')+1);
    auto out_name = name.substr(0, name.find('.')) + "_out.png";
    cv::imwrite(out_name, mat);
  }
}

static std::vector<ResamplingTestEntry> ResampleTests = {
  {
    "imgproc_test/blobs.png", "imgproc_test/dots.png",
    { 4, 4 }, nearest(), 0
  },
  {
    "imgproc_test/dots.png", "imgproc_test/blobs.png",
    { 300, 300 }, lin(), 0
  },
  {
    "imgproc_test/alley.png", "imgproc_test/ref_out/alley_tri_300x300.png",
    { 300, 300 }, tri(), 1
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

static std::vector<ResamplingTestEntry> CropResampleTests = {
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



INSTANTIATE_TEST_SUITE_P(Basic, ResamplingTestCPU, ::testing::ValuesIn(ResampleTests));
INSTANTIATE_TEST_SUITE_P(Crop , ResamplingTestCPU, ::testing::ValuesIn(CropResampleTests));


}  // namespace resample_test
}  // namespace kernels
}  // namespace dali
