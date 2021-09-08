// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "dali/kernels/imgproc/warp_cpu.h"
#include "dali/kernels/imgproc/warp/affine.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/dump_diff.h"
#include "dali/test/mat2tensor.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/scratch.h"
#include "dali/test/dali_test_config.h"
#include "dali/core/geom/transform.h"
#include "dali/kernels/test/warp_test/warp_test_helper.h"

namespace dali {
namespace kernels {

TEST(WarpCPU, check_kernel) {
  check_kernel<WarpCPU<AffineMapping2D, 2, float, uint8_t, float>>();
  SUCCEED();
}

TEST(WarpCPU, Affine_Transpose_Single) {
  AffineMapping2D mapping_cpu = mat2x3{{
    { 0, 1, 0 },
    { 1, 0, 0 }
  }};

  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/alley.png");
  auto cpu_img = view_as_tensor<uint8_t>(cv_img);

  WarpCPU<AffineMapping2D, 2, uint8_t, uint8_t, BorderClamp> warp;

  ScratchpadAllocator scratch_alloc;

  TensorShape<2> out_shape = { cpu_img.shape[1], cpu_img.shape[0] };
  KernelContext ctx = {};

  auto interp = DALI_INTERP_NN;
  KernelRequirements req;

  req = warp.Setup(ctx, cpu_img, mapping_cpu, out_shape, interp);

  scratch_alloc.Reserve(req.scratch_sizes);
  TestTensorList<uint8_t, 3> out;
  out.reshape(req.output_shapes[0].to_static<3>());
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  warp.Run(ctx, out.cpu(0)[0], cpu_img, mapping_cpu, out_shape, interp);

  auto cpu_out = out.cpu(0)[0];
  ASSERT_EQ(cpu_out.shape[0], cpu_img.shape[1]);
  ASSERT_EQ(cpu_out.shape[1], cpu_img.shape[0]);
  ASSERT_EQ(cpu_out.shape[2], 3);

  int errors = 0;
  int printed = 0;
  for (int y = 0; y < cpu_out.shape[0]; y++) {
    for (int x = 0; x < cpu_out.shape[1]; x++) {
      for (int c = 0; c < 3; c++) {
        if (*cpu_out(y, x, c) != *cpu_img(x, y, c)) {
          if (errors++ < 100) {
            printed++;
            EXPECT_EQ(*cpu_out(y, x, c), *cpu_img(x, y, c))
              << "@ x = " << x << " y = " << y << " c = " << c;
          }
        }
      }
    }
  }
  if (printed != errors) {
    FAIL() << (errors - printed) << " more erors.";
  }
}

TEST(WarpCPU, Affine_RotateScale) {
  WarpCPU<AffineMapping2D, 2, uint8_t, uint8_t, uint8_t> warp;
  ScratchpadAllocator scratch_alloc;

  static const std::string names[] = { "dots", "alley" };
  static const float scales[] = { 10.0f, 0.5f };
  for (int img_idx = 0; img_idx < 2; img_idx++) {
    const auto &name = names[img_idx];
    float scale = scales[img_idx];

    cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/" + name + ".png");
    auto cpu_img = view_as_tensor<uint8_t>(cv_img);

    vec2 center(cv_img.cols * 0.5f, cv_img.rows * 0.5f);

    auto tr = translation(center) * rotation2D(-M_PI/4) *
              translation(-center) * scaling(vec2(1.0f/scale, 1.0f/scale));
    AffineMapping2D mapping_cpu = sub<2, 3>(tr, 0, 0);

    int out_h = cpu_img.shape[0] * scale;
    int out_w = cpu_img.shape[1] * scale;
    TensorShape<2> out_shape = { out_h, out_w };
    KernelContext ctx = {};

    auto interp = DALI_INTERP_LINEAR;
    KernelRequirements req;

    req = warp.Setup(ctx, cpu_img, mapping_cpu, out_shape, interp, 255);

    scratch_alloc.Reserve(req.scratch_sizes);
    TestTensorList<uint8_t, 3> out;
    out.reshape(req.output_shapes[0].to_static<3>());
    auto scratchpad = scratch_alloc.GetScratchpad();
    ctx.scratchpad = &scratchpad;
    warp.Run(ctx, out.cpu(0)[0], cpu_img, mapping_cpu, out_shape, interp, 255);

    auto cpu_out = out.cpu(0)[0];
    ASSERT_EQ(cpu_out.shape[0], out_shape[0]);
    ASSERT_EQ(cpu_out.shape[1], out_shape[1]);
    ASSERT_EQ(cpu_out.shape[2], 3);

    cv::Mat cv_out(cpu_out.shape[0], cpu_out.shape[1], CV_8UC3, cpu_out.data);

    cv::Matx<float, 2, 3> cv_transform = AffineToCV(mapping_cpu);

    cv::Mat cv_ref;
    cv::warpAffine(cv_img, cv_ref,
                  cv_transform, cv::Size(out_shape[1], out_shape[0]),
                  cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,
                  cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255, 255));
    auto ref_img = view_as_tensor<uint8_t>(cv_ref);
    Check(cpu_out, ref_img, EqualEps(8));
    if (HasFailure())
      testing::DumpDiff("WarpAffine_RotateScale_" + name, cv_out, cv_ref);
  }
}

}  // namespace kernels
}  // namespace dali
