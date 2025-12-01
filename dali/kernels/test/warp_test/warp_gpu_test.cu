// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/imgproc/warp_gpu.h"
#include "dali/kernels/imgproc/warp/affine.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/dump_diff.h"
#include "dali/test/mat2tensor.h"
#include "dali/test/test_tensors.h"
#include "dali/core/mm/memory.h"
#include "dali/test/dali_test_config.h"
#include "dali/core/geom/transform.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {
namespace kernels {


class WarpPrivateTest {
 public:
  template <typename Mapping, int ndim, typename Out, typename In, typename Border>
  static kernels::warp::WarpSetup<ndim, Out, In> &
  GetSetup(kernels::WarpGPU<Mapping, ndim, Out, In, Border> &kernel) {
    return kernel.setup;
  }
};

TEST(WarpGPU, check_kernel) {
  check_kernel<WarpGPU<AffineMapping2D, 2, float, uint8_t, float>>();
  SUCCEED();
}

void WarpGPU_Affine_Transpose(bool force_variable) {
  AffineMapping2D mapping_cpu = mat2x3{{
    { 0, 1, 0 },
    { 1, 0, 0 }
  }};

  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/alley.png");
  auto cpu_img = view_as_tensor<uint8_t>(cv_img);
  auto gpu_img = copy<mm::memory_kind::device>(cpu_img);
  auto img_tensor = gpu_img.first;

  TensorListView<StorageGPU, uint8_t, 3> in_list;
  in_list.resize(1, 3);
  in_list.shape.set_tensor_shape(0, img_tensor.shape);
  in_list.data[0] = img_tensor.data;

  WarpGPU<AffineMapping2D, 2, uint8_t, uint8_t, BorderClamp> warp;


  auto mapping_gpu = mm::alloc_raw_unique<AffineMapping2D, mm::memory_kind::device>(1);
  TensorShape<2> out_shape = { img_tensor.shape[1], img_tensor.shape[0] };
  KernelContext ctx = {};
  ctx.gpu.stream = 0;

  auto out_shapes_hw = make_span<1>(&out_shape);
  auto mappings = make_tensor_gpu<1>(mapping_gpu.get(), { 1 });
  copy(mappings, make_tensor_cpu<1>(&mapping_cpu, { 1 }));

  auto interp = DALI_INTERP_NN;
  KernelRequirements req;

  if (force_variable) {
    auto &setup = WarpPrivateTest::GetSetup(warp);
    setup.SetBlockDim(dim3(32, 8, 1));
    auto out_shapes = setup.GetOutputShape(in_list.shape, out_shapes_hw);
    req = setup.Setup(out_shapes, true);
  } else {
    req = warp.Setup(ctx, in_list, mappings, out_shapes_hw, {&interp, 1});
  }

  TestTensorList<uint8_t, 3> out;
  out.reshape(req.output_shapes[0].to_static<3>());

  DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
  ctx.scratchpad = &dyn_scratchpad;

  warp.Run(ctx, out.gpu(0), in_list, mappings, out_shapes_hw, {&interp, 1});

  auto cpu_out = out.cpu(0)[0];
  CUDA_CALL(cudaDeviceSynchronize());
  ASSERT_EQ(cpu_out.shape[0], img_tensor.shape[1]);
  ASSERT_EQ(cpu_out.shape[1], img_tensor.shape[0]);
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

TEST(WarpGPU, Affine_Transpose_ForceVariable) {
  WarpGPU_Affine_Transpose(true);
}

TEST(WarpGPU, Affine_Transpose_Single) {
  WarpGPU_Affine_Transpose(false);
}

/**
 * @brief Apply correction of pixel centers and convert the mapping to
 *        OpenCV matrix type.
 */
inline cv::Matx<float, 2, 3> AffineToCV(const AffineMapping2D &mapping) {
  vec2 translation = mapping({0.5f, 0.5f}) - vec2(0.5f, 0.5f);
  mat2x3 tmp = mapping.transform;
  tmp.set_col(2, translation);

  cv::Matx<float, 2, 3> cv_transform;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
      cv_transform(i, j) = tmp(i, j);
  return cv_transform;
}

TEST(WarpGPU, Affine_RotateScale_Single) {
  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/dots.png");
  auto cpu_img = view_as_tensor<uint8_t>(cv_img);
  auto gpu_img = copy<mm::memory_kind::device>(cpu_img);
  auto img_tensor = gpu_img.first;

  vec2 center(cv_img.cols * 0.5f, cv_img.rows * 0.5f);

  int scale = 10;
  auto tr = translation(center) * rotation2D(-M_PI/4) *
            translation(-center) * scaling(vec2(1.0f/scale, 1.0f/scale));
  AffineMapping2D mapping_cpu = sub<2, 3>(tr, 0, 0);

  TensorListView<StorageGPU, uint8_t, 3> in_list;
  in_list.resize(1, 3);
  in_list.shape.set_tensor_shape(0, img_tensor.shape);
  in_list.data[0] = img_tensor.data;

  WarpGPU<AffineMapping2D, 2, uint8_t, uint8_t, uint8_t> warp;

  auto mapping_gpu = mm::alloc_raw_unique<AffineMapping2D, mm::memory_kind::device>(1);
  TensorShape<2> out_shape = { img_tensor.shape[0] * scale, img_tensor.shape[1] * scale };
  KernelContext ctx = {};
  ctx.gpu.stream = 0;
  auto out_shapes_hw = make_span<1>(&out_shape);
  auto mappings = make_tensor_gpu<1>(mapping_gpu.get(), { 1 });
  copy(mappings, make_tensor_cpu<1>(&mapping_cpu, { 1 }));

  auto interp = DALI_INTERP_LINEAR;
  auto &setup = WarpPrivateTest::GetSetup(warp);
  auto out_shapes = setup.GetOutputShape(in_list.shape, out_shapes_hw);
  setup.SetBlockDim(dim3(32, 24, 1));  // force non-square block
  KernelRequirements req = setup.Setup(out_shapes, true);
  TestTensorList<uint8_t, 3> out;
  out.reshape(req.output_shapes[0].to_static<3>());

  DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
  ctx.scratchpad = &dyn_scratchpad;

  warp.Run(ctx, out.gpu(0), in_list, mappings, out_shapes_hw, {&interp, 1}, 255);

  auto cpu_out = out.cpu(0)[0];
  CUDA_CALL(cudaDeviceSynchronize());
  ASSERT_EQ(cpu_out.shape[0], out_shapes_hw[0][0]);
  ASSERT_EQ(cpu_out.shape[1], out_shapes_hw[0][1]);
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
    testing::DumpDiff("WarpAffine_RotateScale", cv_out, cv_ref);
}


TEST(WarpGPU, Affine_RotateScale_Uniform) {
  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/dots.png");
  auto cpu_img = view_as_tensor<uint8_t>(cv_img);
  auto gpu_img = copy<mm::memory_kind::device>(cpu_img);
  auto img_tensor = gpu_img.first;

  vec2 center(cv_img.cols * 0.5f, cv_img.rows * 0.5f);

  const int samples = 10;
  std::vector<AffineMapping2D> mapping_cpu(samples);
  int scale = 10;

  TensorListView<StorageGPU, uint8_t, 3> in_list;
  in_list.resize(samples, 3);
  for (int i = 0; i < samples; i++) {
    in_list.shape.set_tensor_shape(i, img_tensor.shape);
    in_list.data[i] = img_tensor.data;

    auto tr = translation(center) * rotation2D(-2*M_PI * i / samples) *
              translation(-center) * scaling(vec2(1.0f/scale, 1.0f/scale));
    mapping_cpu[i] = sub<2, 3>(tr, 0, 0);
  }

  WarpGPU<AffineMapping2D, 2, uint8_t, uint8_t, uint8_t> warp;

  auto mapping_gpu = mm::alloc_raw_unique<AffineMapping2D, mm::memory_kind::device>(samples);
  TensorShape<2> out_shape = { img_tensor.shape[0] * scale, img_tensor.shape[1] * scale };
  KernelContext ctx = {};
  ctx.gpu.stream = 0;
  std::vector<TensorShape<2>> out_shapes_hw(samples);
  for (int i = 0; i < samples; i++)
    out_shapes_hw[i] = out_shape;
  auto mappings = make_tensor_gpu<1>(mapping_gpu.get(), { samples });
  copy(mappings, make_tensor_cpu<1>(mapping_cpu.data(), { samples }));

  auto interp = DALI_INTERP_LINEAR;
  KernelRequirements req = warp.Setup(
    ctx, in_list, mappings, make_span(out_shapes_hw), {&interp, 1}, 255);

  TestTensorList<uint8_t, 3> out;
  out.reshape(req.output_shapes[0].to_static<3>());

  DynamicScratchpad dyn_scratchpad(AccessOrder(ctx.gpu.stream));
  ctx.scratchpad = &dyn_scratchpad;

  warp.Run(ctx, out.gpu(0), in_list, mappings, make_span(out_shapes_hw), {&interp, 1}, 255);
  CUDA_CALL(cudaDeviceSynchronize());

  for (int i = 0; i < samples; i++) {
    auto cpu_out = out.cpu(0)[i];
    ASSERT_EQ(cpu_out.shape[0], out_shapes_hw[i][0]);
    ASSERT_EQ(cpu_out.shape[1], out_shapes_hw[i][1]);
    ASSERT_EQ(cpu_out.shape[2], 3);

    cv::Mat cv_out(cpu_out.shape[0], cpu_out.shape[1], CV_8UC3, cpu_out.data);

    cv::Matx<float, 2, 3> cv_transform = AffineToCV(mapping_cpu[i]);

    cv::Mat cv_ref;
    cv::warpAffine(cv_img, cv_ref,
                  cv_transform, cv::Size(out_shape[1], out_shape[0]),
                  cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,
                  cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255, 255));
    auto ref_img = view_as_tensor<uint8_t>(cv_ref);
    Check(cpu_out, ref_img, EqualEps(8));
    if (HasFailure()) {
      auto name = "Warp_Affine_RotateScale_" + std::to_string(i);
      testing::DumpDiff(name, cv_out, cv_ref);
    }
  }
}

}  // namespace kernels
}  // namespace dali
