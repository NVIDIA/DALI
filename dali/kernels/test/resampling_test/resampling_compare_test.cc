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
#include "dali/kernels/backend_tags.h"
#include "dali/kernels/test/test_tensors.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/kernels/test/resampling_test/resampling_test_params.h"

using std::cout;
using std::endl;

namespace dali {
namespace kernels {
namespace resample_test {

namespace {

ResamplingTestBatch SingleImageBatch = {
  {
    "imgproc_test/alley.png", "",
    { 300, 300 }, tri(), 1
  }
};

ResamplingTestBatch Batch1 = {
  {
    "imgproc_test/alley.png", "",
    { 300, 300 }, tri(), 1
  },
  {
    "imgproc_test/alley.png", "",
    { 1000, 1000 }, lin(), 1
  },
  {
    "imgproc_test/score.png", "",
    { 540, 250 }, lanczos(), 1
  },
  {
    "imgproc_test/score.png", "",
    { 200, 93 }, cubic(), 1
  },
  {
    "imgproc_test/alley.png", "",
    { 681, 960 }, gauss(12), 1
  }
};

ResamplingTestBatch CropBatch = {
  {
    "imgproc_test/dots.png", "",
    { 1.0f, 1.0f, 3.0f, 3.0f }, { 2, 2 }, nearest(), 0
  },
  {
    "imgproc_test/alley.png", "",
    { 100.0f, 300.0f, 200.0f, 400.0f }, { 200, 200 }, cubic(), 1
  },
  {
    "imgproc_test/alley.png", "",
    { 200.0f, 300.0f, 100.0f, 400.0f }, { 200, 200 }, cubic(), 1
  },
  {
    "imgproc_test/alley.png", "",
    { 150.0f, 400.0f, 200.0f, 450.0f }, { 150, 200 }, lin(), 1
  },
  {
    "imgproc_test/alley.png", "",
    { 150.0f, 450.0f, 200.0f, 400.0f }, { 150, 200 }, lin(), 1
  },
};

}  // namespace

class ResamplingCompareTest : public ::testing::Test,
                            public ::testing::WithParamInterface<ResamplingTestBatch> {
};

TEST_P(ResamplingCompareTest, ResamplingKernelAPI) {
  const ResamplingTestBatch &batch = GetParam();

  int N = batch.size();
  std::vector<cv::Mat> cv_img(N);
  std::vector<ResamplingParams2D> params(N);

  for (int i = 0; i < N; i++) {
    cv_img[i] = testing::data::image(batch[i].input.c_str());
    params[i] = batch[i].params;
  }

  ScratchpadAllocator scratch_alloc_gpu, scratch_alloc_cpu;
  KernelContext ctx_gpu, ctx_cpu;
  ctx_gpu.gpu.stream = 0;
  ctx_cpu.gpu.stream = 0;
  using KernelGPU = ResampleGPU<uint8_t, uint8_t>;
  using KernelCPU = ResampleCPU<uint8_t, uint8_t>;
  TestTensorList<uint8_t, 3> input, output_gpu, output_cpu;

  std::vector<TensorShape<3>> shapes;
  for (int i = 0; i < N; i++) {
    shapes.push_back(tensor_shape<3>(cv_img[i]));
  }
  input.reshape(shapes);
  OutListGPU<uint8_t, 3> in_gpu = input.gpu();
  for (int i = 0; i < N; i++) {
    copy(in_gpu[i], view_as_tensor<uint8_t, 3>(cv_img[i]));
  }
  OutListCPU<uint8_t, 3> in_cpu = input.cpu();

  auto req_gpu = KernelGPU::GetRequirements(ctx_gpu, in_gpu, make_span(params));
  std::vector<TensorShape<3>> size_cpu;
  ASSERT_EQ(req_gpu.output_shapes.size(), 1);
  ASSERT_EQ(req_gpu.output_shapes[0].num_samples(), N);

  KernelRequirements req_cpu = {};
  std::vector<TensorShape<>> out_shape_cpu;
  for (int i = 0; i < N; i++) {
    auto req_tmp = KernelCPU::GetRequirements(ctx_cpu, in_cpu[i], params[i]);

    for (size_t j = 0; j < req_cpu.scratch_sizes.size(); j++) {
      req_cpu.scratch_sizes[j] = std::max(req_cpu.scratch_sizes[j], req_tmp.scratch_sizes[j]);
    }
    out_shape_cpu.push_back(req_tmp.output_shapes[0][0]);
  }
  req_cpu.output_shapes.push_back(TensorListShape<>(out_shape_cpu));
  ASSERT_NO_FATAL_FAILURE(CheckEqual(req_cpu.output_shapes[0], req_gpu.output_shapes[0]));

  scratch_alloc_gpu.Reserve(req_gpu.scratch_sizes);
  scratch_alloc_cpu.Reserve(req_cpu.scratch_sizes);

  output_gpu.reshape(req_gpu.output_shapes[0].to_static<3>());
  output_cpu.reshape(req_cpu.output_shapes[0].to_static<3>());

  OutListGPU<uint8_t, 3> out_gpu = output_gpu.gpu();
  cudaMemset(out_gpu.data, 0, out_gpu.num_elements()*sizeof(*out_gpu.data));

  OutListCPU<uint8_t, 3> out_cpu = output_cpu.cpu();
  memset(out_cpu.data, 0, out_cpu.num_elements()*sizeof(*out_cpu.data));

  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      in_gpu.shape.tensor_shape_span(i)[2]
    };
    ASSERT_EQ(req_gpu.output_shapes[0].tensor_shape(i), expected_shape);
  }

  auto scratchpad = scratch_alloc_gpu.GetScratchpad();
  ctx_gpu.scratchpad = &scratchpad;
  KernelGPU::Run(ctx_gpu, out_gpu, in_gpu, make_span(params));

  for (int i = 0; i < N; i++) {
    KernelCPU::GetRequirements(ctx_cpu, in_cpu[i], params[i]);
    auto out_tensor = out_cpu[i];
    auto in_tensor = in_cpu[i];
    auto scratchpad = scratch_alloc_cpu.GetScratchpad();
    ctx_cpu.scratchpad = &scratchpad;
    KernelCPU::Run(ctx_cpu, out_tensor, in_tensor, params[i]);
  }


  for (int i = 0; i < N; i++) {
    auto out_tensor_gpu = output_gpu.cpu()[i];
    auto out_tensor_cpu = output_cpu.cpu()[i];
    ASSERT_NO_FATAL_FAILURE(Check(out_tensor_gpu, out_tensor_cpu, EqualEps(batch[i].epsilon)))
    << [&]() {
      cv::Mat tmp1(out_tensor_cpu.shape[0], out_tensor_cpu.shape[1], CV_8UC3, out_tensor_cpu.data);
      cv::Mat tmp2(out_tensor_gpu.shape[0], out_tensor_gpu.shape[1], CV_8UC3, out_tensor_gpu.data);
      std::string inp_name = batch[i].input;
      int ext = inp_name.rfind('.');
#ifdef WINVER
      constexpr char sep = '\\';
#else
      constexpr char sep = '/';
#endif
      int start = inp_name.rfind(sep) + 1;
      std::string dif_name = inp_name.substr(start, ext - start) + "_dif"+inp_name.substr(ext);
      std::string cpu_name = inp_name.substr(start, ext - start) + "_cpu"+inp_name.substr(ext);
      std::string gpu_name = inp_name.substr(start, ext - start) + "_gpu"+inp_name.substr(ext);
      cv::imwrite(dif_name, 127 + tmp2 - tmp1);
      cv::imwrite(cpu_name, tmp1);
      cv::imwrite(gpu_name, tmp2);
      return "Diff written to " + dif_name;
    }();
  }
}

INSTANTIATE_TEST_SUITE_P(SingleImage, ResamplingCompareTest, ::testing::Values(SingleImageBatch));
INSTANTIATE_TEST_SUITE_P(MultipleImages, ResamplingCompareTest, ::testing::Values(Batch1));

INSTANTIATE_TEST_SUITE_P(Crop, ResamplingCompareTest, ::testing::Values(CropBatch));

}  // namespace resample_test
}  // namespace kernels
}  // namespace dali
