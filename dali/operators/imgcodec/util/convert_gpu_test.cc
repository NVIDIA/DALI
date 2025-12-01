// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/imgcodec/util/convert_gpu.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {

template <class Input, class Output>
struct ConversionTestType {
  using In = Input;
  using Out = Output;
};

using TensorTestData = std::vector<std::vector<std::vector<float>>>;

TensorShape<> get_data_shape(const TensorTestData &data) {
  return {
      static_cast<int>(data.size()),
      static_cast<int>(data[0].size()),
      static_cast<int>(data[0][0].size()),
  };
}

template <class T>
void init_test_tensor_list(kernels::TestTensorList<T> &list, const TensorTestData &data) {
  auto shape = get_data_shape(data);
  list.invalidate_cpu();
  list.invalidate_gpu();
  list.reshape({{shape}});
  auto tv = list.cpu()[0];
  for (int i = 0; i < shape[0]; i++)
    for (int j = 0; j < shape[1]; j++)
      for (int k = 0; k < shape[2]; k++)
        *tv(TensorShape<>{i, j, k}) = ConvertSatNorm<T>(data[i][j][k]);
}

TensorTestData empty_data(TensorShape<> shape) {
  return std::vector(shape[0], std::vector(shape[1], std::vector(shape[2], 0.0f)));
}

template <class T>
SampleView<GPUBackend> get_gpu_sample_view(kernels::TestTensorList<T> &list) {
  auto tv = list.gpu()[0];
  return SampleView<GPUBackend>(tv.data, tv.shape, type2id<T>::value);
}

// Helper values, to make testing data more readable
const std::vector<float> pixelA = {0.00f, 0.01f, 0.02f};
const std::vector<float> pixelB = {0.10f, 0.11f, 0.12f};
const std::vector<float> pixelC = {0.20f, 0.21f, 0.22f};
const std::vector<float> pixelD = {0.30f, 0.31f, 0.32f};
const std::vector<float> pixelE = {0.40f, 0.41f, 0.42f};
const std::vector<float> pixelF = {0.50f, 0.51f, 0.52f};

}  // namespace

template <typename ConversionType>
class ConvertGPUTest : public ::testing::Test {
 public:
  using Input = typename ConversionType::In;
  using Output = typename ConversionType::Out;

  void SetReference(const TensorTestData &data) {
    init_test_tensor_list(reference_list_, data);
    init_test_tensor_list(output_list_, empty_data(get_data_shape(data)));
  }

  void SetInput(const TensorTestData &data) {
    init_test_tensor_list(input_list_, data);
  }

  void CheckConvert(TensorLayout out_layout, DALIImageType out_format, TensorLayout in_layout,
                    DALIImageType in_format, const ROI &roi = {},
                    nvimgcodecOrientation_t orientation = {}, float multiplier = 1.0f) {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    auto out = get_gpu_sample_view(output_list_);
    auto in = get_gpu_sample_view(input_list_);
    auto stream = CUDAStreamPool::instance().Get(device_id);
    ConvertGPU(out, out_layout, out_format, in, in_layout, in_format, stream, roi, orientation,
               multiplier);
    output_list_.invalidate_cpu();
    auto tv = output_list_.cpu(stream)[0];  // here d2h copy happens
    CUDA_CALL(cudaStreamSynchronize(stream));
    Check(output_list_.cpu()[0], reference_list_.cpu()[0], EqualConvertNorm(eps_));
  }

 private:
  kernels::TestTensorList<Input> input_list_;
  kernels::TestTensorList<Output> output_list_;
  kernels::TestTensorList<Output> reference_list_;
  const float eps_ = 0.01f;
};

using ConversionTypes =
    ::testing::Types<ConversionTestType<uint8_t, int16_t>, ConversionTestType<float, uint8_t>,
                     ConversionTestType<uint16_t, float>>;

TYPED_TEST_SUITE(ConvertGPUTest, ConversionTypes);

TYPED_TEST(ConvertGPUTest, Multiply) {
  this->SetInput({
      {{0.00f, 0.01f, 0.02f}, {0.03f, 0.04f, 0.05f}},
      {{0.10f, 0.11f, 0.12f}, {0.13f, 0.14f, 0.15f}},
  });

  this->SetReference({
      {{0.00f, 0.02f, 0.04f}, {0.06f, 0.08f, 0.10f}},
      {{0.20f, 0.22f, 0.24f}, {0.26f, 0.28f, 0.30f}},
  });

  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, {}, 2.0f);
}

TYPED_TEST(ConvertGPUTest, PlanarToInterleaved) {
  this->SetInput({
      {
          {0.00f, 0.01f, 0.02f, 0.03f},
          {0.10f, 0.11f, 0.12f, 0.13f},
      },
      {
          {0.20f, 0.21f, 0.22f, 0.23f},
          {0.30f, 0.31f, 0.32f, 0.33f},
      },
      {
          {0.40f, 0.41f, 0.42f, 0.43f},
          {0.50f, 0.51f, 0.52f, 0.53f},
      },
  });

  this->SetReference({
      {{0.00f, 0.20f, 0.40f}, {0.01f, 0.21f, 0.41f}, {0.02f, 0.22f, 0.42f}, {0.03f, 0.23f, 0.43f}},
      {{0.10f, 0.30f, 0.50f}, {0.11f, 0.31f, 0.51f}, {0.12f, 0.32f, 0.52f}, {0.13f, 0.33f, 0.53f}},
  });

  this->CheckConvert("HWC", DALI_RGB, "CHW", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, InterleavedToPlanar) {
  this->SetInput({
      {{0.00f, 0.20f, 0.40f}, {0.01f, 0.21f, 0.41f}, {0.02f, 0.22f, 0.42f}, {0.03f, 0.23f, 0.43f}},
      {{0.10f, 0.30f, 0.50f}, {0.11f, 0.31f, 0.51f}, {0.12f, 0.32f, 0.52f}, {0.13f, 0.33f, 0.53f}},
  });

  this->SetReference({{
                          {0.00f, 0.01f, 0.02f, 0.03f},
                          {0.10f, 0.11f, 0.12f, 0.13f},
                      },
                      {
                          {0.20f, 0.21f, 0.22f, 0.23f},
                          {0.30f, 0.31f, 0.32f, 0.33f},
                      },
                      {
                          {0.40f, 0.41f, 0.42f, 0.43f},
                          {0.50f, 0.51f, 0.52f, 0.53f},
                      }});

  this->CheckConvert("CHW", DALI_RGB, "HWC", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, TransposeWithRoi2D) {
  this->SetInput({
      {
          {0.00f, 0.01f, 0.02f, 0.03f},
          {0.10f, 0.11f, 0.12f, 0.13f},
      },
      {
          {0.20f, 0.21f, 0.22f, 0.23f},
          {0.30f, 0.31f, 0.32f, 0.33f},
      },
      {
          {0.40f, 0.41f, 0.42f, 0.43f},
          {0.50f, 0.51f, 0.52f, 0.53f},
      },
  });

  this->SetReference({
      {{0.12f, 0.32f, 0.52f}, {0.13f, 0.33f, 0.53f}},
  });

  this->CheckConvert("HWC", DALI_RGB, "CHW", DALI_RGB, {{1, 2}, {2, 4}});
}

TYPED_TEST(ConvertGPUTest, TransposeWithRoi3D) {
  this->SetInput({
      {
          {0.00f, 0.01f, 0.02f, 0.03f},
          {0.10f, 0.11f, 0.12f, 0.13f},
      },
      {
          {0.20f, 0.21f, 0.22f, 0.23f},
          {0.30f, 0.31f, 0.32f, 0.33f},
      },
      {
          {0.40f, 0.41f, 0.42f, 0.43f},
          {0.50f, 0.51f, 0.52f, 0.53f},
      },
  });

  this->SetReference({
      {{0.12f, 0.32f, 0.52f}, {0.13f, 0.33f, 0.53f}},
  });

  this->CheckConvert("HWC", DALI_RGB, "CHW", DALI_RGB, {{1, 2, 0}, {2, 4, 3}});
}

TYPED_TEST(ConvertGPUTest, RGBToYCbCr) {
  this->SetInput({
      {
          {0.1f, 0.2f, 0.3f},
      },
  });

  this->SetReference({
      {
          {0.218f, 0.558f, 0.449f},
      },
  });

  this->CheckConvert("HWC", DALI_YCbCr, "HWC", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, RGBToBGR) {
  this->SetInput({
      {
          {0.1f, 0.2f, 0.3f},
      },
  });

  this->SetReference({
      {
          {0.3f, 0.2f, 0.1f},
      },
  });

  this->CheckConvert("HWC", DALI_BGR, "HWC", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, RGBToGray) {
  this->SetInput({
      {
          {0.1f, 0.2f, 0.3f},
      },
  });

  this->SetReference({
      {
          {0.181f},
      },
  });

  this->CheckConvert("HWC", DALI_GRAY, "HWC", DALI_RGB);
}

TYPED_TEST(ConvertGPUTest, Rotation90) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelC, pixelF},
      {pixelB, pixelE},
      {pixelA, pixelD},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      90,
                                      false,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, Rotation90FlipX) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelF, pixelC},
      {pixelE, pixelB},
      {pixelD, pixelA},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      90,
                                      true,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, Rotation180) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelF, pixelE, pixelD},
      {pixelC, pixelB, pixelA},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      180,
                                      false,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, Rotation270) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelD, pixelA},
      {pixelE, pixelB},
      {pixelF, pixelC},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      270,
                                      false,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, FlipX) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelC, pixelB, pixelA},
      {pixelF, pixelE, pixelD},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      0,
                                      true,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, FlipY) {
  this->SetInput({
      {pixelA, pixelB, pixelC},
      {pixelD, pixelE, pixelF},
  });

  this->SetReference({
      {pixelD, pixelE, pixelF},
      {pixelA, pixelB, pixelC},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      0,
                                      false,
                                      true};
  this->CheckConvert("HWC", DALI_RGB, "HWC", DALI_RGB, {}, orientation);
}

TYPED_TEST(ConvertGPUTest, TransposeAndRotate_PlanarToInterleaved) {
  this->SetInput({
      {
          {0.00f, 0.01f, 0.02f, 0.03f},
          {0.10f, 0.11f, 0.12f, 0.13f},
      },
      {
          {0.20f, 0.21f, 0.22f, 0.23f},
          {0.30f, 0.31f, 0.32f, 0.33f},
      },
      {
          {0.40f, 0.41f, 0.42f, 0.43f},
          {0.50f, 0.51f, 0.52f, 0.53f},
      },
  });

  this->SetReference({
      {{0.03f, 0.23f, 0.43f}, {0.13f, 0.33f, 0.53f}},
      {{0.02f, 0.22f, 0.42f}, {0.12f, 0.32f, 0.52f}},
      {{0.01f, 0.21f, 0.41f}, {0.11f, 0.31f, 0.51f}},
      {{0.00f, 0.20f, 0.40f}, {0.10f, 0.30f, 0.50f}},
  });

  nvimgcodecOrientation_t orientation{NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION,
                                      sizeof(nvimgcodecOrientation_t),
                                      nullptr,
                                      90,
                                      false,
                                      false};
  this->CheckConvert("HWC", DALI_RGB, "CHW", DALI_RGB, {}, orientation);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
