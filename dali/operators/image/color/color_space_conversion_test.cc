// Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "dali/test/dali_test_conversion.h"
#include "dali/core/cuda_error.h"

namespace dali {

template <typename InputImgType>
class ColorSpaceConversionToBGRTest : public GenericConversionTest<InputImgType, BGR> {};

template <typename InputImgType>
class ColorSpaceConversionToRGBTest : public GenericConversionTest<InputImgType, RGB> {};

template <typename InputImgType>
class ColorSpaceConversionToGrayTest : public GenericConversionTest<InputImgType, Gray> {};

template <typename InputImgType>
class ColorSpaceConversionToYCbCrTest : public GenericConversionTest<InputImgType, YCbCr> {};

struct Lab {
  static const DALIImageType type = DALI_LAB;
};

typedef ::testing::Types<RGB, Gray, YCbCr> ConvertibleToBGR;
TYPED_TEST_SUITE(ColorSpaceConversionToBGRTest, ConvertibleToBGR);

TYPED_TEST(ColorSpaceConversionToBGRTest, test) {
  this->RunTest("ColorSpaceConversion");
}

typedef ::testing::Types<BGR, Gray, YCbCr> ConvertibleToRGB;
TYPED_TEST_SUITE(ColorSpaceConversionToRGBTest, ConvertibleToRGB);

TYPED_TEST(ColorSpaceConversionToRGBTest, test) {
  this->RunTest("ColorSpaceConversion");
}

typedef ::testing::Types<RGB, BGR, YCbCr> ConvertibleToGray;
TYPED_TEST_SUITE(ColorSpaceConversionToGrayTest, ConvertibleToGray);

TYPED_TEST(ColorSpaceConversionToGrayTest, test) {
  this->RunTest("ColorSpaceConversion");
}

typedef ::testing::Types<RGB, BGR, Gray> ConvertibleToYCbCr;
TYPED_TEST_SUITE(ColorSpaceConversionToYCbCrTest, ConvertibleToYCbCr);

TYPED_TEST(ColorSpaceConversionToYCbCrTest, test) {
  this->RunTest("ColorSpaceConversion", nullptr, 0, false, 0.002);
}

template <typename InputImgType>
class ColorSpaceConversionToLabTest : public GenericConversionTest<InputImgType, Lab> {};

typedef ::testing::Types<RGB, BGR> ConvertibleToLab;
TYPED_TEST_SUITE(ColorSpaceConversionToLabTest, ConvertibleToLab);

TYPED_TEST(ColorSpaceConversionToLabTest, test) {
  this->RunTest("ColorSpaceConversion", nullptr, 0, false, 0.06);
}

TEST(ColorSpaceConversionTest, ZeroExtentSampleGPU) {
  TensorList<CPUBackend> input;
  input.Resize(TensorListShape<3>({{0, 2, 3}, {1, 1, 3}}), DALI_UINT8);
  input.SetLayout("HWC");
  auto *pixel = input.mutable_tensor<uint8_t>(1);
  pixel[0] = 1;
  pixel[1] = 2;
  pixel[2] = 3;

  Pipeline pipe(2, 1, 0);
  pipe.AddExternalInput("input");
  pipe.AddOperator(OpSpec("Flip")
                       .AddArg("device", "gpu")
                       .AddInput("input", StorageDevice::GPU)
                       .AddOutput("flipped", StorageDevice::GPU));
  pipe.AddOperator(OpSpec("ColorSpaceConversion")
                       .AddArg("device", "gpu")
                       .AddArg("image_type", DALI_RGB)
                       .AddArg("output_type", DALI_BGR)
                       .AddInput("flipped", StorageDevice::GPU)
                       .AddOutput("output", StorageDevice::GPU));
  pipe.Build(std::vector<std::pair<std::string, std::string>>{{"output", "gpu"}});
  pipe.SetExternalInput("input", input);

  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);
  TensorList<CPUBackend> output;
  output.Copy(ws.Output<GPUBackend>(0));
  EXPECT_EQ(output.num_samples(), 2);
  EXPECT_EQ(output.tensor_shape(0), input.tensor_shape(0));
  EXPECT_EQ(output.tensor_shape(1), input.tensor_shape(1));
  const auto *converted = output.tensor<uint8_t>(1);
  EXPECT_EQ(converted[0], 3);
  EXPECT_EQ(converted[1], 2);
  EXPECT_EQ(converted[2], 1);
}

TEST(ColorSpaceConversionTest, ZeroExtentLabSampleCPU) {
  TensorList<CPUBackend> input;
  input.Resize(TensorListShape<3>(std::vector<TensorShape<3>>{{0, 2, 3}}), DALI_UINT8);
  input.SetLayout("HWC");

  Pipeline pipe(1, 1, 0);
  pipe.AddExternalInput("input");
  pipe.AddOperator(OpSpec("ColorSpaceConversion")
                       .AddArg("device", "cpu")
                       .AddArg("image_type", DALI_RGB)
                       .AddArg("output_type", DALI_LAB)
                       .AddInput("input", StorageDevice::CPU)
                       .AddOutput("output", StorageDevice::CPU));
  pipe.Build({{"output", "cpu"}});
  pipe.SetExternalInput("input", input);

  Workspace ws;
  pipe.Run();
  pipe.Outputs(&ws);
  EXPECT_EQ(ws.Output<CPUBackend>(0).tensor_shape(0), input.tensor_shape(0));
}

TEST(ColorSpaceConversionTest, ZeroExtentUnsupportedConversionGPU) {
  TensorList<CPUBackend> input;
  input.Resize(TensorListShape<3>(std::vector<TensorShape<3>>{{0, 1, 3}}), DALI_UINT8);
  input.SetLayout("HWC");

  Pipeline pipe(1, 1, 0);
  pipe.AddExternalInput("input");
  pipe.AddOperator(OpSpec("ColorSpaceConversion")
                       .AddArg("device", "gpu")
                       .AddArg("image_type", DALI_RGB)
                       .AddArg("output_type", DALI_RGB)
                       .AddInput("input", StorageDevice::GPU)
                       .AddOutput("output", StorageDevice::GPU));
  pipe.Build(std::vector<std::pair<std::string, std::string>>{{"output", "gpu"}});
  pipe.SetExternalInput("input", input);

  Workspace ws;
  EXPECT_ANY_THROW({
    pipe.Run();
    pipe.Outputs(&ws);
  });
}

TEST(ColorSpaceConversionTest, LabWhitePoint) {
  TensorList<CPUBackend> input;
  input.Resize(TensorListShape<3>(std::vector<TensorShape<3>>{{1, 3, 3}}), DALI_UINT8);
  input.SetLayout("HWC");
  auto *px = input.mutable_tensor<uint8_t>(0);
  // Red, Green, Blue
  px[0] = 255; px[1] = 0;   px[2] = 0;
  px[3] = 0;   px[4] = 255; px[5] = 0;
  px[6] = 0;   px[7] = 0;   px[8] = 255;

  std::array<std::array<uint8_t, 9>, 2> cpu_lab = {};
  for (int white_point_idx = 0; white_point_idx < 2; white_point_idx++) {
    float wp = white_point_idx == 0 ? 65.0f : 50.0f;
    Pipeline pipe(1, 1, 0);
    pipe.AddExternalInput("input");
    pipe.AddOperator(OpSpec("ColorSpaceConversion")
                         .AddArg("device", "cpu")
                         .AddArg("image_type", DALI_RGB)
                         .AddArg("output_type", DALI_LAB)
                         .AddArg("white_point", wp)
                         .AddInput("input", StorageDevice::CPU)
                         .AddOutput("lab_cpu", StorageDevice::CPU));
    pipe.AddOperator(OpSpec("ColorSpaceConversion")
                         .AddArg("device", "gpu")
                         .AddArg("image_type", DALI_RGB)
                         .AddArg("output_type", DALI_LAB)
                         .AddArg("white_point", wp)
                         .AddInput("input", StorageDevice::GPU)
                         .AddOutput("lab_gpu", StorageDevice::GPU));
    pipe.Build({{"lab_cpu", "cpu"}, {"lab_gpu", "gpu"}});
    pipe.SetExternalInput("input", input);

    Workspace ws;
    pipe.Run();
    pipe.Outputs(&ws);

    const auto &lab_cpu = ws.Output<CPUBackend>(0);
    TensorList<CPUBackend> lab_gpu;
    lab_gpu.Copy(ws.Output<GPUBackend>(1));

    const auto *c_ptr = lab_cpu.tensor<uint8_t>(0);
    const auto *g_ptr = lab_gpu.tensor<uint8_t>(0);
    for (int i = 0; i < 9; i++) {
      cpu_lab[white_point_idx][i] = c_ptr[i];
      EXPECT_NEAR(c_ptr[i], g_ptr[i], 3);
    }
  }
  EXPECT_NE(cpu_lab[0], cpu_lab[1]);
  EXPECT_EQ(cpu_lab[0][0], 136);
  EXPECT_EQ(cpu_lab[0][1], 208);
  EXPECT_EQ(cpu_lab[0][2], 195);
  EXPECT_EQ(cpu_lab[1][0], 138);
  EXPECT_EQ(cpu_lab[1][1], 209);
  EXPECT_EQ(cpu_lab[1][2], 198);
}

TEST(ColorSpaceConversionTest, LabToRgbAndBgrCpuGpu) {
  TensorList<CPUBackend> input;
  input.Resize(TensorListShape<3>(std::vector<TensorShape<3>>{{1, 3, 3}}), DALI_UINT8);
  input.SetLayout("HWC");
  auto *px = input.mutable_tensor<uint8_t>(0);
  // D65 and D50 Lab encodings of red, green, and blue, respectively.
  const std::array<std::array<uint8_t, 9>, 2> lab_pixels = {{
      {{136, 208, 195, 224, 42, 211, 82, 207, 20}},
      {{138, 209, 198, 224, 49, 209, 75, 196, 16}}
  }};
  const std::array<uint8_t, 9> rgb_pixels = {{
      255, 0, 0,
      0, 255, 0,
      0, 0, 255
  }};

  for (int white_point_idx = 0; white_point_idx < 2; white_point_idx++) {
    float wp = white_point_idx == 0 ? 65.0f : 50.0f;
    for (int i = 0; i < 9; i++) {
      px[i] = lab_pixels[white_point_idx][i];
    }
    for (DALIImageType output_type : {DALI_RGB, DALI_BGR}) {
      Pipeline pipe(1, 1, 0);
      pipe.AddExternalInput("input");
      pipe.AddOperator(OpSpec("ColorSpaceConversion")
                           .AddArg("device", "cpu")
                           .AddArg("image_type", DALI_LAB)
                           .AddArg("output_type", output_type)
                           .AddArg("white_point", wp)
                           .AddInput("input", StorageDevice::CPU)
                           .AddOutput("rgb_cpu", StorageDevice::CPU));
      pipe.AddOperator(OpSpec("ColorSpaceConversion")
                           .AddArg("device", "gpu")
                           .AddArg("image_type", DALI_LAB)
                           .AddArg("output_type", output_type)
                           .AddArg("white_point", wp)
                           .AddInput("input", StorageDevice::GPU)
                           .AddOutput("rgb_gpu", StorageDevice::GPU));
      pipe.Build({{"rgb_cpu", "cpu"}, {"rgb_gpu", "gpu"}});
      pipe.SetExternalInput("input", input);

      Workspace ws;
      pipe.Run();
      pipe.Outputs(&ws);

      const auto &rgb_cpu = ws.Output<CPUBackend>(0);
      TensorList<CPUBackend> rgb_gpu;
      rgb_gpu.Copy(ws.Output<GPUBackend>(1));

      const auto *c_ptr = rgb_cpu.tensor<uint8_t>(0);
      const auto *g_ptr = rgb_gpu.tensor<uint8_t>(0);
      for (int i = 0; i < 9; i++) {
        EXPECT_NEAR(c_ptr[i], g_ptr[i], 3);
        int channel = i % 3;
        int expected_channel = output_type == DALI_RGB ? channel : 2 - channel;
        EXPECT_NEAR(c_ptr[i], rgb_pixels[i - channel + expected_channel], 12);
      }
    }
  }
}

TEST(ColorSpaceConversionTest, LabInvalidWhitePointThrows) {
  Pipeline pipe(1, 1, 0);
  pipe.AddExternalInput("input");
  pipe.AddOperator(OpSpec("ColorSpaceConversion")
                       .AddArg("device", "cpu")
                       .AddArg("image_type", DALI_RGB)
                       .AddArg("output_type", DALI_LAB)
                       .AddArg("white_point", 55.0f)
                       .AddInput("input", StorageDevice::CPU)
                       .AddOutput("output", StorageDevice::CPU));
  EXPECT_THROW(pipe.Build({{"output", "cpu"}}), DALIException);
}

}  // namespace dali
