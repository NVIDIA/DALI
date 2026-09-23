// Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST(ColorSpaceConversionTest, ZeroExtentUnsupportedConversionGPU) {
  TensorList<CPUBackend> input;
  input.Resize(TensorListShape<3>({{0, 1, 3}}), DALI_UINT8);
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

}  // namespace dali
