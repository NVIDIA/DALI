// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include <vector>
#include <memory>
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/mm/memory.h"
#include "dali/imgcodec/util/convert.h"
#include "dali/imgcodec/decoders/decoder_test_helper.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {
const auto &dali_extra = dali::testing::dali_extra_path();
auto dir = dali_extra + "/db/imgcodec/colorspaces/";
auto img = "cat-111793_640";

std::string image_path(const std::string &name, const std::string &type_name) {
  return make_string(dir, img, "_", name, "_", type_name, ".npy");
}

}  // namespace

template <typename ImageType>
class ColorConversionTest : public NumpyDecoderTestBase<ImageType> {
 public:
  Tensor<CPUBackend> RunConvert(const std::string& input_path,
                                DALIImageType input_format, DALIImageType output_format) {
    auto input = this->ReadReferenceFrom(input_path);
    ConstSampleView<CPUBackend> input_view(input.raw_mutable_data(), input.shape(), input.type());

    Tensor<CPUBackend> output;
    output.Resize({input.shape()[0], input.shape()[1], NumberOfChannels(output_format)},
                  input.type());
    SampleView<CPUBackend> output_view(output.raw_mutable_data(), output.shape(), output.type());

    Convert(output_view, TensorLayout("HWC"), output_format,
            input_view, TensorLayout("HWC"), input_format,
            {}, {});

    return output;
  }

  void Test(const std::string& input_name,
            DALIImageType input_format, DALIImageType output_format,
            const std::string& reference_name) {
    auto input_path = this->InputImagePath(input_name);
    auto reference_path = this->ReferencePath(reference_name);
    this->AssertClose(this->RunConvert(input_path, input_format, output_format),
                      this->ReadReferenceFrom(reference_path), this->Eps());
  }

  std::string InputImagePath(const std::string &name) {
    return image_path(name, this->TypeName());
  }

  std::string ReferencePath(const std::string &name) {
    return image_path(name, "float");
  }

  std::string TypeName() {
    if (std::is_same_v<uint8_t, ImageType>) return "uint8";
    else if (std::is_same_v<float, ImageType>) return "float";
    else
      DALI_FAIL("Unsupported ImageType in ColorConversionTest");
  }

  double Eps() {
    if (std::is_same_v<uint8_t, ImageType>) return 3.001;
    else if (std::is_same_v<float, ImageType>) return 0.001;
    else
      DALI_FAIL("Unsupported ImageType in ColorConversionTest");
  }

 protected:
  // This test will only read numpy files, so we don't need a decoder/parser
  std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) override {
    return nullptr;
  }
  std::shared_ptr<ImageParser> CreateParser() override {
    return nullptr;
  }
};

using ImageTypes = ::testing::Types<uint8_t, float>;
TYPED_TEST_SUITE(ColorConversionTest, ImageTypes);

TYPED_TEST(ColorConversionTest, RgbToGray) {
  this->Test("rgb", DALI_RGB, DALI_GRAY, "gray");
}

TYPED_TEST(ColorConversionTest, GrayToRgb) {
  this->Test("gray", DALI_GRAY, DALI_RGB, "rgb_from_gray");
}

TYPED_TEST(ColorConversionTest, RgbToYCbCr) {
  this->Test("rgb", DALI_RGB, DALI_YCbCr, "ycbcr");
}

TYPED_TEST(ColorConversionTest, YCbCrToRgb) {
  this->Test("ycbcr", DALI_YCbCr, DALI_RGB, "rgb");
}

TYPED_TEST(ColorConversionTest, YCbCrToGray) {
  this->Test("ycbcr", DALI_YCbCr, DALI_GRAY, "gray");
}

TYPED_TEST(ColorConversionTest, GrayToYCbCr) {
  this->Test("gray", DALI_GRAY, DALI_YCbCr, "ycbcr_from_gray");
}

TYPED_TEST(ColorConversionTest, RgbToBgr) {
  this->Test("rgb", DALI_RGB, DALI_BGR, "bgr");
}

TYPED_TEST(ColorConversionTest, BgrToRgb) {
  this->Test("bgr", DALI_BGR, DALI_RGB, "rgb");
}

TYPED_TEST(ColorConversionTest, BgrToGray) {
  this->Test("bgr", DALI_BGR, DALI_GRAY, "gray");
}

TYPED_TEST(ColorConversionTest, BgrToYCbCr) {
  this->Test("bgr", DALI_BGR, DALI_YCbCr, "ycbcr");
}

TYPED_TEST(ColorConversionTest, GrayToBgr) {
  this->Test("gray", DALI_GRAY, DALI_BGR, "bgr_from_gray");
}

TYPED_TEST(ColorConversionTest, YCbCrToBgr) {
  this->Test("ycbcr", DALI_YCbCr, DALI_BGR, "bgr");
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
