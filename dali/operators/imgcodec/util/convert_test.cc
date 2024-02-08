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

#include <gtest/gtest.h>
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/operators/imgcodec/decoder_test_helper.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {
const auto &dali_extra = dali::testing::dali_extra_path();
auto colorspace_dir = dali_extra + "/db/imgcodec/colorspaces/";
auto colorspace_img = "cat-111793_640";

/**
 * @brief Returns a full path of an image.
 *
 * The images used in the tests are stored as numpy files. The same picture is stored in different
 * types and color spaces. This functions allows to get the path of the appropriate image.
 *
 * @param name Name of the image. In most cases that's the name of color space, for example "rgb"
 * @param type_name Name of image datatype, for example "float" or "uint8".
 */
std::string image_path(const std::string &name, const std::string &type_name) {
  return make_string(colorspace_dir, colorspace_img, "_", name, "_", type_name, ".npy");
}

}  // namespace

template <typename ImageType>
class ConversionTestBase : public ::testing::Test {
 protected:
  /**
   * @brief Reads an image and converts it from `input_format` to `output_format`
   */
  Tensor<CPUBackend> RunConvert(const std::string& input_path,
                                DALIImageType input_format, DALIImageType output_format,
                                TensorLayout layout = "HWC", const ROI &roi = {},
                                int channels_hint = 0) {
    auto input = ReadReferenceFrom(input_path);
    ConstSampleView<CPUBackend> input_view(input.raw_mutable_data(), input.shape(), input.type());

    Tensor<CPUBackend> output;
    int output_channels = NumberOfChannels(output_format,
                                           NumberOfChannels(input_format, channels_hint));
    auto output_shape = input.shape();
    int channel_index = ImageLayoutInfo::ChannelDimIndex(layout);
    assert(channel_index >= 0);
    output_shape[channel_index] = output_channels;
    if (roi) {
      for (int d = 0; d < channel_index; d++)
        output_shape[d] = roi.shape()[d];
      for (int d = channel_index + 1; d < output_shape.size(); d++)
        output_shape[d] = roi.shape()[d - 1];
    }
    output.Resize(output_shape, input.type());
    SampleView<CPUBackend> output_view(output.raw_mutable_data(), output.shape(), output.type());

    ConvertCPU(output_view, layout, output_format,
            input_view, layout, input_format,
            roi);

    return output;
  }
};

/**
 * @brief A class template for testing color conversion of images
 *
 * @tparam ImageType The type of images to run the test on
 */
template <typename ImageType>
class ColorConversionTest : public ConversionTestBase<ImageType> {
 public:
  /**
   * @brief Checks if the conversion result matches the reference.
   *
   * Reads an image named `input_name`, converts it from `input_format` to `output_format` and
   * checks if the result is close enough (see Eps()) to the reference image.
   */
  void Test(const std::string& input_name,
            DALIImageType input_format, DALIImageType output_format,
            const std::string& reference_name, int channels_hint = 0) {
    auto input_path = this->InputImagePath(input_name);
    auto reference_path = this->ReferencePath(reference_name);
    AssertClose(this->RunConvert(input_path, input_format, output_format, "HWC", {}, channels_hint),
                ReadReferenceFrom(reference_path), this->Eps());
  }

 protected:
  /**
   * @brief Returns a path of input image
   */
  std::string InputImagePath(const std::string &name) {
    return image_path(name, this->TypeName());
  }

  /**
   * @brief Returns a path of reference image
   */
  std::string ReferencePath(const std::string &name) {
    return image_path(name, "float");  // We always use floats for reference
  }

  /**
   * @brief Returns the name of the `ImageType` type, which is needed to get image paths
   */
  std::string TypeName() {
    if (std::is_same_v<uint8_t, ImageType>) return "uint8";
    else if (std::is_same_v<float, ImageType>) return "float";
    else
      throw std::logic_error("Invalid ImageType in ColorConversionTest");
  }

  /**
   * @brief Returns the allowed absolute error when comparing images.
   */
  double Eps() {
    // The eps for uint8 is relatively high to account for accumulating rounding errors and
    // inconsistencies in the conversion functions.
    if (std::is_same_v<uint8_t, ImageType>) return 3.001;
    else if (std::is_same_v<float, ImageType>) return 0.001;
    else
      return 0;
  }
};


using ImageTypes = ::testing::Types<uint8_t, float>;
TYPED_TEST_SUITE(ColorConversionTest, ImageTypes);

TYPED_TEST(ColorConversionTest, AnyToAny) {
  // DALI_ANY_DATA -> DALI_ANY_DATA should be a no-op
  this->Test("rgb", DALI_ANY_DATA, DALI_ANY_DATA, "rgb", 3);
}

// RGB -> *

TYPED_TEST(ColorConversionTest, RgbToRgb) {
  this->Test("rgb", DALI_RGB, DALI_RGB, "rgb");
}

TYPED_TEST(ColorConversionTest, RgbToGray) {
  this->Test("rgb", DALI_RGB, DALI_GRAY, "gray");
}

TYPED_TEST(ColorConversionTest, RgbToYCbCr) {
  this->Test("rgb", DALI_RGB, DALI_YCbCr, "ycbcr");
}

TYPED_TEST(ColorConversionTest, RgbToBgr) {
  this->Test("rgb", DALI_RGB, DALI_BGR, "bgr");
}

TYPED_TEST(ColorConversionTest, RgbToAny) {
  // Conversion to DALI_ANY_DATA should be a no-op
  this->Test("rgb", DALI_RGB, DALI_ANY_DATA, "rgb");
}


// Gray -> *

TYPED_TEST(ColorConversionTest, GrayToRgb) {
  this->Test("gray", DALI_GRAY, DALI_RGB, "rgb_from_gray");
}

TYPED_TEST(ColorConversionTest, GrayToGray) {
  this->Test("gray", DALI_GRAY, DALI_GRAY, "gray");
}

TYPED_TEST(ColorConversionTest, GrayToYCbCr) {
  this->Test("gray", DALI_GRAY, DALI_YCbCr, "ycbcr_from_gray");
}

TYPED_TEST(ColorConversionTest, GrayToBgr) {
  this->Test("gray", DALI_GRAY, DALI_BGR, "bgr_from_gray");
}


// YCbCr -> *

TYPED_TEST(ColorConversionTest, YCbCrToRgb) {
  this->Test("ycbcr", DALI_YCbCr, DALI_RGB, "rgb");
}

TYPED_TEST(ColorConversionTest, YCbCrToGray) {
  this->Test("ycbcr", DALI_YCbCr, DALI_GRAY, "gray");
}

TYPED_TEST(ColorConversionTest, YCbCrToYCbCr) {
  this->Test("ycbcr", DALI_YCbCr, DALI_YCbCr, "ycbcr");
}

TYPED_TEST(ColorConversionTest, YCbCrToBgr) {
  this->Test("ycbcr", DALI_YCbCr, DALI_BGR, "bgr");
}


// BGR -> *

TYPED_TEST(ColorConversionTest, BgrToRgb) {
  this->Test("bgr", DALI_BGR, DALI_RGB, "rgb");
}

TYPED_TEST(ColorConversionTest, BgrToGray) {
  this->Test("bgr", DALI_BGR, DALI_GRAY, "gray");
}

TYPED_TEST(ColorConversionTest, BgrToYCbCr) {
  this->Test("bgr", DALI_BGR, DALI_YCbCr, "ycbcr");
}

TYPED_TEST(ColorConversionTest, BgrToBgr) {
  this->Test("bgr", DALI_BGR, DALI_BGR, "bgr");
}


/**
 * @brief Class for testing Convert's support of different layouts.
 */
class ConvertLayoutTest : public ConversionTestBase<float> {
 public:
  void Test(const std::string& layout_name, const ROI &roi = {}) {
    auto rgb_path = GetPath(layout_name, "rgb");
    auto ycbcr_path = GetPath(layout_name, "ycbcr");

    std::string layout_code = layout_name;
    for (auto &c : layout_code) c = toupper(c);
    TensorLayout layout(layout_code);

    auto ref = ReadReferenceFrom(ycbcr_path);
    ref.SetLayout(layout);
    if (roi) {
      ref = Crop(ref, roi);
    }
    AssertClose(RunConvert(rgb_path, DALI_RGB, DALI_YCbCr, layout, roi), ref, 0.01);
  }

 protected:
  std::string GetPath(const std::string &layout_name, const std::string colorspace_name) {
    return make_string(colorspace_dir, "layouts/", colorspace_img, "_", colorspace_name,
                       "_float_", layout_name, ".npy");
  }
};

TEST_F(ConvertLayoutTest, HWC) {
  Test("hwc");
}

TEST_F(ConvertLayoutTest, CHW) {
  Test("chw");
}

TEST_F(ConvertLayoutTest, RoiHWC) {
  Test("hwc", {{20, 30}, {200, 300}});
}

TEST_F(ConvertLayoutTest, RoiCHW) {
  Test("chw", {{20, 30}, {200, 300}});
}


}  // namespace test
}  // namespace imgcodec
}  // namespace dali
