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


// Rotation/flip tests for ConvertCPU. These mirror the rotation tests in convert_gpu_test.cc
// to guarantee CPU and GPU paths agree on EXIF orientation semantics.
class ConvertOrientationTest : public ::testing::Test {
 protected:
  static constexpr int kH = 2;
  static constexpr int kW = 3;
  static constexpr int kC = 3;

  // 2x3 RGB input; each pixel uniquely identifiable so axis mix-ups surface as mismatches.
  static std::vector<uint8_t> MakeInput() {
    return {
      0x00, 0x01, 0x02,  0x10, 0x11, 0x12,  0x20, 0x21, 0x22,  // row 0: A B C
      0x30, 0x31, 0x32,  0x40, 0x41, 0x42,  0x50, 0x51, 0x52,  // row 1: D E F
    };
  }

  void Run(int rotated, bool flip_x, bool flip_y, std::vector<uint8_t> expected) {
    bool swap = rotated % 180 == 90;
    int out_h = swap ? kW : kH;
    int out_w = swap ? kH : kW;
    ASSERT_EQ(expected.size(), static_cast<size_t>(out_h * out_w * kC));

    auto input_buf = MakeInput();
    TensorShape<> in_shape{kH, kW, kC};
    TensorShape<> out_shape{out_h, out_w, kC};
    ConstSampleView<CPUBackend> in_view(input_buf.data(), in_shape, DALI_UINT8);
    std::vector<uint8_t> output_buf(out_h * out_w * kC, 0xFF);
    SampleView<CPUBackend> out_view(output_buf.data(), out_shape, DALI_UINT8);

    nvimgcodecOrientation_t orientation{
        NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t),
        nullptr, rotated, flip_x, flip_y};
    ConvertCPU(out_view, "HWC", DALI_RGB, in_view, "HWC", DALI_RGB, {}, orientation);
    EXPECT_EQ(output_buf, expected);
  }
};

TEST_F(ConvertOrientationTest, Rotated90) {
  // Matches convert_gpu_test.cc Rotation90: A→(2,0), C→(0,0), F→(0,1)
  Run(90, false, false, {
    0x20, 0x21, 0x22,  0x50, 0x51, 0x52,
    0x10, 0x11, 0x12,  0x40, 0x41, 0x42,
    0x00, 0x01, 0x02,  0x30, 0x31, 0x32,
  });
}

TEST_F(ConvertOrientationTest, Rotated270) {
  // Matches convert_gpu_test.cc Rotation270: A→(0,1), D→(0,0)
  Run(270, false, false, {
    0x30, 0x31, 0x32,  0x00, 0x01, 0x02,
    0x40, 0x41, 0x42,  0x10, 0x11, 0x12,
    0x50, 0x51, 0x52,  0x20, 0x21, 0x22,
  });
}

TEST_F(ConvertOrientationTest, Rotated180) {
  Run(180, false, false, {
    0x50, 0x51, 0x52,  0x40, 0x41, 0x42,  0x30, 0x31, 0x32,
    0x20, 0x21, 0x22,  0x10, 0x11, 0x12,  0x00, 0x01, 0x02,
  });
}

TEST_F(ConvertOrientationTest, Rotated90FlipX) {
  // Matches convert_gpu_test.cc Rotation90FlipX: input flipped along W, then rotated 90
  Run(90, true, false, {
    0x50, 0x51, 0x52,  0x20, 0x21, 0x22,
    0x40, 0x41, 0x42,  0x10, 0x11, 0x12,
    0x30, 0x31, 0x32,  0x00, 0x01, 0x02,
  });
}

TEST_F(ConvertOrientationTest, FlipX) {
  Run(0, true, false, {
    0x20, 0x21, 0x22,  0x10, 0x11, 0x12,  0x00, 0x01, 0x02,
    0x50, 0x51, 0x52,  0x40, 0x41, 0x42,  0x30, 0x31, 0x32,
  });
}

TEST_F(ConvertOrientationTest, FlipY) {
  Run(0, false, true, {
    0x30, 0x31, 0x32,  0x40, 0x41, 0x42,  0x50, 0x51, 0x52,
    0x00, 0x01, 0x02,  0x10, 0x11, 0x12,  0x20, 0x21, 0x22,
  });
}

// Regression test for the CPU-side signed-source color-conversion fix: the color-conversion
// coefficients/biases are calibrated for a channel already normalized to Out's range, so a
// signed In must be normalized to Out first (as ConvertGPU already does), instead of running the
// color math directly on the raw signed value.
class ConvertSignedSourceTest : public ::testing::Test {
 protected:
  Tensor<CPUBackend> Convert(std::vector<int8_t> rgb, DALIImageType out_format,
                              int out_channels) {
    TensorShape<> in_shape{1, 1, 3};
    ConstSampleView<CPUBackend> in_view(rgb.data(), in_shape, DALI_INT8);
    Tensor<CPUBackend> out;
    out.Resize(TensorShape<>{1, 1, out_channels}, DALI_UINT8);
    SampleView<CPUBackend> out_view(out.raw_mutable_data(), out.shape(), out.type());
    ConvertCPU(out_view, "HWC", out_format, in_view, "HWC", DALI_RGB, {});
    return out;
  }
};

TEST_F(ConvertSignedSourceTest, AchromaticRgbToGrayMatchesNormalizedInput) {
  // For an achromatic pixel (R == G == B), the gray formula's coefficients sum to exactly 1.0,
  // so the correctly-converted gray value must equal the plain range-normalized input, regardless
  // of the color-conversion math. Before the fix, ConvertCPU ran that math on the raw *signed*
  // int8 value instead of first normalizing it to the output's [0, 255] range, so a negative
  // sample produced wildly wrong, unclamped output instead of this simple identity.
  for (int8_t v : {int8_t{-128}, int8_t{-1}, int8_t{0}, int8_t{1}, int8_t{127}}) {
    auto out = Convert({v, v, v}, DALI_GRAY, 1);
    uint8_t expected = ConvertSatNorm<uint8_t>(v);
    EXPECT_EQ(*out.data<uint8_t>(), expected) << "v=" << static_cast<int>(v);
  }
}

TEST_F(ConvertSignedSourceTest, RgbToGrayMatchesJpegFormulaOnNormalizedInput) {
  // Non-achromatic case: the expected value is computed independently from the normalized
  // (ConvertSatNorm) RGB values using the standard JPEG luma weights, mirroring what the fixed
  // ConvertPixel<Out, In, DALI_GRAY, DALI_RGB> does, but recomputed here rather than re-using its
  // internal helper.
  std::vector<int8_t> rgb = {int8_t{-128}, int8_t{0}, int8_t{127}};
  auto out = Convert(rgb, DALI_GRAY, 1);
  float r = ConvertSatNorm<uint8_t>(rgb[0]);
  float g = ConvertSatNorm<uint8_t>(rgb[1]);
  float b = ConvertSatNorm<uint8_t>(rgb[2]);
  uint8_t expected = ConvertSat<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
  EXPECT_EQ(*out.data<uint8_t>(), expected);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
