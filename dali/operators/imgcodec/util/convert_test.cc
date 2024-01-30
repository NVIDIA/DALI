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
#include <string>
#include <vector>
#include <memory>
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/mm/memory.h"
#include "dali/operators/imgcodec/util/convert.h"
#include "dali/util/file.h"
#include "dali/util/numpy.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/pipeline/data/views.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {
const auto &dali_extra = dali::testing::dali_extra_path();
auto colorspace_dir = dali_extra + "/db/imgcodec/colorspaces/";
auto colorspace_img = "cat-111793_640";
auto orientation_dir = dali_extra + "/db/imgcodec/orientation/";
auto orientation_img = "kitty-2948404_640";

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

static cv::Mat rgb2bgr(const cv::Mat &img) {
  cv::Mat bgr;
  cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);
  return bgr;
}

static cv::Mat bgr2rgb(const cv::Mat &img) {
  return rgb2bgr(img);
}

/**
 * @brief Extends ROI with channel dimension of given shape.
 */
static ROI ExtendRoi(ROI roi, const TensorShape<> &shape, const TensorLayout &layout) {
  int channel_dim = ImageLayoutInfo::ChannelDimIndex(layout);
  if (channel_dim == -1)
    channel_dim = shape.size() - 1;

  int ndim = shape.sample_dim();
  if (roi.begin.size() == ndim - 1) {
    roi.begin = shape_cat(shape_cat(roi.begin.first(channel_dim), 0),
                          roi.begin.last(roi.begin.size() - channel_dim));
  }
  if (roi.end.size() == ndim - 1) {
    roi.end = shape_cat(shape_cat(roi.end.first(channel_dim), shape[channel_dim]),
                        roi.end.last(roi.end.size() - channel_dim));
  }
  return roi;
}

static TensorShape<> AdjustToRoi(const TensorShape<> &shape, const ROI &roi) {
  if (roi) {
    auto result = roi.shape();
    int ndim = shape.sample_dim();
    if (roi.shape().sample_dim() != ndim) {
      assert(roi.shape().sample_dim() + 1 == ndim);
      result.resize(ndim);
      result[ndim - 1] = shape[ndim - 1];
    }
    return result;
  } else {
    return shape;
  }
}

/**
 * @brief Crops a tensor to specified roi_shape, anchored at roi_begin.
 * Does not support padding.
 */
template <typename T, int ndim>
void Crop(const TensorView<StorageCPU, T, ndim> &output,
          const TensorView<StorageCPU, const T, ndim> &input, const ROI &requested_roi,
          const TensorLayout &layout = "HWC") {
  auto roi = ExtendRoi(requested_roi, input.shape, layout);

  static_assert(ndim >= 0, "expected static ndim");
  ASSERT_TRUE(output.shape == roi.shape());  // output should have the desired shape

  kernels::SliceCPU<T, T, ndim> kernel;
  kernels::SliceArgs<T, ndim> args;
  args.anchor = roi.begin;
  args.shape = roi.shape();
  kernels::KernelContext ctx;
  // no need to run Setup (we already know the output shape)
  kernel.Run(ctx, output, input, args);
}

template <typename T, int ndim>
Tensor<CPUBackend> Crop(const TensorView<StorageCPU, const T, ndim> &input,
                        const ROI &requested_roi, const TensorLayout &layout = "HWC") {
  auto roi = ExtendRoi(requested_roi, input.shape, layout);
  auto num_dims = input.shape.sample_dim();
  assert(roi.shape().sample_dim() == num_dims);
  Tensor<CPUBackend> output;
  output.Resize(roi.shape(), type2id<T>::value);

  auto out_view = view<T, ndim>(output);
  Crop(out_view, input, roi);

  return output;
}

static Tensor<CPUBackend> Crop(const Tensor<CPUBackend> &input, const ROI &roi) {
  int ndim = input.shape().sample_dim();
  VALUE_SWITCH(ndim, Dims, (2, 3, 4), (
    TYPE_SWITCH(input.type(), type2id, InputType, (IMGCODEC_TYPES, double), (
      return Crop(view<const InputType, Dims>(input), roi, input.GetLayout());
    ), DALI_FAIL(make_string("Unsupported type ", input.type())););  // NOLINT
  ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT
}

/**
 * @brief Checks if the image and the reference are equal
 */
template <typename T>
void AssertEqual(const TensorView<StorageCPU, const T> &img,
                 const TensorView<StorageCPU, const T> &ref) {
  Check(img, ref);
}

/**
 * @brief Checks if the image and the reference are equal after converting the reference
 * with ConvertSatNorm
 */
template <typename T, typename RefType>
void AssertEqualSatNorm(const TensorView<StorageCPU, const T> &img,
                        const TensorView<StorageCPU, const RefType> &ref) {
  Check(img, ref, EqualConvertSatNorm());
}

template <typename T>
void AssertEqualSatNorm(const TensorView<StorageCPU, const T> &img, const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    Check(img, view<const RefType>(ref), EqualConvertSatNorm());
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

static void AssertEqualSatNorm(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(img.type(), type2id, T, (IMGCODEC_TYPES), (
    AssertEqualSatNorm<T>(view<const T>(img), ref);
  ), DALI_FAIL(make_string("Unsupported type: ", img.type())));  // NOLINT
}


/**
 * @brief Checks that the image is similar to the reference based on mean square error
 *
 * Unlike AssertClose, this check allows for a rather large max. error, but has a much lower
 * limit on mean square error. It's used for different JPEG decoders which have differences in
 * chroma upsampling or apply deblocking filters.
 */
template <typename OutputType, typename RefType>
void AssertSimilar(const TensorView<StorageCPU, const OutputType> &img,
                   const TensorView<StorageCPU, const RefType> &ref) {
  float eps = ConvertSatNorm<OutputType>(0.3);
  Check(img, ref, EqualConvertNorm(eps));

  double mean_square_error = 0;
  uint64_t size = volume(img.shape);
  for (size_t i = 0; i < size; i++) {
    double img_value = ConvertSatNorm<double>(img.data[i]);
    double ref_value = ConvertSatNorm<double>(ref.data[i]);
    double error = img_value - ref_value;
    mean_square_error += error * error;
  }
  mean_square_error = sqrt(mean_square_error / size);
  if (mean_square_error >= 0.04 && img.shape[2] == 3) {
    cv::Mat img_mat(img.shape[0], img.shape[1], CV_8UC3,
                    const_cast<void *>(reinterpret_cast<const void *>(img.data)));
    cv::Mat ref_mat(ref.shape[0], ref.shape[1], CV_8UC3,
                    const_cast<void *>(reinterpret_cast<const void *>(ref.data)));
    cv::Mat diff;
    cv::absdiff(img_mat, ref_mat, diff);

    cv::imwrite("out.bmp", rgb2bgr(img_mat));
    cv::imwrite("ref.bmp", rgb2bgr(ref_mat));
    cv::imwrite("diff.bmp", rgb2bgr(diff));
  }
  EXPECT_LT(mean_square_error, 0.04);
}


template <typename T>
void AssertSimilar(const TensorView<StorageCPU, const T> &img, const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    AssertSimilar(img, view<const RefType>(ref));
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

static void AssertSimilar(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref) {
  TYPE_SWITCH(img.type(), type2id, T, (IMGCODEC_TYPES), (
    AssertSimilar<T>(view<const T>(img), ref);
  ), DALI_FAIL(make_string("Unsupported type: ", img.type())));  // NOLINT
}


/**
 * @brief Checks if an image is close to a reference
 *
 * The eps parameter should be specified in the dynamic range of the image.
 */
template <typename T, typename RefType>
void AssertClose(const TensorView<StorageCPU, const T> &img,
                 const TensorView<StorageCPU, const RefType> &ref, float eps) {
  if (std::is_integral<T>::value)
    eps /= max_value<T>();
  Check(img, ref, EqualConvertNorm(eps));
}

template <typename T>
void AssertClose(const TensorView<StorageCPU, const T> &img, const Tensor<CPUBackend> &ref,
                 float eps) {
  TYPE_SWITCH(ref.type(), type2id, RefType, NUMPY_ALLOWED_TYPES, (
    AssertClose(img, view<const RefType>(ref), eps);
  ), DALI_FAIL(make_string("Unsupported reference type: ", ref.type())));  // NOLINT
}

static void AssertClose(const Tensor<CPUBackend> &img, const Tensor<CPUBackend> &ref, float eps) {
  TYPE_SWITCH(img.type(), type2id, T, (IMGCODEC_TYPES), (
    AssertClose<T>(view<const T>(img), ref, eps);
  ), DALI_FAIL(make_string("Unsupported type: ", img.type())));  // NOLINT
}

Tensor<CPUBackend> ReadReference(InputStream *src, TensorLayout layout = "HWC") {
  auto tensor = numpy::ReadTensor(src);
  tensor.SetLayout(layout);
  return tensor;
}

/**
 * @brief Reads the reference image from specified path and returns it as a tensor.
 */
Tensor<CPUBackend> ReadReferenceFrom(const std::string &reference_path,
                                      TensorLayout layout = "HWC") {
  auto src = FileStream::Open(reference_path, false, false);
  return ReadReference(src.get(), layout);
}


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
