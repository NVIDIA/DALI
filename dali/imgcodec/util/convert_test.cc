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

auto rgb_path = dir + "/cat-111793_640_rgb_uint8.npy";
auto gray_path = dir + "/cat-111793_640_gray_uint8.npy";
auto ycbcr_path = dir + "/cat-111793_640_ycbcr_uint8.npy";

auto rgb_ref_path = dir + "/cat-111793_640_rgb_float.npy";
auto gray_ref_path = dir + "/cat-111793_640_gray_float.npy";
auto ycbcr_ref_path = dir + "/cat-111793_640_ycbcr_float.npy";
}  // namespace

class ColorConversionTest : public NumpyDecoderTestBase<uint8_t> {
 public:
  Tensor<CPUBackend> RunConvert(const std::string& input_path,
                                DALIImageType input_format, DALIImageType output_format) {
    auto input = ReadReferenceFrom(input_path);
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

  void Test(const std::string& input_path,
            DALIImageType input_format, DALIImageType output_format,
            const std::string& reference_path) {
    AssertClose(RunConvert(input_path, input_format, output_format),
                ReadReferenceFrom(reference_path), 0.501);
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

TEST_F(ColorConversionTest, RgbToGray) {
  Test(rgb_path, DALI_RGB, DALI_GRAY, gray_ref_path);
}

TEST_F(ColorConversionTest, GrayToRgb) {
  auto rgb = RunConvert(gray_path, DALI_GRAY, DALI_RGB);

  auto w = rgb.shape()[0], h = rgb.shape()[1];
  auto red   = Crop(rgb, {{0, 0, 0}, {w, h, 1}});
  auto green = Crop(rgb, {{0, 0, 1}, {w, h, 2}});
  auto blue  = Crop(rgb, {{0, 0, 2}, {w, h, 3}});

  auto gray = ReadReferenceFrom(gray_path);

  AssertClose(red, gray, 0.501);
  AssertClose(green, gray, 0.501);
  AssertClose(blue, gray, 0.501);
}

TEST_F(ColorConversionTest, RgbToYcbCr) {
  Test(rgb_path, DALI_RGB, DALI_YCbCr, ycbcr_ref_path);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
