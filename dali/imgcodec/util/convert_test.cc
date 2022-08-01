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
auto dir = dali_extra + "/db/single/reference/tiff/0/";
auto rgb_path = dir + "/cat-111793_640.tiff.npy";
auto gray_path = dir + "/cat-111793_640_bw.tiff.npy";
}  // namespace

class ColorConversionTest : public NumpyDecoderTestBase<uint8_t> {
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
  auto rgb = ReadReferenceFrom(rgb_path);
  ConstSampleView<CPUBackend> rgb_view(rgb.raw_mutable_data(), rgb.shape(), rgb.type());

  Tensor<CPUBackend> gray;
  gray.Resize({rgb.shape()[0], rgb.shape()[1], 1}, rgb.type());
  SampleView<CPUBackend> gray_view(gray.raw_mutable_data(), gray.shape(), gray.type());

  Convert(gray_view, TensorLayout("HWC"), DALI_GRAY,
          rgb_view, TensorLayout("HWC"), DALI_RGB,
          {}, {});
  AssertEqual(gray, ReadReferenceFrom(gray_path));
}

TEST_F(ColorConversionTest, GrayToRgb) {
  auto gray = ReadReferenceFrom(gray_path);
  ConstSampleView<CPUBackend> gray_view(gray.raw_mutable_data(), gray.shape(), gray.type());

  Tensor<CPUBackend> rgb;
  rgb.Resize({gray.shape()[0], gray.shape()[1], 3}, gray.type());
  SampleView<CPUBackend> rgb_view(rgb.raw_mutable_data(), rgb.shape(), rgb.type());

  Convert(rgb_view, TensorLayout("HWC"), DALI_RGB,
          gray_view, TensorLayout("HWC"), DALI_GRAY,
          {}, {});

  auto w = rgb.shape()[0], h = rgb.shape()[1];
  auto red   = Crop(rgb, {{0, 0, 0}, {w, h, 1}});
  auto green = Crop(rgb, {{0, 0, 1}, {w, h, 2}});
  auto blue  = Crop(rgb, {{0, 0, 2}, {w, h, 3}});

  AssertEqualSatNorm(red, gray);
  AssertEqualSatNorm(green, gray);
  AssertEqualSatNorm(blue, gray);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
