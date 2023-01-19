// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/imgcodec/decoders/decoder_test_helper.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

TEST(NvJpegDecoderTest, Factory) {
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  NvJpegDecoderFactory decoder;
  EXPECT_TRUE(decoder.IsSupported(device_id));
  auto props = decoder.GetProperties();
  EXPECT_TRUE(static_cast<bool>(props.supported_input_kinds & InputKind::HostMemory));
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::Filename));
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::DeviceMemory));
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::Stream));

  auto instance = decoder.Create(device_id, {{"num_threads", 4}});
  EXPECT_NE(instance, nullptr);
}

std::string from_dali_extra(const std::string& path_relative_to_dali_extra) {
  return make_string_delim('/', testing::dali_extra_path(), path_relative_to_dali_extra);
}

std::vector<std::string> orientation_files = {
  "db/imgcodec/jpeg/orientation/padlock-406986_640_horizontal",
  "db/imgcodec/jpeg/orientation/padlock-406986_640_mirror_horizontal",
  "db/imgcodec/jpeg/orientation/padlock-406986_640_mirror_horizontal_rotate_270",
  "db/imgcodec/jpeg/orientation/padlock-406986_640_mirror_horizontal_rotate_90",
  "db/imgcodec/jpeg/orientation/padlock-406986_640_mirror_vertical",
  "db/imgcodec/jpeg/orientation/padlock-406986_640_no_exif",
  "db/imgcodec/jpeg/orientation/padlock-406986_640_no_orientation",
  "db/imgcodec/jpeg/orientation/padlock-406986_640_rotate_270",
  "db/imgcodec/jpeg/orientation/padlock-406986_640_rotate_90",
};

template<typename OutputType>
class NvJpegDecoderTest : public NumpyDecoderTestBase<GPUBackend, OutputType> {
 public:
  explicit NvJpegDecoderTest(int threads_cnt = 1)
  : NumpyDecoderTestBase<GPUBackend, OutputType>(threads_cnt) {}

 protected:
  static const auto dtype = type2id<OutputType>::value;

  std::shared_ptr<ImageDecoderInstance> CreateDecoder() override {
    return NvJpegDecoderFactory().Create(this->GetDeviceId(), {{"num_threads", 4}});
  }

  std::shared_ptr<ImageParser> CreateParser() override {
    return std::make_shared<JpegParser>();
  }

  DecodeParams GetParams() {
    DecodeParams opts{};
    opts.dtype = dtype;
    return opts;
  }

  void RunSingleTest(const ROI& roi = {}) {
    ImageBuffer image(from_dali_extra("db/single/jpeg/134/site-1534685_1280.jpg"));
    auto decoded = this->Decode(&image.src, this->GetParams(), roi);
    auto ref = this->ReadReferenceFrom(
      from_dali_extra("db/single/reference/jpeg/site-1534685_1280.npy"));

    if (roi.use_roi()) {
      AssertSimilar(decoded, Crop(ref, roi));
    } else {
      AssertSimilar(decoded, ref);
    }
  }

  void RunSingleYCbCrTest() {
    ImageBuffer image(from_dali_extra("db/single/jpeg/134/site-1534685_1280.jpg"));

    auto params = this->GetParams();
    params.format = DALI_YCbCr;

    auto decoded = this->Decode(&image.src, params);
    auto ref = this->ReadReferenceFrom(
      from_dali_extra("db/single/reference/jpeg/site-1534685_1280_ycbcr.npy"));

    AssertSimilar(decoded, ref);
  }

  void RunOrientationTest(ROI roi = {}) {
    std::vector<ImageBuffer> buffers;
    for (const auto &name : orientation_files)
      buffers.emplace_back(from_dali_extra(name + ".jpg"));

    std::vector<ImageSource*> sources;
    for (auto &buff : buffers)
      sources.push_back(&buff.src);

    const size_t batch_size = orientation_files.size();
    std::vector<ROI> rois(batch_size, roi);

    auto decoded = this->Decode(make_cspan(sources), this->GetParams(), make_span(rois));
    assert(decoded.size() == static_cast<int>(sources.size()));
    for (int i = 0; i < decoded.size(); i++) {
      auto ref = this->ReadReferenceFrom(from_dali_extra(orientation_files[i] + ".npy"));
      if (roi)
        AssertSimilar(decoded[i], Crop(ref, roi));
      else
        AssertSimilar(decoded[i], ref);
    }
  }
};

using DecodeOutputTypes = ::testing::Types<uint8_t, int16_t, float>;
TYPED_TEST_SUITE(NvJpegDecoderTest, DecodeOutputTypes);

TYPED_TEST(NvJpegDecoderTest, DecodeSingle) {
  this->RunSingleTest();
}

TYPED_TEST(NvJpegDecoderTest, DecodeSingleYCbCr) {
  this->RunSingleYCbCrTest();
}

TYPED_TEST(NvJpegDecoderTest, DecodeSingleRoi) {
  this->RunSingleTest({{12, 34}, {340, 450}});
}

TYPED_TEST(NvJpegDecoderTest, DecodeOrientationBatched) {
  this->RunOrientationTest();
}

TYPED_TEST(NvJpegDecoderTest, DecodeOrientationWithRoiBatched) {
  this->RunOrientationTest({{1, 2}, {21, 37}});
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
