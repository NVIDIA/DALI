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
#include "dali/imgcodec/decoders/decoder_test_helper.h"
#include "dali/imgcodec/decoders/nvjpeg.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

TEST(NvJpegDecoderTest, Factory) {
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  NvJpegDecoder decoder;
  EXPECT_TRUE(decoder.IsSupported(device_id));
  auto props = decoder.GetProperties();
  EXPECT_TRUE(static_cast<bool>(props.supported_input_kinds & InputKind::HostMemory));;
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::Filename));;
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::DeviceMemory));;
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::Stream));

  ThreadPool tp(4, device_id, false, "nvjpeg decoder test");
  auto instance = decoder.Create(device_id, tp);
  EXPECT_NE(instance, nullptr);
}

std::string from_dali_extra(const std::string& path_relative_to_dali_extra) {
  return make_string_delim('/', testing::dali_extra_path(), path_relative_to_dali_extra);
}

struct ImageBuffer {
  std::vector<uint8_t> buffer;
  ImageSource src;

  explicit ImageBuffer(const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    buffer = {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    src = ImageSource::FromHostMem(buffer.data(), buffer.size());
  }
};

template<typename OutputType>
class NvJpegDecoderTest : public NumpyDecoderTestBase<GPUBackend, OutputType> {
 public:
  explicit NvJpegDecoderTest(int threads_cnt = 1)
  : NumpyDecoderTestBase<GPUBackend, OutputType>(threads_cnt) {}

 protected:
  static const auto dtype = type2id<OutputType>::value;

  std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) override {
    return NvJpegDecoder().Create(this->GetDeviceId(), tp);
  }

  std::shared_ptr<ImageParser> CreateParser() override {
    return std::make_shared<JpegParser>();
  }

  DecodeParams GetParams() {
    DecodeParams opts{};
    opts.dtype = dtype;
    return opts;
  }

  void RunSingleTest() {
    ImageBuffer image(from_dali_extra("db/single/jpeg/134/site-1534685_1280.jpg"));
    auto decoded = this->Decode(&image.src, this->GetParams());
    auto ref = this->ReadReferenceFrom(
      from_dali_extra("db/single/reference/jpeg/site-1534685_1280.npy"));

    this->AssertEqualSatNorm(decoded, ref);
  }
};

using DecodeOutputTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(NvJpegDecoderTest, DecodeOutputTypes);

TYPED_TEST(NvJpegDecoderTest, DecodeSingle) {
  this->RunSingleTest();
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
