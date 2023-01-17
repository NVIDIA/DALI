// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_lossless.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {

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

}  // namespace

TEST(NvJpegLosslessDecoderTest, Factory) {
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  NvJpegLosslessDecoderFactory decoder;
  EXPECT_TRUE(decoder.IsSupported(device_id));
  auto props = decoder.GetProperties();
  EXPECT_TRUE(static_cast<bool>(props.supported_input_kinds & InputKind::HostMemory));
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::Filename));
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::DeviceMemory));
  EXPECT_FALSE(static_cast<bool>(props.supported_input_kinds & InputKind::Stream));

  auto instance = decoder.Create(device_id, {{"num_threads", 4}});
  EXPECT_NE(instance, nullptr);
}

template<typename OutputType>
class NvJpegLosslessDecoderTest : public NumpyDecoderTestBase<GPUBackend, OutputType> {
 public:
  explicit NvJpegLosslessDecoderTest(int threads_cnt = 1)
  : NumpyDecoderTestBase<GPUBackend, OutputType>(threads_cnt) {}

 protected:
  static const auto dtype = type2id<OutputType>::value;

  std::shared_ptr<ImageDecoderInstance> CreateDecoder() override {
    return NvJpegLosslessDecoderFactory().Create(this->GetDeviceId(), {{"num_threads", 4}});
  }

  std::shared_ptr<ImageParser> CreateParser() override {
    return std::make_shared<JpegParser>();
  }

  DecodeParams GetParams() {
    DecodeParams opts{};
    opts.dtype = dtype;
    opts.format = DALI_ANY_DATA;
    opts.planar = false;
    opts.use_orientation = true;
    return opts;
  }
};

class NvJpegLosslessDecoder16bitTest : public NvJpegLosslessDecoderTest<uint16_t> {};
class NvJpegLosslessDecoder8bitTest : public NvJpegLosslessDecoderTest<uint8_t> {};

TEST_F(NvJpegLosslessDecoder16bitTest, DecodeSingle) {
  ImageBuffer image(
      from_dali_extra("db/single/jpeg_lossless/0/cat-3449999_640_grayscale_16bit.jpg"));
  auto decoded = this->Decode(&image.src, this->GetParams(), ROI{});

  auto ref = this->ReadReferenceFrom(
      from_dali_extra("db/single/reference/jpeg_lossless/cat-3449999_640_grayscale_16bit.npy"));
  AssertSimilar(decoded, ref);
}

TEST_F(NvJpegLosslessDecoder8bitTest, DecodeSingle) {
  ImageBuffer image(
      from_dali_extra("db/single/jpeg_lossless/0/cat-3449999_640_grayscale_8bit.jpg"));
  auto decoded = this->Decode(&image.src, this->GetParams(), ROI{});

  auto ref = this->ReadReferenceFrom(
      from_dali_extra("db/single/reference/jpeg_lossless/cat-3449999_640_grayscale_8bit.npy"));
  AssertSimilar(decoded, ref);
}

TEST_F(NvJpegLosslessDecoder16bitTest, DecodeBatch) {
  std::vector<ImageBuffer> buffers;
  buffers.emplace_back(
      from_dali_extra("db/single/jpeg_lossless/0/cat-1245673_640_grayscale_16bit.jpg"));
  buffers.emplace_back(
      from_dali_extra("db/single/jpeg_lossless/0/cat-3449999_640_grayscale_16bit.jpg"));
  buffers.emplace_back(
      from_dali_extra("db/single/jpeg_lossless/0/cat-3449999_640_grayscale_12bit.jpg"));

  std::vector<Tensor<CPUBackend>> reference;
  reference.push_back(this->ReadReferenceFrom(
      from_dali_extra("db/single/reference/jpeg_lossless/cat-1245673_640_grayscale_16bit.npy")));
  reference.push_back(this->ReadReferenceFrom(
      from_dali_extra("db/single/reference/jpeg_lossless/cat-3449999_640_grayscale_16bit.npy")));
  reference.push_back(this->ReadReferenceFrom(
      from_dali_extra("db/single/reference/jpeg_lossless/cat-3449999_640_grayscale_12bit.npy")));

  std::vector<ImageSource *> sources;
  for (auto &buff : buffers)
    sources.push_back(&buff.src);

  int nsamples = sources.size();
  auto decoded = this->Decode(make_cspan(sources), this->GetParams());
  assert(decoded.size() == nsamples);
  for (int i = 0; i < nsamples; i++) {
    AssertSimilar(decoded[i], reference[i]);
  }
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
