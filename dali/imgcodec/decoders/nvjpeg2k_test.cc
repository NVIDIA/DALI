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
#include "dali/imgcodec/decoders/nvjpeg2k.h"
#include "dali/imgcodec/decoders/decoder_test_helper.h"
#include "dali/imgcodec/parsers/jpeg2000.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/mm/memory.h"
#include "dali/core/convert.h"
#include "dali/util/numpy.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {
template<typename... Args>
std::string join(Args... args) {
  return make_string_delim('/', args...);
}

std::vector<uint8_t> read_file(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

struct ImageBuffer {
  std::vector<uint8_t> buffer;
  ImageSource src;
  explicit ImageBuffer(const std::string &filename)
  : buffer(read_file(filename))
  , src(ImageSource::FromHostMem(buffer.data(), buffer.size())) {}
};

const auto img_dir = join(dali::testing::dali_extra_path(), "db/single/jpeg2k");
const auto ref_dir = join(dali::testing::dali_extra_path(), "db/single/reference/jpeg2k");
const std::vector<std::string> images = {"cat-1245673_640", "cat-300572_640"};
}  // namespace

TEST(NvJpeg2000DecoderTest, Factory) {
  NvJpeg2000Decoder decoder;
  int device_id = 0;
  EXPECT_TRUE(decoder.IsSupported(device_id));
  auto props = decoder.GetProperties();
  EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::HostMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Filename));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::DeviceMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Stream));

  ThreadPool tp(1, CPU_ONLY_DEVICE_ID, false, "nvjpeg2k decoder test");
  auto instance = decoder.Create(device_id, tp);
  EXPECT_NE(instance, nullptr);
}

template<typename OutputType>
class NvJpeg2kDecoderTest : public NumpyDecoderTestBase<GPUBackend, OutputType> {
 public:
  NvJpeg2kDecoderTest() : NumpyDecoderTestBase<GPUBackend, OutputType>(1) {}

 protected:
  static const auto dtype = type2id<OutputType>::value;

  std::shared_ptr<ImageDecoderInstance> CreateDecoder(ThreadPool &tp) override {
    return NvJpeg2000Decoder{}.Create(this->GetDeviceId(), tp);
  }

  std::shared_ptr<ImageParser> CreateParser() override {
    return std::make_shared<Jpeg2000Parser>();
  }

  DecodeParams GetParams() {
    DecodeParams opts{};
    opts.dtype = dtype;
    return opts;
  }
};

using DecodeOutputTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(NvJpeg2kDecoderTest, DecodeOutputTypes);

TYPED_TEST(NvJpeg2kDecoderTest, DecodeSingle) {
  ImageBuffer image(join(img_dir, "0", images[0]) + ".jp2");
  auto decoded = this->Decode(&image.src, this->GetParams());
  auto ref = this->ReadReferenceFrom(join(ref_dir, images[0]) + ".npy");
  this->AssertEqualSatNorm(decoded, ref);
}

TYPED_TEST(NvJpeg2kDecoderTest, DecodeBatch) {
  size_t batch_size = images.size();
  std::vector<ImageBuffer> imgbufs;
  for (size_t i = 0; i < batch_size; i++) {
    const auto filename = join(img_dir, "0", images[i]) + ".jp2";
    imgbufs.emplace_back(filename);
  }
  std::vector<ImageSource *> in(batch_size);
  for (size_t i = 0; i < batch_size; i++)
    in[i] = &imgbufs[i].src;

  auto decoded = this->Decode(make_span(in), this->GetParams());
  for (size_t i = 0; i < batch_size; i++) {
    const auto filename = join(ref_dir, images[i]) + ".npy";
    auto ref = this->ReadReferenceFrom(filename);
    this->AssertEqualSatNorm(decoded[i], ref);
  }
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
