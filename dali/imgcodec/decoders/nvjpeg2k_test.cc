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
#include <utility>
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
const std::vector<std::string> images = {"cat-1245673_640", "cat-2184682_640",
                                         "cat-300572_640",  "cat-3113513_640"};

auto gen_batch_input() {
  std::vector<std::string> image_names, ref_names;
  for (size_t i = 0; i < images.size(); i++) {
    image_names.push_back(join(img_dir, "0", images[i]) + ".jp2");
    ref_names.push_back(join(ref_dir, images[i]) + ".npy");
  }
  return std::make_pair(image_names, ref_names);
}
}  // namespace

TEST(NvJpeg2000DecoderTest, Factory) {
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  NvJpeg2000Decoder decoder;
  EXPECT_TRUE(decoder.IsSupported(device_id));
  auto props = decoder.GetProperties();
  EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::HostMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Filename));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::DeviceMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Stream));

  ThreadPool tp(4, device_id, false, "nvjpeg2k decoder test");
  auto instance = decoder.Create(device_id, tp);
  EXPECT_NE(instance, nullptr);
}

template<typename OutputType>
class NvJpeg2000DecoderTest : public NumpyDecoderTestBase<GPUBackend, OutputType> {
 public:
  explicit NvJpeg2000DecoderTest(int threads_cnt = 1)
  : NumpyDecoderTestBase<GPUBackend, OutputType>(threads_cnt) {}

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

  void RunSingleTest(std::string image_name, std::string ref_name) {
    ImageBuffer image(join(img_dir, "0", images[0]) + ".jp2");
    auto decoded = this->Decode(&image.src, this->GetParams());
    auto ref = this->ReadReferenceFrom(join(ref_dir, images[0]) + ".npy");
    this->AssertEqualSatNorm(decoded, ref);
  }

  void RunBatchTest(std::vector<std::string> image_names, std::vector<std::string> ref_names) {
    assert(image_names.size() == ref_names.size());
    size_t batch_size = image_names.size();
    std::vector<ImageBuffer> imgbufs;
    for (size_t i = 0; i < batch_size; i++)
      imgbufs.emplace_back(image_names[i]);
    std::vector<ImageSource *> in(batch_size);
    for (size_t i = 0; i < batch_size; i++)
      in[i] = &imgbufs[i].src;

    auto decoded = this->Decode(make_span(in), this->GetParams());
    for (size_t i = 0; i < batch_size; i++) {
      auto ref = this->ReadReferenceFrom(ref_names[i]);
      this->AssertEqualSatNorm(decoded[i], ref);
    }
  }
};

using DecodeOutputTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(NvJpeg2000DecoderTest, DecodeOutputTypes);

TYPED_TEST(NvJpeg2000DecoderTest, DecodeSingle) {
  const auto image_name = join(img_dir, "0", images[0]) + ".jp2";
  const auto ref_name = join(ref_dir, images[0]) + ".npy";
  this->RunSingleTest(image_name, ref_name);
}

TYPED_TEST(NvJpeg2000DecoderTest, DecodeBatchSingleThread) {
  auto input = gen_batch_input();
  this->RunBatchTest(input.first, input.second);
}

template<class OutputType>
struct NvJpeg2000DecoderTestMultithreaded : NvJpeg2000DecoderTest<OutputType> {
  NvJpeg2000DecoderTestMultithreaded() : NvJpeg2000DecoderTest<OutputType>(2) {}
};

TYPED_TEST_SUITE(NvJpeg2000DecoderTestMultithreaded, DecodeOutputTypes);

TYPED_TEST(NvJpeg2000DecoderTestMultithreaded, DecodeBatch) {
  auto input = gen_batch_input();
  this->RunBatchTest(input.first, input.second);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
