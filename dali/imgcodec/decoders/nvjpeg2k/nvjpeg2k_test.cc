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
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/imgcodec/decoders/nvjpeg2k/nvjpeg2k.h"
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
  assert(stream.is_open());
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
const auto imgcodec_dir = join(dali::testing::dali_extra_path(), "db/imgcodec/jpeg2k");

const std::vector<std::string> images = {
  "0/cat-1245673_640",
  "0/cat-2184682_640",
  "0/cat-300572_640",
  "0/cat-3113513_640",
};

const std::vector<std::pair<std::string, ROI>> roi_images = {
  {"0/cat-1245673_640", {{17, 33}, {276, 489}}},
  {"2/tiled-cat-1046544_640", {{178, 220}, {456, 290}}},
  {"2/tiled-cat-111793_640", {{9, 317}, {58, 325}}},
  {"2/tiled-cat-3113513_640", {{2, 1}, {200, 600}}},
};

const char bitdepth_converted_imgname[] = "0/cat-1245673_640";

struct ImageTestingData {
  std::string img_path;
  std::string ref_path;
  ROI roi;
};

ImageTestingData from_regular_file(std::string filename) {
  return {join(img_dir, filename) + ".jp2", join(ref_dir, filename) + ".npy", {}};
}

ImageTestingData from_regular_file(std::string filename, ROI roi) {
  return {join(img_dir, filename) + ".jp2", join(ref_dir, filename) + "_roi.npy", roi};
}

ImageTestingData from_custom_bitdepth_file(std::string filename, int bpp) {
  auto imgpath = make_string(join(imgcodec_dir, filename), "-", bpp, "bit.jp2");
  auto refpath = make_string(join(ref_dir, filename), ".npy");
  return {imgpath, refpath, {}};
}

}  // namespace

TEST(NvJpeg2000DecoderTest, Factory) {
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  NvJpeg2000DecoderFactory decoder;
  EXPECT_TRUE(decoder.IsSupported(device_id));
  auto props = decoder.GetProperties();
  EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::HostMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Filename));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::DeviceMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Stream));

  auto instance = decoder.Create(device_id);
  EXPECT_NE(instance, nullptr);
}

template<typename OutputType>
class NvJpeg2000DecoderTest : public NumpyDecoderTestBase<GPUBackend, OutputType> {
 public:
  explicit NvJpeg2000DecoderTest(int threads_cnt = 1)
  : NumpyDecoderTestBase<GPUBackend, OutputType>(threads_cnt) {}

 protected:
  static const auto dtype = type2id<OutputType>::value;
  using Type = OutputType;

  std::shared_ptr<ImageDecoderInstance> CreateDecoder() override {
    return NvJpeg2000DecoderFactory().Create(this->GetDeviceId());
  }

  std::shared_ptr<ImageParser> CreateParser() override {
    return std::make_shared<Jpeg2000Parser>();
  }

  DecodeParams GetParams() {
    DecodeParams opts{};
    opts.dtype = dtype;
    return opts;
  }

  template<class T, class U>
  void AssertEqual(const T &img, const U &ref, std::optional<float> eps) {
    if (eps)
      AssertClose(img, ref, eps.value());
    else
      AssertEqualSatNorm(img, ref);
  }

  void RunTest(const ImageTestingData &data, std::optional<float> eps = std::nullopt) {
    ImageBuffer image(data.img_path);
    auto decoded = this->Decode(&image.src, this->GetParams(), data.roi);
    auto ref = this->ReadReferenceFrom(data.ref_path);
    AssertEqual(decoded, ref, eps);
  }

  void RunTest(const std::vector<ImageTestingData> &data,
               std::optional<float> eps = std::nullopt) {
    size_t batch_size = images.size();
    std::vector<ImageBuffer> imgbufs;
    for (size_t i = 0; i < batch_size; i++)
      imgbufs.emplace_back(data[i].img_path);

    std::vector<ImageSource *> in(batch_size);
    std::vector<ROI> rois(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      in[i] = &imgbufs[i].src;
      rois[i] = data[i].roi;
    }

    auto decoded = this->Decode(make_span(in), this->GetParams(), make_span(rois));
    for (size_t i = 0; i < batch_size; i++) {
      auto ref = this->ReadReferenceFrom(data[i].ref_path);
      AssertEqual(decoded[i], ref, eps);
    }
  }
};

using DecodeOutputTypes = ::testing::Types<uint8_t, int16_t, float>;
TYPED_TEST_SUITE(NvJpeg2000DecoderTest, DecodeOutputTypes);

TYPED_TEST(NvJpeg2000DecoderTest, DecodeSingle) {
  for (const auto &name : images)
    this->RunTest(from_regular_file(name));
}

TYPED_TEST(NvJpeg2000DecoderTest, DecodeSingleRoi) {
  for (const auto &[name, roi] : roi_images)
    this->RunTest(from_regular_file(name, roi));
}

TYPED_TEST(NvJpeg2000DecoderTest, DecodeBatchSingleThread) {
  std::vector<ImageTestingData> data;
  for (const auto &name : images)
    data.push_back(from_regular_file(name));
  this->RunTest(data);
}

TYPED_TEST(NvJpeg2000DecoderTest, 5BitImage) {
  using OutputType = typename TestFixture::Type;
  float eps;
  if (std::is_floating_point_v<OutputType>)
    eps = 0.07;
  else if (std::is_same_v<OutputType, uint8_t>)
    eps = 16.05;
  else if (std::is_same_v<OutputType, int16_t>)
    eps = 256 * 16.05;
  else
    assert(false);

  this->RunTest(from_custom_bitdepth_file(bitdepth_converted_imgname, 5), eps);
}

TYPED_TEST(NvJpeg2000DecoderTest, 12BitImage) {
  using OutputType = typename TestFixture::Type;
  float eps;
  if (std::is_floating_point_v<OutputType>)
    eps = 0.01;
  else if (std::is_same_v<OutputType, uint8_t>)
    eps = 1.05;
  else if (std::is_same_v<OutputType, int16_t>)
    eps = 128.05;
  else
    assert(false);

  this->RunTest(from_custom_bitdepth_file(bitdepth_converted_imgname, 12), eps);
}

template<class OutputType>
struct NvJpeg2000DecoderTestMultithreaded : NvJpeg2000DecoderTest<OutputType> {
  NvJpeg2000DecoderTestMultithreaded() : NvJpeg2000DecoderTest<OutputType>(2) {}
};

TYPED_TEST_SUITE(NvJpeg2000DecoderTestMultithreaded, DecodeOutputTypes);

TYPED_TEST(NvJpeg2000DecoderTestMultithreaded, DecodeBatch) {
  constexpr int copies_cnt = 4;
  std::vector<ImageTestingData> data;
  for (int i = 0; i < copies_cnt; i++)
    for (const auto &name : images)
      data.push_back(from_regular_file(name));
  this->RunTest(data);
}

TYPED_TEST(NvJpeg2000DecoderTestMultithreaded, DecodeBatchRoi) {
  constexpr int copies_cnt = 4;
  std::vector<ImageTestingData> data;
  for (int i = 0; i < copies_cnt; i++)
    for (const auto &[name, roi] : roi_images)
      data.push_back(from_regular_file(name, roi));
  this->RunTest(data);
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
