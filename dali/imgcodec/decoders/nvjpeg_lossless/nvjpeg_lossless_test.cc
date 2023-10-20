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

#include "dali/imgcodec/decoders/nvjpeg_lossless/nvjpeg_lossless.h"
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include "dali/imgcodec/decoders/decoder_test_helper.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {

std::string data_path(const std::string &relative_path) {
  return make_string_delim('/', testing::dali_extra_path(), "db/single/jpeg_lossless",
                           relative_path);
}
std::string reference_path(const std::string &relative_path) {
  return make_string_delim('/', testing::dali_extra_path(), "db/single/reference/jpeg_lossless",
                           relative_path);
}

}  // namespace

TEST(NvJpegLosslessDecoderTest, Factory) {
  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  NvJpegLosslessDecoderFactory decoder;

  if (!decoder.IsSupported(device_id))
    GTEST_SKIP() << "Need SM60+ and nvJPEG >= 12.2 to execute this test\n";

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

  template <typename T>
  void CompareWithRef(const TensorView<StorageCPU, const T>& data,
                      const std::string& ref_path, int input_precision = -1) {
    auto ref = this->ReadReferenceFrom(ref_path);
    if (input_precision > 0)
      ScaleDynRangeIfNeeded<T>(ref, input_precision);
    AssertSimilar(data, ref);
  }

  bool IsLosslessSupported() {
    return NvJpegLosslessDecoderFactory().IsSupported(this->GetDeviceId());
  }

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

 private:
  template <typename T>
  void ScaleDynRangeIfNeeded(Tensor<CPUBackend> &data, int input_precision) {
    static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value);
    auto *ptr = data.mutable_data<T>();
    int64_t n = data.shape().num_elements();

    int out_bpp = static_cast<int>(sizeof(T) * 8);
    if (out_bpp == input_precision)
      return;

    assert(input_precision < out_bpp);
    constexpr double out_range = std::numeric_limits<T>::max();
    double scale_factor = out_range / ((1 << input_precision) - 1);
    for (int64_t i = 0; i < n; i++)
      ptr[i] = ConvertSat<T>(scale_factor * ptr[i]);
  }
};

class NvJpegLosslessDecoder16bitTest : public NvJpegLosslessDecoderTest<uint16_t> {};
class NvJpegLosslessDecoder8bitTest : public NvJpegLosslessDecoderTest<uint8_t> {};

TEST_F(NvJpegLosslessDecoder16bitTest, DecodeSingle) {
  if (!this->IsLosslessSupported())
    GTEST_SKIP() << "Need SM60+ to execute this test\n";
  ImageBuffer image(data_path("0/cat-3449999_640_grayscale_16bit.jpg"));
  auto decoded = this->Decode(&image.src, this->GetParams(), ROI{});

  auto ref_path = reference_path("cat-3449999_640_grayscale_16bit.npy");
  CompareWithRef(decoded, ref_path);
}

TEST_F(NvJpegLosslessDecoder16bitTest, DecodeSingle12bit) {
  if (!this->IsLosslessSupported())
    GTEST_SKIP() << "Need SM60+ to execute this test\n";
  ImageBuffer image(data_path("0/cat-3449999_640_grayscale_12bit.jpg"));
  auto decoded = this->Decode(&image.src, this->GetParams(), ROI{});

  auto ref_path = reference_path("cat-3449999_640_grayscale_12bit.npy");
  CompareWithRef(decoded, ref_path, 12);
}

TEST_F(NvJpegLosslessDecoder8bitTest, DecodeSingle) {
  if (!this->IsLosslessSupported())
    GTEST_SKIP() << "Need SM60+ to execute this test\n";
  ImageBuffer image(data_path("0/cat-3449999_640_grayscale_8bit.jpg"));
  auto decoded = this->Decode(&image.src, this->GetParams(), ROI{});

  auto ref_path = reference_path("cat-3449999_640_grayscale_8bit.npy");
  CompareWithRef(decoded, ref_path);
}

TEST_F(NvJpegLosslessDecoder16bitTest, DecodeBatch) {
  if (!this->IsLosslessSupported())
    GTEST_SKIP() << "Need SM60+ to execute this test\n";
  std::vector<ImageBuffer> buffers;
  buffers.emplace_back(data_path("0/cat-1245673_640_grayscale_16bit.jpg"));
  buffers.emplace_back(data_path("0/cat-3449999_640_grayscale_16bit.jpg"));
  buffers.emplace_back(data_path("0/cat-3449999_640_grayscale_12bit.jpg"));

  std::vector<std::string> reference;
  reference.push_back(reference_path("cat-1245673_640_grayscale_16bit.npy"));
  reference.push_back(reference_path("cat-3449999_640_grayscale_16bit.npy"));
  reference.push_back(reference_path("cat-3449999_640_grayscale_12bit.npy"));

  std::vector<int> precision = {16, 16, 12};

  std::vector<ImageSource *> sources;
  for (auto &buff : buffers)
    sources.push_back(&buff.src);

  int nsamples = sources.size();
  auto decoded = this->Decode(make_cspan(sources), this->GetParams());
  assert(decoded.size() == nsamples);
  for (int i = 0; i < nsamples; i++) {
    CompareWithRef(decoded[i], reference[i], precision[i]);
  }
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
