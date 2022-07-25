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
#include "dali/imgcodec/decoders/libjpeg_turbo.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/test/dali_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/util/thread_pool.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/mm/memory.h"
#include "dali/core/convert.h"

namespace dali {
namespace imgcodec {
namespace test {

namespace {
const auto &dali_extra = dali::testing::dali_extra_path();
const auto imgname = dali_extra + "/db/single/jpeg/312/grasshopper-4357907_1280.jpg";

std::vector<uint8_t> read_file(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST(LibJpegTurboTest, Factory) {
  LibJpegTurboDecoder decoder;
  EXPECT_TRUE(decoder.IsSupported(CPU_ONLY_DEVICE_ID));
  auto props = decoder.GetProperties();
  EXPECT_TRUE(!!(props.supported_input_kinds & InputKind::HostMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Filename));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::DeviceMemory));;
  EXPECT_FALSE(!!(props.supported_input_kinds & InputKind::Stream));

  ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "libjpeg-turbo decoder test");
  auto instance = decoder.Create(CPU_ONLY_DEVICE_ID, tp);
  EXPECT_NE(instance, nullptr);
}

template <typename OutputType>
class LibJpegTurboDecodeTest : public ::testing::Test {
};

using DecodeOutputTypes = ::testing::Types<uint8_t, int16_t, float>;
TYPED_TEST_SUITE(LibJpegTurboDecodeTest, DecodeOutputTypes);

TYPED_TEST(LibJpegTurboDecodeTest, Decode) {
  using OutputType = TypeParam;

  ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "libjpeg-turbo decoder test");
  LibJpegTurboDecoder decoder;
  const auto file_content = read_file(imgname);
  std::cerr << "file_content.size(): " << file_content.size() << "\n";
  auto source = ImageSource::FromHostMem(file_content.data(), file_content.size());
  auto instance = decoder.Create(CPU_ONLY_DEVICE_ID, tp);
  ASSERT_NE(instance, nullptr);

  JpegParser parser;
  EXPECT_TRUE(parser.CanParse(&source));
  ImageInfo info = parser.GetInfo(&source);
  DecodeParams params;
  int64_t n = volume(info.shape);
  ASSERT_GE(n, 0);
  ASSERT_LE(n, 100000000);  // sanity check - less than 100M elements
  auto mem = mm::alloc_raw_unique<OutputType, mm::memory_kind::host>(n);
  SampleView<CPUBackend> sv(mem.get(), info.shape, type2id<OutputType>::value);
  auto result = instance->Decode(sv, &source, params);
  if (result.exception) {
    EXPECT_NO_THROW(std::rethrow_exception(result.exception));
  }
  ASSERT_TRUE(result.success);

  cv::Mat m = cv::imread(imgname, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
  cv::cvtColor(m, m, cv::COLOR_BGR2RGB);
  int64_t out_row_stride = info.shape[1] * info.shape[2];
  for (int y = 0; y < m.rows; y++) {
    const OutputType *out_row = sv.data<OutputType>() + y * out_row_stride;
    const uint8_t *ref_row = m.ptr<uint8_t>(y);
    for (int x = 0; x < m.cols; x++) {
      for (int c = 0; c < 3; c++) {
        ASSERT_EQ(out_row[3*x + c], ConvertSatNorm<OutputType>(ref_row[3*x + c]))
          << " at " << x << ", " << y << ", " << c;
      }
    }
  }
}

TYPED_TEST(LibJpegTurboDecodeTest, DecodeROI) {
  using OutputType = TypeParam;

  ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "libjpeg-turbo decoder test");
  LibJpegTurboDecoder decoder;
  const auto file_content = read_file(imgname);
  auto source = ImageSource::FromHostMem(file_content.data(), file_content.size());
  auto instance = decoder.Create(CPU_ONLY_DEVICE_ID, tp);
  ASSERT_NE(instance, nullptr);

  JpegParser parser;
  EXPECT_TRUE(parser.CanParse(&source));
  ImageInfo info = parser.GetInfo(&source);

  DecodeParams params;
  params.use_roi = true;
  params.roi.begin = { 13, 17 };
  params.roi.end = { info.shape[0] - 23, info.shape[1] - 19 };
  TensorShape<> out_shape;
  out_shape = shape_cat(params.roi.shape(), 3);
  int64_t n = volume(out_shape);
  ASSERT_GE(n, 0);
  ASSERT_LE(n, 100000000);  // sanity check - less than 100M elements
  auto mem = mm::alloc_raw_unique<OutputType, mm::memory_kind::host>(n);
  SampleView<CPUBackend> sv(mem.get(), out_shape, type2id<OutputType>::value);
  auto result = instance->Decode(sv, &source, params);
  if (result.exception) {
    EXPECT_NO_THROW(std::rethrow_exception(result.exception));
  }
  ASSERT_TRUE(result.success);

  cv::Mat m = cv::imread(imgname, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
  cv::Rect roi_rect(cv::Point2i(params.roi.begin[1], params.roi.begin[0]),
                    cv::Point2i(params.roi.end[1],   params.roi.end[0]));
  m = m(roi_rect);
  cv::cvtColor(m, m, cv::COLOR_BGR2RGB);
  int64_t out_row_stride = out_shape[1] * out_shape[2];
  for (int y = 0; y < m.rows; y++) {
    const OutputType *out_row = sv.data<OutputType>() + y * out_row_stride;
    const uint8_t *ref_row = m.ptr<uint8_t>(y);
    for (int x = 0; x < m.cols; x++) {
      for (int c = 0; c < 3; c++) {
        ASSERT_EQ(out_row[3*x + c], ConvertSatNorm<OutputType>(ref_row[3*x + c]))
          << " at " << x << ", " << y << ", " << c;
      }
    }
  }
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
