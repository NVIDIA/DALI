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
#include "dali/imgcodec/decoders/tiff_libtiff.h"
#include "dali/imgcodec/parsers/tiff.h"
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
auto img_color   = dali_extra + "/db/single/tiff/0/cat-1245673_640.tiff";
}  // namespace

template <typename OutputType>
class LibTiffDecodeTest : public ::testing::Test {
};

using DecodeOutputTypes = ::testing::Types<uint8_t>;  // TODO(skarpinski) Add more types here
TYPED_TEST_SUITE(LibTiffDecodeTest, DecodeOutputTypes);

TYPED_TEST(LibTiffDecodeTest, Decode) {
  using OutputType = TypeParam;

  ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "LibTiff decoder test");
  LibTiffDecoder decoder;
  auto filename = img_color;
  auto source = ImageSource::FromFilename(filename);
  auto instance = decoder.Create(CPU_ONLY_DEVICE_ID, tp);
  ASSERT_NE(instance, nullptr);

  TiffParser parser;
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

  cv::Mat m = cv::imread(filename, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
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

}  // namespace test
}  // namespace imgcodec
}  // namespace dali
