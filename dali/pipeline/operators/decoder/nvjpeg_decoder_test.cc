// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dali_test_decoder.h"

namespace dali {

template <typename ImgType>
class nvjpegDecodeTest : public GenericDecoderTest<ImgType> {
 protected:
  const OpSpec DecodingOp() const override {
    return OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", this->img_type_)
      .AddArg("use_batched_decode", batched_)
      .AddInput("encoded", "cpu")
      .AddOutput("decoded", "gpu");
  }

 protected:
  void TestDecode(bool batched, int num_threads) {
    batched_ = batched;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_jpegImgType, 2.0);
  }

 private:
  bool batched_ = false;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(nvjpegDecodeTest, Types);

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode) {
  this->TestDecode(false, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode2T) {
  this->TestDecode(false, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode3T) {
  this->TestDecode(false, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode4T) {
  this->TestDecode(false, 4);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode) {
  this->TestDecode(true, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode2T) {
  this->TestDecode(true, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode3T) {
  this->TestDecode(true, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode4T) {
  this->TestDecode(true, 4);
}

}  // namespace dali

