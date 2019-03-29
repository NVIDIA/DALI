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
  OpSpec DecodingOp() const override {
    return OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", this->img_type_)
      .AddArg("use_batched_decode", batched_)
      .AddInput("encoded", "cpu")
      .AddOutput("decoded", "gpu");
  }

  void JpegTestDecode(bool batched, int num_threads) {
    batched_ = batched;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_jpegImgType);
  }

  void PngTestDecode(bool batched, int num_threads) {
    batched_ = batched;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_pngImgType);
  }

  void TiffTestDecode(bool batched, int num_threads) {
    batched_ = batched;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_tiffImgType);
  }

 private:
  bool batched_ = false;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(nvjpegDecodeTest, Types);

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode) {
  this->JpegTestDecode(false, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode2T) {
  this->JpegTestDecode(false, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode3T) {
  this->JpegTestDecode(false, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleJPEGDecode4T) {
  this->JpegTestDecode(false, 4);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode) {
  this->JpegTestDecode(true, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode2T) {
  this->JpegTestDecode(true, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode3T) {
  this->JpegTestDecode(true, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedJPEGDecode4T) {
  this->JpegTestDecode(true, 4);
}

TYPED_TEST(nvjpegDecodeTest, TestSinglePNGDecode) {
  this->PngTestDecode(false, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestSinglePNGDecode2T) {
  this->PngTestDecode(false, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestSinglePNGDecode3T) {
  this->PngTestDecode(false, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestSinglePNGDecode4T) {
  this->PngTestDecode(false, 4);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedPNGDecode) {
  this->PngTestDecode(true, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedPNGDecode2T) {
  this->PngTestDecode(true, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedPNGDecode3T) {
  this->PngTestDecode(true, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedPNGDecode4T) {
  this->PngTestDecode(true, 4);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleTiffDecode) {
  this->TiffTestDecode(false, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleTiffDecode2T) {
  this->TiffTestDecode(false, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleTiffDecode3T) {
  this->TiffTestDecode(false, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestSingleTiffDecode4T) {
  this->TiffTestDecode(false, 4);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedTiffDecode) {
  this->TiffTestDecode(true, 1);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedTiffDecode2T) {
  this->TiffTestDecode(true, 2);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedTiffDecode3T) {
  this->TiffTestDecode(true, 3);
}

TYPED_TEST(nvjpegDecodeTest, TestBatchedTiffDecode4T) {
  this->TiffTestDecode(true, 4);
}

}  // namespace dali
