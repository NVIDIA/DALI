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
class nvjpegDecodeNewTest : public GenericDecoderTest<ImgType> {
 protected:
  const OpSpec DecodingOp() const override {
    return OpSpec("nvJPEGDecoderNew")
      .AddArg("device", "mixed")
      .AddArg("output_type", this->img_type_)
      .AddArg("hybrid_huffman_threshold", hybrid_huffman_threshold_)
      .AddInput("encoded", "cpu")
      .AddOutput("decoded", "gpu");
  }

  void JpegTestDecode(int num_threads, unsigned int hybrid_huffman_threshold = 450*450) {
    hybrid_huffman_threshold_ = hybrid_huffman_threshold;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_jpegImgType, 0.7);
  }

  void PngTestDecode(int num_threads) {
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_pngImgType, 0.7);
  }

  void TiffTestDecode(int num_threads) {
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_tiffImgType , 0.7);
  }

 private:
  unsigned int hybrid_huffman_threshold_ = 1000*1000;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(nvjpegDecodeNewTest, Types);


/***********************************************
**** Default JPEG decode (mix host/hybrid) *****
***********************************************/

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode) {
  this->JpegTestDecode(1);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode2T) {
  this->JpegTestDecode(2);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode3T) {
  this->JpegTestDecode(3);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode4T) {
  this->JpegTestDecode(4);
}

/***********************************************
******** Host huffman only JPEG decode *********
***********************************************/
// H*W never > threshold so host huffman decoder is always chosen
TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecodeHostHuffman) {
  this->JpegTestDecode(1, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode2THostHuffman) {
  this->JpegTestDecode(2, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode3THostHuffman) {
  this->JpegTestDecode(3, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode4THostHuffman) {
  this->JpegTestDecode(4, std::numeric_limits<unsigned int>::max());
}

/***********************************************
******* Hybrid huffman only JPEG decode ********
***********************************************/
// H*W always > threshold so hybrid huffman decoder is always chosen
TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecodeHybridHuffman) {
  this->JpegTestDecode(1, 0);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode2THybridHuffman) {
  this->JpegTestDecode(2, 0);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode3THybridHuffman) {
  this->JpegTestDecode(3, 0);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleJPEGDecode4THybridHuffman) {
  this->JpegTestDecode(4, 0);
}

/***********************************************
************* PNG fallback decode **************
***********************************************/
TYPED_TEST(nvjpegDecodeNewTest, TestSinglePNGDecode) {
  this->PngTestDecode(1);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSinglePNGDecode2T) {
  this->PngTestDecode(2);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSinglePNGDecode3T) {
  this->PngTestDecode(3);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSinglePNGDecode4T) {
  this->PngTestDecode(4);
}

/***********************************************
************ TIFF fallback decode **************
***********************************************/
TYPED_TEST(nvjpegDecodeNewTest, TestSingleTiffDecode) {
  this->TiffTestDecode(1);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleTiffDecode2T) {
  this->TiffTestDecode(2);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleTiffDecode3T) {
  this->TiffTestDecode(3);
}

TYPED_TEST(nvjpegDecodeNewTest, TestSingleTiffDecode4T) {
  this->TiffTestDecode(4);
}

}  // namespace dali
