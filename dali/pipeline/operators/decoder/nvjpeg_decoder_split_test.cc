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
class nvjpegDecodeSplitTest : public GenericDecoderTest<ImgType> {
 protected:
  OpSpec DecodingOp() const override {
    return OpSpec("nvJPEGDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", this->img_type_)
      .AddArg("hybrid_huffman_threshold", hybrid_huffman_threshold_)
      .AddArg("use_chunk_allocator", use_chunk_allocator_)
      .AddArg("split_stages", true)
      .AddInput("encoded", "cpu")
      .AddOutput("decoded", "gpu");
  }

  void JpegTestDecode(int num_threads, unsigned int hybrid_huffman_threshold) {
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

  void SetCustomAllocator() {
    use_chunk_allocator_ = true;
  }

 private:
  unsigned int hybrid_huffman_threshold_;
  bool use_chunk_allocator_ = false;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(nvjpegDecodeSplitTest, Types);


/***********************************************
**** Default JPEG decode (mix host/hybrid) *****
***********************************************/

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode) {
  this->JpegTestDecode(1, 512u*512u);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode2T) {
  this->JpegTestDecode(2, 512u*512u);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode3T) {
  this->JpegTestDecode(3, 512u*512u);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode4T) {
  this->JpegTestDecode(4, 512u*512u);
}

/***********************************************
******* JPEG Decode with chunk allocator *******
***********************************************/

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecodeChunkAlloc) {
  this->SetCustomAllocator();
  this->JpegTestDecode(1, 512u*512u);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecodeChunkAlloc2T) {
  this->SetCustomAllocator();
  this->JpegTestDecode(2, 512u*512u);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecodeChunkAlloc3T) {
  this->SetCustomAllocator();
  this->JpegTestDecode(3, 512u*512u);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecodeChunkAlloc4T) {
  this->SetCustomAllocator();
  this->JpegTestDecode(4, 512u*512u);
}

/***********************************************
******** Host huffman only JPEG decode *********
***********************************************/
// H*W never > threshold so host huffman decoder is always chosen
TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecodeHostHuffman) {
  this->JpegTestDecode(1, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode2THostHuffman) {
  this->JpegTestDecode(2, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode3THostHuffman) {
  this->JpegTestDecode(3, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode4THostHuffman) {
  this->JpegTestDecode(4, std::numeric_limits<unsigned int>::max());
}

/***********************************************
******* Hybrid huffman only JPEG decode ********
***********************************************/
// H*W always > threshold so hybrid huffman decoder is always chosen
TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecodeHybridHuffman) {
  this->JpegTestDecode(1, 0);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode2THybridHuffman) {
  this->JpegTestDecode(2, 0);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode3THybridHuffman) {
  this->JpegTestDecode(3, 0);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleJPEGDecode4THybridHuffman) {
  this->JpegTestDecode(4, 0);
}

/***********************************************
************* PNG fallback decode **************
***********************************************/
TYPED_TEST(nvjpegDecodeSplitTest, TestSinglePNGDecode) {
  this->PngTestDecode(1);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSinglePNGDecode2T) {
  this->PngTestDecode(2);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSinglePNGDecode3T) {
  this->PngTestDecode(3);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSinglePNGDecode4T) {
  this->PngTestDecode(4);
}

/***********************************************
************ TIFF fallback decode **************
***********************************************/
TYPED_TEST(nvjpegDecodeSplitTest, TestSingleTiffDecode) {
  this->TiffTestDecode(1);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleTiffDecode2T) {
  this->TiffTestDecode(2);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleTiffDecode3T) {
  this->TiffTestDecode(3);
}

TYPED_TEST(nvjpegDecodeSplitTest, TestSingleTiffDecode4T) {
  this->TiffTestDecode(4);
}
}  // namespace dali
