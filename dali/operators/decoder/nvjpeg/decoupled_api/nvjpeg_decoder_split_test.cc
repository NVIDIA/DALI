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

#include <limits>

#include "dali/test/dali_test_decoder.h"

namespace dali {

template <typename ImgType>
class ImageDecoderSplitTest_GPU : public GenericDecoderTest<ImgType> {
 protected:
  OpSpec DecodingOp() const override {
    return OpSpec("ImageDecoder")
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

  void BmpTestDecode(int num_threads) {
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_bmpImgType, 0.7);
  }

  void TiffTestDecode(int num_threads) {
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_tiffImgType , 0.7);
  }

  void SetCustomAllocator() {
    use_chunk_allocator_ = true;
  }

 private:
  unsigned int hybrid_huffman_threshold_ = std::numeric_limits<unsigned int>::max();
  bool use_chunk_allocator_ = false;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(ImageDecoderSplitTest_GPU, Types);


/***********************************************
**** Default JPEG decode (mix host/hybrid) *****
***********************************************/

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode) {
  this->JpegTestDecode(1, 512u*512u);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode2T) {
  this->JpegTestDecode(2, 512u*512u);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode3T) {
  this->JpegTestDecode(3, 512u*512u);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode4T) {
  this->JpegTestDecode(4, 512u*512u);
}

/***********************************************
******* JPEG Decode with chunk allocator *******
***********************************************/

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecodeChunkAlloc) {
  this->SetCustomAllocator();
  this->JpegTestDecode(1, 512u*512u);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecodeChunkAlloc2T) {
  this->SetCustomAllocator();
  this->JpegTestDecode(2, 512u*512u);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecodeChunkAlloc3T) {
  this->SetCustomAllocator();
  this->JpegTestDecode(3, 512u*512u);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecodeChunkAlloc4T) {
  this->SetCustomAllocator();
  this->JpegTestDecode(4, 512u*512u);
}

/***********************************************
******** Host huffman only JPEG decode *********
***********************************************/
// H*W never > threshold so host huffman decoder is always chosen
TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecodeHostHuffman) {
  this->JpegTestDecode(1, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode2THostHuffman) {
  this->JpegTestDecode(2, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode3THostHuffman) {
  this->JpegTestDecode(3, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode4THostHuffman) {
  this->JpegTestDecode(4, std::numeric_limits<unsigned int>::max());
}

/***********************************************
******* Hybrid huffman only JPEG decode ********
***********************************************/
// H*W always > threshold so hybrid huffman decoder is always chosen
TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecodeHybridHuffman) {
  this->JpegTestDecode(1, 0);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode2THybridHuffman) {
  this->JpegTestDecode(2, 0);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode3THybridHuffman) {
  this->JpegTestDecode(3, 0);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleJPEGDecode4THybridHuffman) {
  this->JpegTestDecode(4, 0);
}

/***********************************************
************* BMP fallback decode **************
***********************************************/
TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleBmpDecode) {
  this->BmpTestDecode(1);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleBmpDecode2T) {
  this->BmpTestDecode(2);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleBmpDecode3T) {
  this->BmpTestDecode(3);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleBmpDecode4T) {
  this->BmpTestDecode(4);
}

/***********************************************
************* PNG fallback decode **************
***********************************************/
TYPED_TEST(ImageDecoderSplitTest_GPU, TestSinglePNGDecode) {
  this->PngTestDecode(1);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSinglePNGDecode2T) {
  this->PngTestDecode(2);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSinglePNGDecode3T) {
  this->PngTestDecode(3);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSinglePNGDecode4T) {
  this->PngTestDecode(4);
}


/***********************************************
************ TIFF fallback decode **************
***********************************************/
TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleTiffDecode) {
  this->TiffTestDecode(1);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleTiffDecode2T) {
  this->TiffTestDecode(2);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleTiffDecode3T) {
  this->TiffTestDecode(3);
}

TYPED_TEST(ImageDecoderSplitTest_GPU, TestSingleTiffDecode4T) {
  this->TiffTestDecode(4);
}
}  // namespace dali
