// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <nvjpeg.h>  // for NVJPEG_VER_MAJOR define
#include <limits>

#include "dali/test/dali_test_decoder.h"
#include "dali/util/nvml.h"

namespace dali {

template <typename ImgType>
class nvjpegDecodeDecoupledAPITest : public GenericDecoderTest<ImgType> {
 protected:
  OpSpec DecodingOp() const override {
    return OpSpec("ImageDecoder")
      .AddArg("device", "mixed")
      .AddArg("output_type", this->img_type_)
      .AddArg("hybrid_huffman_threshold", hybrid_huffman_threshold_)
      .AddInput("encoded", "cpu")
      .AddOutput("decoded", "gpu");
  }

  void JpegTestDecode(int num_threads, unsigned int hybrid_huffman_threshold) {
    hybrid_huffman_threshold_ = hybrid_huffman_threshold;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_jpegImgType);
  }

  void PngTestDecode(int num_threads) {
    hybrid_huffman_threshold_ = 0;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_pngImgType);
  }

  void BmpTestDecode(int num_threads) {
    hybrid_huffman_threshold_ = 0;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_bmpImgType);
  }


  void TiffTestDecode(int num_threads) {
    hybrid_huffman_threshold_ = 0;
    this->SetNumThreads(num_threads);
    this->RunTestDecode(t_tiffImgType);
  }

 private:
  unsigned int hybrid_huffman_threshold_ = std::numeric_limits<unsigned int>::max();
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(nvjpegDecodeDecoupledAPITest, Types);


/***********************************************
**** Default JPEG decode (mix host/hybrid) *****
***********************************************/

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode) {
  this->JpegTestDecode(1, 512u*512u);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode2T) {
  this->JpegTestDecode(2, 512u*512u);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode3T) {
  this->JpegTestDecode(3, 512u*512u);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode4T) {
  this->JpegTestDecode(4, 512u*512u);
}

/***********************************************
******** Host huffman only JPEG decode *********
***********************************************/
// H*W never > threshold so host huffman decoder is always chosen
TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecodeHostHuffman) {
  this->JpegTestDecode(1, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode2THostHuffman) {
  this->JpegTestDecode(2, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode3THostHuffman) {
  this->JpegTestDecode(3, std::numeric_limits<unsigned int>::max());
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode4THostHuffman) {
  this->JpegTestDecode(4, std::numeric_limits<unsigned int>::max());
}

/***********************************************
******* Hybrid huffman only JPEG decode ********
***********************************************/
// H*W always > threshold so hybrid huffman decoder is always chosen
TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecodeHybridHuffman) {
  this->JpegTestDecode(1, 0);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode2THybridHuffman) {
  this->JpegTestDecode(2, 0);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode3THybridHuffman) {
  this->JpegTestDecode(3, 0);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleJPEGDecode4THybridHuffman) {
  this->JpegTestDecode(4, 0);
}

/***********************************************
************* PNG fallback decode **************
***********************************************/
TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSinglePNGDecode) {
  this->PngTestDecode(1);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSinglePNGDecode2T) {
  this->PngTestDecode(2);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSinglePNGDecode3T) {
  this->PngTestDecode(3);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSinglePNGDecode4T) {
  this->PngTestDecode(4);
}

/***********************************************
************* BMP fallback decode **************
***********************************************/
TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleBmpDecode) {
  this->BmpTestDecode(1);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleBmpDecode2T) {
  this->BmpTestDecode(2);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleBmpDecode3T) {
  this->BmpTestDecode(3);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleBmpDecode4T) {
  this->BmpTestDecode(4);
}

/***********************************************
************ TIFF fallback decode **************
***********************************************/
TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleTiffDecode) {
  this->TiffTestDecode(1);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleTiffDecode2T) {
  this->TiffTestDecode(2);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleTiffDecode3T) {
  this->TiffTestDecode(3);
}

TYPED_TEST(nvjpegDecodeDecoupledAPITest, TestSingleTiffDecode4T) {
  this->TiffTestDecode(4);
}

#if NVJPEG_VER_MAJOR >= 11
class HwDecoderUtilizationTest : public ::testing::Test {
 public:
  void SetUp() final {
    dali::string list_root(testing::dali_extra_path() + "/db/single/jpeg");

    pipeline_.AddOperator(
            OpSpec("FileReader")
                    .AddArg("device", "cpu")
                    .AddArg("file_root", list_root)
                    .AddOutput("compressed_images", "cpu")
                    .AddOutput("labels", "cpu"));
    auto decoder_spec =
            OpSpec("ImageDecoder")
                    .AddArg("device", "mixed")
                    .AddArg("output_type", DALI_RGB)
                    .AddArg("hw_decoder_load", .7f)
                    .AddInput("compressed_images", "cpu")
                    .AddOutput("images", "gpu");
    pipeline_.AddOperator(decoder_spec, decoder_name_);

    pipeline_.Build(outputs_);

    auto node = pipeline_.GetOperatorNode(decoder_name_);
    if (!node->op->GetDiagnostic<bool>("using_hw_decoder")) {
      if (nvml::HasCuda11NvmlFunctions()) {
          unsigned int device_count;
          CUDA_CALL(nvmlDeviceGetCount_v2(&device_count));
          for (unsigned int device_idx = 0; device_idx < device_count; device_idx++) {
            auto info = nvml::GetDeviceInfo(device_idx);
            std::cerr << "Device " << device_idx
                      << " brand " << info.type
                      << " cc_M " << info.cap_major
                      << " cc_m " << info.cap_minor
                      << std::endl;
          }
          if (nvml::HasHwDecoder()) {
            FAIL() << "HW Decoder exists in the system and failed to open";
          }
      }
      GTEST_SKIP();
    }
  }


  int batch_size_ = 47;
  Pipeline pipeline_{batch_size_, 1, 0, -1, false, 2, false};
  vector<std::pair<string, string>> outputs_ = {{"images", "gpu"}};
  std::string decoder_name_ = "Lorem Ipsum";
};


TEST_F(HwDecoderUtilizationTest, UtilizationTest) {
  this->pipeline_.RunCPU();
  this->pipeline_.RunGPU();

  auto node = this->pipeline_.GetOperatorNode(this->decoder_name_);
  auto nsamples_hw = node->op->GetDiagnostic<int64_t>("nsamples_hw");
  auto nsamples_cuda = node->op->GetDiagnostic<int64_t>("nsamples_cuda");
  auto nsamples_host = node->op->GetDiagnostic<int64_t>("nsamples_host");
  EXPECT_EQ(nsamples_hw, 35) << "HW Decoder malfunction: incorrect number of images decoded in HW";
  EXPECT_EQ(nsamples_cuda, 12);
  EXPECT_EQ(nsamples_host, 0)
                << "Image decoding malfuntion: all images should've been decoded by CUDA or HW";
}

class HwDecoderMemoryPoolTest : public ::testing::Test {
 public:
  void SetUp() final {
    dali::string list_root(testing::dali_extra_path() + "/db/single/jpeg");

    pipeline_.AddOperator(
            OpSpec("FileReader")
                    .AddArg("device", "cpu")
                    .AddArg("file_root", list_root)
                    .AddOutput("compressed_images", "cpu")
                    .AddOutput("labels", "cpu"));
    auto decoder_spec =
            OpSpec("ImageDecoder")
                    .AddArg("device", "mixed")
                    .AddArg("output_type", DALI_RGB)
                    .AddArg("hw_decoder_load", .7f)
                    .AddArg("preallocate_width_hint", 400)
                    .AddArg("preallocate_height_hint", 600)
                    .AddInput("compressed_images", "cpu")
                    .AddOutput("images", "gpu");
    pipeline_.AddOperator(decoder_spec, decoder_name_);

    pipeline_.Build(outputs_);
  }


  int batch_size_ = 47;
  Pipeline pipeline_{batch_size_, 1, 0, -1, false, 2, false};
  vector<std::pair<string, string>> outputs_ = {{"images", "gpu"}};
  std::string decoder_name_ = "Lorem Ipsum";
};


TEST_F(HwDecoderMemoryPoolTest, MemoryPoolTest) {
  this->pipeline_.RunCPU();
  this->pipeline_.RunGPU();
}
#endif

class Nvjpeg2kTest : public ::testing::Test {
 public:
  void SetUp() final {
    dali::string list_root(testing::dali_extra_path() + "/db/single/jpeg2k");

    pipeline_.AddOperator(
            OpSpec("FileReader")
                    .AddArg("device", "cpu")
                    .AddArg("file_root", list_root)
                    .AddOutput("compressed_images", "cpu")
                    .AddOutput("labels", "cpu"));
    auto decoder_spec =
            OpSpec("ImageDecoder")
                    .AddArg("device", "mixed")
                    .AddArg("output_type", DALI_RGB)
                    .AddInput("compressed_images", "cpu")
                    .AddOutput("images", "gpu");
    pipeline_.AddOperator(decoder_spec, decoder_name_);

    pipeline_.Build(outputs_);
  }


  int batch_size_ = 21;
  Pipeline pipeline_{batch_size_, 1, 0, -1, false, 2, false};
  vector<std::pair<string, string>> outputs_ = {{"images", "gpu"}};
  std::string decoder_name_ = "Lorem Ipsum";
};


TEST_F(Nvjpeg2kTest, UtilizationTest) {
  this->pipeline_.RunCPU();
  this->pipeline_.RunGPU();

  auto node = this->pipeline_.GetOperatorNode(this->decoder_name_);
  auto nsamples_nvjpeg2k = node->op->GetDiagnostic<int64_t>("nsamples_nvjpeg2k");
  auto nsamples_host = node->op->GetDiagnostic<int64_t>("nsamples_host");
#if NVJPEG2K_ENABLED
  std::cout << "Using nvJPEG2k" << std::endl;
  EXPECT_EQ(nsamples_nvjpeg2k, 21);
  EXPECT_EQ(nsamples_host, 0);
#else
  std::cout << "Using CPU fallback" << std::endl;
  EXPECT_EQ(nsamples_nvjpeg2k, 0);
  EXPECT_EQ(nsamples_host, 21);
#endif  // NVJPEG2K_ENABLED
}


}  // namespace dali

