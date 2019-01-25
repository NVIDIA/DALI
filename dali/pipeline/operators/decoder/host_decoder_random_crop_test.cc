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

#include "dali/test/dali_test_decoder.h"

namespace dali {

template <typename ImgType>
class CroppedDecoderTest : public GenericDecoderTest<ImgType> {
 public:
  void SetUp() override {
    GenericDecoderTest<ImgType>::SetUp();
    this->SetNumThreads(1);
  }

  vector<TensorList<CPUBackend> *> Reference(
      const vector<TensorList<CPUBackend> *> &inputs, DeviceWorkspace *ws) override {
    // single input - encoded images
    // single output - decoded images

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend> &encoded_data = *inputs[0];

    const int c = this->GetNumColorComp();
    RandomCropGenerator random_crop_generator(aspect_ratio_range, area_range, seed);
    for (size_t i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = Product(encoded_data.tensor_shape(i));

      this->DecodeImage(data, data_size, c, this->ImageType(), &out[i],
        nullptr, &random_crop_generator);
    }

    vector<TensorList<CPUBackend> *> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

  int64_t seed = 1212334;
  AspectRatioRange aspect_ratio_range{3.0f/4.0f, 4.0f/3.0f};
  AreaRange area_range{0.08f, 1.0f};
};

template <typename ImgType>
class HostDecodeRandomCropTest : public CroppedDecoderTest<ImgType> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_loadJPEGs + t_loadPNGs;
  }

  const OpSpec DecodingOp() const override {
    return OpSpec("HostDecoderRandomCrop")
      .AddArg("device", "cpu")
      .AddArg("output_type", this->img_type_)
      .AddArg("seed", 1212334)
      .AddInput("encoded", "cpu")
      .AddOutput("decoded", "cpu");
  }

  uint32_t GetTestCheckType() const  override {
    return t_checkColorComp + t_checkElements;  // + t_checkAll + t_checkNoAssert;
  }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(HostDecodeRandomCropTest, Types);


template<typename ImageType>
class HostDecodeRandomCropTestJpeg : public HostDecodeRandomCropTest<ImageType> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_jpegImgType;
  }
};

TYPED_TEST_CASE(HostDecodeRandomCropTestJpeg, Types);

TYPED_TEST(HostDecodeRandomCropTestJpeg, TestJpegDecode) {
  this->RunTestDecode(t_jpegImgType, 0.7);
}


template<typename ImageType>
class HostDecodeRandomCropTestPng : public HostDecodeRandomCropTest<ImageType> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_pngImgType;
  }
};

TYPED_TEST_CASE(HostDecodeRandomCropTestPng, Types);

TYPED_TEST(HostDecodeRandomCropTestPng, TestPngDecode) {
  this->RunTestDecode(t_pngImgType, 0.75);
}


template<typename ImageType>
class HostDecodeRandomCropTestTiff : public HostDecodeRandomCropTest<ImageType> {
 protected:
  uint32_t GetImageLoadingFlags() const override {
    return t_tiffImgType;
  }
};

TYPED_TEST_CASE(HostDecodeRandomCropTestTiff, Types);

TYPED_TEST(HostDecodeRandomCropTestTiff, TestTiffDecode) {
  this->RunTestDecode(t_tiffImgType, 0.75);
}
}  // namespace dali
