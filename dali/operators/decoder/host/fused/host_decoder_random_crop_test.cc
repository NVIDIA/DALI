// Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/decoder/decoder_test.h"
#include "dali/operators/image/crop/random_crop_attr.h"

namespace dali {

static constexpr int64_t kSeed = 1212334;

template <typename ImgType>
class ImageDecoderRandomCropTest_CPU : public DecodeTestBase<ImgType> {
 public:
  ImageDecoderRandomCropTest_CPU()
    : random_crop_attr(
      OpSpec("RandomCropAttr")
        .AddArg("max_batch_size", this->batch_size_)
        .AddArg("seed", kSeed)) {}

 protected:
  OpSpec DecodingOp() const override {
    return this->GetOpSpec("ImageDecoderRandomCrop")
      .AddArg("seed", kSeed);
  }

  CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return random_crop_attr.GetCropWindowGenerator(data_idx);
  }

  RandomCropAttr random_crop_attr;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(ImageDecoderRandomCropTest_CPU, Types);

TYPED_TEST(ImageDecoderRandomCropTest_CPU, JpegDecode) {
  this->Run(t_jpegImgType);
}

TYPED_TEST(ImageDecoderRandomCropTest_CPU, PngDecode) {
  this->Run(t_pngImgType);
}

TYPED_TEST(ImageDecoderRandomCropTest_CPU, BmpDecode) {
  this->Run(t_bmpImgType);
}

TYPED_TEST(ImageDecoderRandomCropTest_CPU, TiffDecode) {
  this->Run(t_tiffImgType);
}

TYPED_TEST(ImageDecoderRandomCropTest_CPU, Jpeg2kDecode) {
  this->Run(t_jpeg2kImgType);
}

}  // namespace dali
