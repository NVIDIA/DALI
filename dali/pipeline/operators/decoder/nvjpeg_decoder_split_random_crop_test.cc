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

#include "dali/pipeline/operators/decoder/decoder_test.h"
#include "dali/pipeline/operators/crop/random_crop_attr.h"

namespace dali {

static constexpr int64_t kSeed = 1212334;

template <typename ImgType>
class nvJpegDecoderSplitRandomCropTest : public DecodeTestBase<ImgType> {
 public:
  nvJpegDecoderSplitRandomCropTest()
    : random_crop_attr(
      OpSpec("RandomCropAttr")
        .AddArg("batch_size", this->batch_size_)
        .AddArg("seed", kSeed)) {}

 protected:
  OpSpec DecodingOp() const override {
    return this->GetOpSpec("nvJPEGDecoderRandomCrop", "mixed")
      .AddArg("seed", kSeed)
      .AddArg("split_stages", true);
  }

  CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return random_crop_attr.GetCropWindowGenerator(data_idx);
  }

  RandomCropAttr random_crop_attr;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(nvJpegDecoderSplitRandomCropTest, Types);

TYPED_TEST(nvJpegDecoderSplitRandomCropTest, JpegDecode) {
  this->Run(t_jpegImgType);
}

TYPED_TEST(nvJpegDecoderSplitRandomCropTest, PngDecode) {
  this->Run(t_pngImgType);
}

TYPED_TEST(nvJpegDecoderSplitRandomCropTest, TiffDecode) {
  this->Run(t_tiffImgType);
}

}  // namespace dali
