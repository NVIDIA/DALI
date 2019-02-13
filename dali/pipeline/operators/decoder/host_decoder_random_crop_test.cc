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

#include "dali/pipeline/operators/decoder/host_decoder_test.h"
#include "dali/util/random_crop_generator.h"

namespace dali {

template <typename ImgType>
class HostDecoderRandomCropTest : public HostDecodeTestBase<ImgType> {
 public:
  HostDecoderRandomCropTest()
    : random_crop_generator(
        new RandomCropGenerator(aspect_ratio_range, area_range, seed)) {
  }

 protected:
  const OpSpec DecodingOp() const override {
    return this->GetOpSpec("HostDecoderRandomCrop")
      .AddArg("seed", seed);
  }

  CropWindowGenerator GetCropWindowGenerator() const override {
    return std::bind(
      &RandomCropGenerator::GenerateCropWindow,
      random_crop_generator,
      std::placeholders::_1, std::placeholders::_2);
  }

  int64_t seed = 1212334;
  AspectRatioRange aspect_ratio_range{3.0f/4.0f, 4.0f/3.0f};
  AreaRange area_range{0.08f, 1.0f};
  std::shared_ptr<RandomCropGenerator> random_crop_generator;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(HostDecoderRandomCropTest, Types);

TYPED_TEST(HostDecoderRandomCropTest, JpegDecode) {
  this->Run(t_jpegImgType);
}

TYPED_TEST(HostDecoderRandomCropTest, PngDecode) {
  this->Run(t_pngImgType);
}

TYPED_TEST(HostDecoderRandomCropTest, TiffDecode) {
  this->Run(t_tiffImgType);
}

}  // namespace dali
