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

#include <cmath>
#include "dali/operators/decoder/decoder_test.h"

namespace dali {

template <typename ImgType>
class ImageDecoderCropTest_GPU : public DecodeTestBase<ImgType> {
 protected:
  OpSpec DecodingOp() const override {
    return this->GetOpSpec("ImageDecoderCrop", "mixed")
      .AddArg("crop", std::vector<float>{1.0f*crop_H, 1.0f*crop_W});
  }

  CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return [this] (const TensorShape<>& shape,
                   const TensorLayout& shape_layout) {
      DALI_ENFORCE(shape_layout == "HW",
        make_string("Unexpected input shape layout: ", shape_layout, " vs HW"));
      CropWindow crop_window;
      crop_window.shape[0] = crop_H;
      crop_window.shape[1] = crop_W;
      crop_window.anchor[0] = std::round(0.5f * (shape[0] - crop_window.shape[0]));
      crop_window.anchor[1] = std::round(0.5f * (shape[1] - crop_window.shape[1]));
      return crop_window;
    };
  }

  int crop_H = 224, crop_W = 200;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(ImageDecoderCropTest_GPU, Types);

TYPED_TEST(ImageDecoderCropTest_GPU, JpegDecode) {
  this->Run(t_jpegImgType);
}

TYPED_TEST(ImageDecoderCropTest_GPU, PngDecode) {
  this->Run(t_pngImgType);
}

TYPED_TEST(ImageDecoderCropTest_GPU, BmpDecode) {
  this->Run(t_bmpImgType);
}

TYPED_TEST(ImageDecoderCropTest_GPU, TiffDecode) {
  this->crop_H = 100;
  this->crop_W = 90;
  this->Run(t_tiffImgType);
}

TYPED_TEST(ImageDecoderCropTest_GPU, Jpeg2kDecode) {
  this->Run(t_jpeg2kImgType);
}

}  // namespace dali
