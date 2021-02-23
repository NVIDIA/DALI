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

#include "dali/core/tensor_shape.h"
#include "dali/operators/decoder/decoder_test.h"

namespace dali {

template <typename ImgType>
class ImageDecoderSliceTest_GPU : public DecodeTestBase<ImgType> {
 public:
  ImageDecoderSliceTest_GPU() {
  }

 protected:
  TensorList<CPUBackend> begin_data;
  TensorList<CPUBackend> crop_data;

  void AddAdditionalInputs(
    vector<std::pair<string, TensorList<CPUBackend>*>>& inputs) override {
      auto shape = uniform_list_shape(this->batch_size_, {2});

      begin_data.set_type(TypeInfo::Create<float>());
      begin_data.Resize(shape);
      for (int k = 0; k < this->batch_size_; k++) {
        begin_data.mutable_tensor<float>(k)[0] = crop_x;
        begin_data.mutable_tensor<float>(k)[1] = crop_y;
      }

      crop_data.set_type(TypeInfo::Create<float>());
      crop_data.Resize(shape);
      for (int k = 0; k < this->batch_size_; k++) {
        crop_data.mutable_tensor<float>(k)[0] = crop_w;
        crop_data.mutable_tensor<float>(k)[1] = crop_h;
      }

      inputs.push_back(std::make_pair("begin", &begin_data));
      inputs.push_back(std::make_pair("crop", &crop_data));
  }

  OpSpec DecodingOp() const override {
    return this->GetOpSpec("ImageDecoderSlice", "mixed")
      .AddInput("begin", "cpu")
      .AddInput("crop", "cpu");
  }

  CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return [this] (const TensorShape<>& shape,
                   const TensorLayout& shape_layout) {
      DALI_ENFORCE(shape_layout == "HW",
        make_string("Unexpected input shape layout: ", shape_layout, " vs HW"));
      CropWindow crop_window;
      crop_window.anchor[0] = std::lround(crop_y * shape[0]);
      crop_window.anchor[1] = std::lround(crop_x * shape[1]);
      crop_window.shape[0] = std::lround((crop_h + crop_y) * shape[0] - crop_window.anchor[0]);
      crop_window.shape[1] = std::lround((crop_w + crop_x) * shape[1] - crop_window.anchor[1]);
      return crop_window;
    };
  }

  float crop_h = 0.5f, crop_w = 0.25f;
  float crop_y = 0.25f, crop_x = 0.125f;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(ImageDecoderSliceTest_GPU, Types);

TYPED_TEST(ImageDecoderSliceTest_GPU, JpegDecode) {
  this->Run(t_jpegImgType);
}

TYPED_TEST(ImageDecoderSliceTest_GPU, PngDecode) {
  this->Run(t_pngImgType);
}

TYPED_TEST(ImageDecoderSliceTest_GPU, BmpDecode) {
  this->Run(t_bmpImgType);
}

TYPED_TEST(ImageDecoderSliceTest_GPU, TiffDecode) {
  this->Run(t_tiffImgType);
}

TYPED_TEST(ImageDecoderSliceTest_GPU, Jpeg2kDecode) {
  this->Run(t_jpeg2kImgType);
}

}  // namespace dali
