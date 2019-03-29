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

namespace dali {

template <typename ImgType>
class HostDecoderSliceTest : public DecodeTestBase<ImgType> {
 protected:
  TensorList<CPUBackend> begin_data;
  TensorList<CPUBackend> crop_data;

  void AddAdditionalInputs(
    vector<std::pair<string, TensorList<CPUBackend>*>>& inputs) override {
      vector<Dims> shape(this->batch_size_, {2});

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
    return this->GetOpSpec("HostDecoderSlice")
      .AddInput("begin", "cpu")
      .AddInput("crop", "cpu");
  }

  CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return [this] (int H, int W) {
      CropWindow crop_window;
      crop_window.y = crop_y * H;
      crop_window.x = crop_x * W;
      crop_window.h = (crop_h + crop_y) * H - crop_window.y;
      crop_window.w = (crop_w + crop_x) * W - crop_window.x;
      return crop_window;
    };
  }

  float crop_h = 0.5f, crop_w = 0.25f;
  float crop_y = 0.25f, crop_x = 0.125f;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_SUITE(HostDecoderSliceTest, Types);

TYPED_TEST(HostDecoderSliceTest, JpegDecode) {
  this->Run(t_jpegImgType);
}

TYPED_TEST(HostDecoderSliceTest, PngDecode) {
  this->Run(t_pngImgType);
}

TYPED_TEST(HostDecoderSliceTest, TiffDecode) {
  this->Run(t_tiffImgType);
}

}  // namespace dali
