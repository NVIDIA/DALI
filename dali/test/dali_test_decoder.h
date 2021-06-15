// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_DALI_TEST_DECODER_H_
#define DALI_TEST_DALI_TEST_DECODER_H_

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "dali/image/image_factory.h"
#include "dali/test/dali_test_single_op.h"

namespace dali {

template <typename ImgType>
class GenericDecoderTest : public DALISingleOpTest<ImgType> {
 public:
  vector<std::shared_ptr<TensorList<CPUBackend>>> Reference(
      const vector<TensorList<CPUBackend> *> &inputs, DeviceWorkspace *ws) override {
    // single input - encoded images
    // single output - decoded images

     TensorVector<CPUBackend> out(inputs[0]->ntensor());

    const TensorList<CPUBackend> &encoded_data = *inputs[0];

    const int c = this->GetNumColorComp();
    for (size_t i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = volume(encoded_data.tensor_shape(i));

      this->DecodeImage(data, data_size, c, this->ImageType(), &out[i]);
    }

    vector<std::shared_ptr<TensorList<CPUBackend>>> outputs;
    outputs.push_back(std::make_shared<TensorList<CPUBackend>>());
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  virtual OpSpec DecodingOp() const { return OpSpec(); }

  inline uint32_t GetTestCheckType() const override {
    return t_checkColorComp;  // + t_checkElements + t_checkAll + t_checkNoAssert;
  }

  virtual void
    AddAdditionalInputs(
      vector<std::pair<string, TensorList<CPUBackend>*>>&) {}

  void RunTestDecode(t_imgType imageType, float eps = 0.7) {
    TensorList<CPUBackend> encoded_data;
    switch (imageType) {
      case t_jpegImgType:
        this->EncodedJPEGData(&encoded_data);
        break;
      case t_pngImgType:
        this->EncodedPNGData(&encoded_data);
        break;
      case t_tiffImgType:
        this->EncodedTiffData(&encoded_data);
        break;
      case t_bmpImgType:
        this->EncodedBmpData(&encoded_data);
        break;
      case t_jpeg2kImgType:
        this->EncodedJpeg2kData(&encoded_data);
        break;
      default: {
        char buff[32];
        snprintf(buff, sizeof(buff), "%d", imageType);
        DALI_FAIL("Image of type `" + string(buff) + "` cannot be decoded");
      }
    }
    std::vector<std::pair<std::string, TensorList<CPUBackend>*>> inputs{
      std::make_pair("encoded", &encoded_data)};
    AddAdditionalInputs(inputs);
    this->AddExternalInputs(inputs);
    this->RunOperator(DecodingOp(), eps);
  }

  void RunTestDecode(const ImgSetDescr &imgs, float eps = 5e-2) {
    this->SetEps(eps);
    for (size_t imgIdx = 0; imgIdx < imgs.nImages(); ++imgIdx) {
      Tensor<CPUBackend> image;

      auto decoded_image = ImageFactory::CreateImage(
          imgs.data_[imgIdx], imgs.sizes_[imgIdx], this->img_type_);
      decoded_image->Decode();
      const auto shape = decoded_image->GetShape();
      // resize the output tensor
      image.Resize(shape);
      // force allocation
      image.mutable_data<uint8_t>();

      decoded_image->GetImage(image.mutable_data<uint8_t>());

#if DALI_DEBUG
      WriteHWCImage(image.data<uint8_t>(), image.dim(0), image.dim(1),
                    image.dim(2), std::to_string(imgIdx) + "-img");
#endif
      this->VerifyDecode(image.data<uint8_t>(), image.dim(0), image.dim(1),
                         imgs, imgIdx);
    }
  }

  void VerifyDecode(const uint8 *img, int h, int w, const ImgSetDescr &imgs,
                    int img_id) const {
    // Compare w/ opencv result
    const auto imgData = imgs.data_[img_id];
    const auto imgSize = imgs.sizes_[img_id];

    Tensor<CPUBackend> out;
    const int c = this->GetNumColorComp();
    this->DecodeImage(imgData, imgSize, c, this->ImageType(), &out);
    this->CheckBuffers(h * w * c, out.mutable_data<uint8>(), img, false, nullptr, {h, w, c});
  }

  uint32_t GetImageLoadingFlags() const override {
    return t_loadJPEGs | t_loadPNGs | t_loadTiffs | t_loadBmps | t_loadJPEG2ks;
  }
};

}  // namespace dali
#endif  // DALI_TEST_DALI_TEST_DECODER_H_
