// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_DECODER_TEST_H_
#define DALI_OPERATORS_DECODER_DECODER_TEST_H_

#include <string>
#include <vector>
#include <memory>
#include "dali/pipeline/data/types.h"
#include "dali/test/dali_test_decoder.h"

namespace dali {

template <typename ImgType>
class DecodeTestBase : public GenericDecoderTest<ImgType> {
 public:
  void SetUp() override {
    GenericDecoderTest<ImgType>::SetUp();
    this->SetNumThreads(1);
  }

 protected:
  inline void SetImageType(t_imgType image_type) {
    image_type_ = image_type;
  }

  inline virtual CropWindowGenerator GetCropWindowGenerator(int data_idx) const {
    return {};
  }

  inline OpSpec GetOpSpec(const std::string& op_name,
                          const std::string& device = "cpu") const {
    const bool is_mixed = (device == "mixed");
    auto input_device = is_mixed ? StorageDevice::CPU : ParseStorageDevice(device);
    auto output_device = is_mixed ? StorageDevice::GPU : ParseStorageDevice(device);
    return OpSpec(op_name)
      .AddArg("device", device)
      .AddArg("output_type", this->img_type_)
      .AddInput("encoded", input_device)
      .AddOutput("decoded", output_device);
  }

  inline uint32_t GetTestCheckType() const override {
    return t_checkColorComp;  // + t_checkElements + t_checkAll + t_checkNoAssert;
  }

  inline void Run(t_imgType image_type, double eps = 0.75) {
    SetImageType(image_type);
    this->RunTestDecode(image_type_, eps);
  }

  vector<std::shared_ptr<TensorList<CPUBackend>>> Reference(
    const vector<TensorList<CPUBackend> *> &inputs,
    Workspace *ws) override {
    // single input - encoded images
    // single output - decoded images
    TensorList<CPUBackend> out(inputs[0]->num_samples());
    std::vector<Tensor<CPUBackend>> tmp_out(inputs[0]->num_samples());
    const TensorList<CPUBackend> &encoded_data = *inputs[0];
    const int c = this->GetNumColorComp();

    for (int i = 0; i < encoded_data.num_samples(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = volume(encoded_data.tensor_shape(i));
      this->DecodeImage(
        data, data_size, c, this->ImageType(),
        &tmp_out[i], GetCropWindowGenerator(i));
    }

    TensorListShape<> out_shape(inputs[0]->num_samples(), 3);
    for (int i = 0; i < encoded_data.num_samples(); ++i) {
      out_shape.set_tensor_shape(i, tmp_out[i].shape());
    }
    out.SetupLike(tmp_out[0]);
    out.Resize(out_shape, DALI_UINT8);
    for (int i = 0; i < encoded_data.num_samples(); ++i) {
      out.SetSample(i, tmp_out[i]);
    }

    vector<std::shared_ptr<TensorList<CPUBackend>>> outputs;
    outputs.push_back(std::make_shared<TensorList<CPUBackend>>());
    outputs[0]->Copy(out);
    return outputs;
  }

  t_imgType image_type_ = t_jpegImgType;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_DECODER_TEST_H_
