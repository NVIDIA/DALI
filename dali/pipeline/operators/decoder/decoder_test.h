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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_DECODER_TEST_H_
#define DALI_PIPELINE_OPERATORS_DECODER_DECODER_TEST_H_

#include <string>
#include <vector>
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
    return OpSpec(op_name)
      .AddArg("device", device)
      .AddArg("output_type", this->img_type_)
      .AddInput("encoded", is_mixed ? "cpu" : device)
      .AddOutput("decoded", is_mixed ? "gpu" : device);
  }

  inline uint32_t GetTestCheckType() const override {
    return t_checkColorComp;  // + t_checkElements + t_checkAll + t_checkNoAssert;
  }

  inline void Run(t_imgType image_type, double eps = 0.75) {
    SetImageType(image_type);
    this->RunTestDecode(image_type_, eps);
  }

  vector<TensorList<CPUBackend> *> Reference(
    const vector<TensorList<CPUBackend> *> &inputs,
    DeviceWorkspace *ws) override {
    // single input - encoded images
    // single output - decoded images
    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());
    const TensorList<CPUBackend> &encoded_data = *inputs[0];
    const int c = this->GetNumColorComp();

    for (size_t i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = volume(encoded_data.tensor_shape(i));
      this->DecodeImage(
        data, data_size, c, this->ImageType(),
        &out[i], GetCropWindowGenerator(i));
    }

    vector<TensorList<CPUBackend> *> outputs(1);
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

  t_imgType image_type_ = t_jpegImgType;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_DECODER_TEST_H_
