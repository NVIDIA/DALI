// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dali_test_single_op.h"

namespace dali {

template <typename ImgType>
class HostDecodeTest : public DALISingleOpTest {
 public:
  USING_DALI_SINGLE_OP_TEST();

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs,
            DeviceWorkspace *ws) {
    // single input - encoded images
    // single output - decoded images
    vector<TensorList<CPUBackend>*> outputs(1);

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend>& encoded_data = *inputs[0];

    c_ = (IsColor(img_type_) ? 3 : 1);
    for (int i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = Product(encoded_data.tensor_shape(i));

      cv::Mat tmp = cv::imdecode(cv::Mat(1, data_size,
                                         CV_8UC1,
                                         const_cast<unsigned char*>(data)),
                                 (c_ == 1) ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
      out[i].Resize({tmp.rows, tmp.cols, c_});
      auto *out_data = out[i].mutable_data<unsigned char>();

      std::memcpy(out_data, tmp.ptr(), tmp.rows * tmp.cols * c_);
    }
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }

 protected:
  const DALIImageType img_type_ = ImgType::type;
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(HostDecodeTest, Types);

TYPED_TEST(HostDecodeTest, TestJPEGDecode) {
  TensorList<CPUBackend> encoded_data;
  this->EncodedJPEGData(&encoded_data, this->batch_size_);
  this->SetExternalInputs({std::make_pair("encoded", &encoded_data)});

  this->AddSingleOp(OpSpec("HostDecoder")
              .AddArg("device", "cpu")
              .AddArg("output_type", this->img_type_)
              .AddInput("encoded", "cpu")
              .AddOutput("decoded", "cpu"));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  // Note: lower accuracy due to TJPG and OCV implementations for BGR/RGB.
  // Difference is consistent, deterministic and goes away if I force OCV
  // instead of TJPG decoding.
  this->SetEps(5e-2);

  this->CheckAnswers(&ws, {0});
}

TYPED_TEST(HostDecodeTest, TestPNGDecode) {
  TensorList<CPUBackend> encoded_data;
  this->EncodedPNGData(&encoded_data, this->batch_size_);
  this->SetExternalInputs({std::make_pair("encoded", &encoded_data)});

  this->AddSingleOp(OpSpec("HostDecoder")
              .AddArg("device", "cpu")
              .AddArg("output_type", this->img_type_)
              .AddInput("encoded", "cpu")
              .AddOutput("decoded", "cpu"));

  DeviceWorkspace ws;
  this->RunOperator(&ws);

  this->CheckAnswers(&ws, {0});
}

}  // namespace dali
