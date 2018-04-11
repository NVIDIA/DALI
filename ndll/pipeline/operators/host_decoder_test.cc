// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/test/ndll_test_single_op.h"

namespace ndll {

template <typename ImgType>
class HostDecodeTest : public NDLLSingleOpTest {
 public:
  USING_NDLL_SINGLE_OP_TEST();

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs) {
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
  const NDLLImageType img_type_ = ImgType::type;
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

}  // namespace ndll
