// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/test/ndll_test_single_op.h"

namespace ndll {

class OCVDecodeTest : public NDLLSingleOpTest {
 public:
  USING_NDLL_SINGLE_OP_TEST();

  vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs) {
    // single input - encoded images
    // single output - decoded images
    vector<TensorList<CPUBackend>*> outputs(1);

    vector<Tensor<CPUBackend>> out(inputs[0]->ntensor());

    const TensorList<CPUBackend>& encoded_data = *inputs[0];

    for (int i = 0; i < encoded_data.ntensor(); ++i) {
      auto *data = encoded_data.tensor<unsigned char>(i);
      auto data_size = Product(encoded_data.tensor_shape(i));

      cv::Mat tmp = cv::imdecode(cv::Mat(1, data_size,
                                         CV_8UC1,
                                         const_cast<unsigned char*>(data)),
                                 CV_LOAD_IMAGE_COLOR);
      out[i].Resize({tmp.rows, tmp.cols, 3});
      auto *out_data = out[i].mutable_data<unsigned char>();

      std::memcpy(out_data, tmp.ptr(), tmp.rows * tmp.cols * 3);
    }
    outputs[0] = new TensorList<CPUBackend>();
    outputs[0]->Copy(out, 0);
    return outputs;
  }
};

TEST_F(OCVDecodeTest, TestDecode) {
  TensorList<CPUBackend> encoded_data;
  EncodedData(&encoded_data);
  SetExternalInputs({std::make_pair("encoded", &encoded_data)});

  AddSingleOp(OpSpec("OCVDecoder")
              .AddArg("device", "cpu")
              .AddArg("output_type", NDLL_RGB)
              .AddInput("encoded", "cpu")
              .AddOutput("decoded", "cpu"));

  DeviceWorkspace ws;
  RunOperator(&ws);

  CheckAnswers(&ws, {0});
}

}  // namespace ndll
